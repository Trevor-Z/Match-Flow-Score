import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import os
import pandas as pd
from tqdm import tqdm
import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gc
import re
from pathlib import Path

# ---------------------- USER CONFIG ----------------------
BASE_DIR = Path(__file__).resolve().parent
VIDEO_FOLDER = ''
CSV_FILE = ''
MODEL_PATH = str((BASE_DIR / "model.pth").resolve())

# Round start frames
ROUND1_START = 950
ROUND2_START = 2400
ROUND3_OFFSET_AFTER_R2 = 2300     # When r2 is found, search r3 from (r2 + this)
ROUND3_EARLY_FALLBACK = 500       # Try 500 frames earlier if round 3 not found

MATCH_CLASS = 0                   # "FIGHT" class id in classifier
CONFIDENCE_THRESHOLD = 0.998       # baseline acceptance threshold

# ROI + model input size
ROI_X = 524
ROI_Y = 365
ROI_WIDTH = 176
ROI_HEIGHT = 352
RESIZE_WIDTH = 44
RESIZE_HEIGHT = 88

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_CSV = str((BASE_DIR / "output" / "fight_frames.csv").resolve())
VIDEO_EXTENSIONS = ('.mkv', '.mp4', '.avi', '.mov')

SAVE_MATCH_FRAMES = False
OUTPUT_FRAMES_FOLDER = ''

# -------- Adjacent-frame confirmation (stability check) --------
ADJACENT_CONFIRM_ENABLED = True         # True to enable the extra check below
ADJACENT_CONFIRM_DIRECTION = 'either'   # 'next' | 'prev' | 'either'  <-- NEW OPTION
ADJACENT_CONFIRM_N_FRAMES = 1           # how many frames to check in that/those direction(s)
ADJACENT_CONFIRM_CONFIDENCE = 0.99      # min confidence for those frames
ADJACENT_CONFIRM_STRATEGY = 'any'       # 'any' (≥1 passes) or 'all' (all must pass)
# ----------------------------------------------------------------

# ---------------------- PERFORMANCE TUNING ----------------------
BATCH_SIZE = 128
FRAME_SKIP = 1
NUM_WORKERS = 6
USE_MIXED_PRECISION = True
MEMORY_EFFICIENT = True
# ---------------------------------------------------------------

def parse_rounds_from_filename(filename):
    base_name = os.path.splitext(filename)[0]
    match = re.search(r'_([23])r$', base_name)
    if match:
        return int(match.group(1))
    if re.search(r'3r', base_name, re.IGNORECASE):
        return 3
    elif re.search(r'2r', base_name, re.IGNORECASE):
        return 2
    return 2

def obter_videos_para_processar(pasta_videos, caminho_csv=None, extensoes_video=VIDEO_EXTENSIONS):
    videos_pasta = {f: None for f in os.listdir(pasta_videos) if f.endswith(extensoes_video)}
    if caminho_csv is None:
        print(f"Processando todos os {len(videos_pasta)} vídeos da pasta.")
        return videos_pasta
    try:
        df = pd.read_csv(caminho_csv, header=None, usecols=[0], keep_default_na=False)
        df.columns = ['filename']
    except Exception as e:
        print(f"[Aviso] Falha ao ler CSV ({e}). Processando todos os vídeos da pasta.")
        return videos_pasta

    base_map = {os.path.splitext(f)[0]: f for f in videos_pasta}
    out = {}
    missing = []
    for _, row in df.iterrows():
        base = os.path.splitext(row['filename'])[0]
        if base in base_map:
            out[base_map[base]] = None
        else:
            missing.append(row['filename'])
    if missing:
        print(f"[Aviso] {len(missing)} arquivos listados no CSV não encontrados na pasta (ex.: {missing[:3]})")
    print(f"Vídeos do CSV prontos: {len(out)}")
    return out

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.squeeze.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.expand1x1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.expand3x3.weight, mode='fan_out', nonlinearity='relu')
        if self.squeeze.bias is not None: nn.init.zeros_(self.squeeze.bias)
        if self.expand1x1.bias is not None: nn.init.zeros_(self.expand1x1.bias)
        if self.expand3x3.bias is not None: nn.init.zeros_(self.expand3x3.bias)

    def forward(self, x):
        x = F.relu(self.squeeze(x), inplace=True)
        return torch.cat([F.relu(self.expand1x1(x), inplace=True),
                          F.relu(self.expand3x3(x), inplace=True)], 1)

class CustomSqueezeNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            FireModule(64, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            FireModule(128, 32, 128, 128),
            FireModule(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not isinstance(m, FireModule):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def save_frame(frame, output_path):
    cv2.imwrite(output_path, frame)

def _format_conf(c: float) -> str:
    try:
        return f"{float(c):.3f}"
    except Exception:
        return "nan"

class FrameProcessor:
    def __init__(self, transform, device): self.transform, self.device = transform, device
    def extract_and_process_roi(self, frame):
        try:
            roi = frame[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
            if roi.shape[0] == 0 or roi.shape[1] == 0: return None
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            return self.transform(roi_pil)
        except Exception:
            return None

def load_frames_batch(cap, start_frame, batch_size, skip_frames=1):
    frames, frame_numbers = [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(batch_size):
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
        frame_numbers.append(start_frame + i * skip_frames)
        if skip_frames > 1:
            for _ in range(skip_frames - 1):
                ret, _ = cap.read()
                if not ret: break
    return frames, frame_numbers

def process_batch_inference(model, roi_tensors, device, use_mixed_precision=True):
    if not roi_tensors: return [], []
    batch_tensor = torch.stack(roi_tensors).to(device, non_blocking=True)
    with torch.no_grad():
        if use_mixed_precision and device == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(batch_tensor)
                probs = F.softmax(outputs, dim=1)
        else:
            probs = F.softmax(model(batch_tensor), dim=1)
        confs, pred = torch.max(probs, 1)
    return pred.cpu().numpy(), confs.cpu().numpy()

def _read_specific_frames(video_path, indices):
    cap = cv2.VideoCapture(video_path)
    out = []
    for idx in indices:
        if idx < 0: 
            out.append(None)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        out.append(frame if ret else None)
    cap.release()
    return out

def _infer_on_raw_frames(frames, transform, model, device):
    processor = FrameProcessor(transform, device)
    roi_tensors, valid_idx = [], []
    for i, fr in enumerate(frames):
        if fr is None: 
            continue
        t = processor.extract_and_process_roi(fr)
        if t is not None:
            roi_tensors.append(t); valid_idx.append(i)
    preds = [None]*len(frames)
    confs = [None]*len(frames)
    if roi_tensors:
        p, c = process_batch_inference(model, roi_tensors, device, USE_MIXED_PRECISION)
        for slot, (pp, cc) in zip(valid_idx, zip(p, c)):
            preds[slot] = int(pp)
            confs[slot] = float(cc)
    return preds, confs

def confirm_adjacent(video_path, transform, model, candidate_frame, direction='next',
                     n=3, device='cuda', threshold=0.98, strategy='any'):
    """
    Confirm candidate by checking adjacent frames.
    direction: 'next' | 'prev' | 'either'
    n: number of frames to test per direction
    threshold: min confidence for MATCH_CLASS on each frame
    strategy: 'any' (≥1 frame passes) or 'all' (all n frames must pass)
    Semantics for 'either': pass if (prev passes) OR (next passes), where each side
    is evaluated with the given 'strategy'.
    """
    if n <= 0:
        return True

    def _check_dir(dir_name: str) -> bool:
        if dir_name == 'next':
            indices = [candidate_frame + i for i in range(1, n+1)]
        else:  # 'prev'
            indices = [candidate_frame - i for i in range(1, n+1)]

        raw_frames = _read_specific_frames(video_path, indices)
        preds, confs = _infer_on_raw_frames(raw_frames, transform, model, device)

        flags = []
        for pr, cf in zip(preds, confs):
            ok = (pr == MATCH_CLASS) and (cf is not None) and (cf >= threshold)
            flags.append(bool(ok))

        if strategy == 'all':
            return all(flags) if flags else False
        else:  # 'any'
            return any(flags)

    if direction == 'next':
        return _check_dir('next')
    elif direction == 'prev':
        return _check_dir('prev')
    elif direction == 'either':  # NEW
        return _check_dir('prev') or _check_dir('next')
    else:
        # Fallback to 'next' if misconfigured
        return _check_dir('next')

def infer_video_optimized(video_path, model, transform, start_frame=0, device='cuda',
                          use_adjacent_confirm=False, confirm_direction='next',
                          confirm_n=3, confirm_threshold=0.98, confirm_strategy='any'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir: {video_path}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame >= total:
        cap.release(); return None

    processor = FrameProcessor(transform, device)
    current_frame = start_frame
    pbar = tqdm(total=max(0, (total - start_frame)//FRAME_SKIP),
                desc=f"{os.path.basename(video_path)} @ {start_frame}")

    while current_frame < total:
        frames, frame_numbers = load_frames_batch(cap, current_frame, BATCH_SIZE, FRAME_SKIP)
        if not frames: break

        roi_tensors, valid_idx = [], []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {ex.submit(processor.extract_and_process_roi, fr): i for i, fr in enumerate(frames)}
            for fut, i in zip(futures.keys(), futures.values()):
                t = fut.result()
                if t is not None:
                    roi_tensors.append(t); valid_idx.append(i)

        if roi_tensors:
            pred, conf = process_batch_inference(model, roi_tensors, device, USE_MIXED_PRECISION)
            for i, (c, p) in enumerate(zip(conf, pred)):
                if p == MATCH_CLASS and c >= CONFIDENCE_THRESHOLD:
                    orig = valid_idx[i]
                    frame_num = frame_numbers[orig]
                    matched_frame = frames[orig]

                    if use_adjacent_confirm:
                        ok = confirm_adjacent(
                            video_path, transform, model, frame_num,
                            direction=confirm_direction,
                            n=confirm_n, device=device,
                            threshold=confirm_threshold, strategy=confirm_strategy
                        )
                        if not ok:
                            continue

                    cap.release(); pbar.close()
                    return frame_num, float(c), matched_frame

        pbar.update(len(frames))
        current_frame += len(frames) * FRAME_SKIP
        if MEMORY_EFFICIENT and device == 'cuda':
            torch.cuda.empty_cache()
    cap.release(); pbar.close()
    return None

def optimize_gpu_settings():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")

def main(video_folder, model_path, device=DEVICE, output_csv=OUTPUT_CSV,
         video_extensions=VIDEO_EXTENSIONS, save_frames=SAVE_MATCH_FRAMES,
         frames_folder=OUTPUT_FRAMES_FOLDER, csv_file=CSV_FILE):

    optimize_gpu_settings()

    videos = obter_videos_para_processar(video_folder, csv_file, video_extensions)
    if not videos:
        print("Nenhum vídeo encontrado."); return

    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    model = CustomSqueezeNet(num_classes=2).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    if device == 'cuda':
        try: model = torch.jit.script(model)
        except Exception: pass

    transform = transforms.Compose([
        transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if save_frames: os.makedirs(frames_folder, exist_ok=True)
    if os.path.exists(output_csv):
        base, ext = os.path.splitext(output_csv)
        output_csv = f"{base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"

    rows = []
    t0 = datetime.datetime.now()

    for idx, (video_file, _) in enumerate(videos.items(), 1):
        vp = os.path.join(video_folder, video_file)
        num_rounds = parse_rounds_from_filename(video_file)
        print(f"\n[{idx}/{len(videos)}] {video_file} ({num_rounds} rounds)")

        r1 = infer_video_optimized(
            vp, model, transform, start_frame=ROUND1_START, device=device,
            use_adjacent_confirm=ADJACENT_CONFIRM_ENABLED,
            confirm_direction=ADJACENT_CONFIRM_DIRECTION,
            confirm_n=ADJACENT_CONFIRM_N_FRAMES,
            confirm_threshold=ADJACENT_CONFIRM_CONFIDENCE,
            confirm_strategy=ADJACENT_CONFIRM_STRATEGY
        )
        r1_frame = r1[0] if r1 else None
        r1_conf = r1[1] if r1 else None
        if r1 and save_frames:
            out = os.path.join(
                frames_folder,
                f"{os.path.splitext(video_file)[0]}_r1_{r1_frame}_c{_format_conf(r1_conf)}.jpg"
            )
            save_frame(r1[2], out)

        r2 = infer_video_optimized(
            vp, model, transform, start_frame=ROUND2_START, device=device,
            use_adjacent_confirm=ADJACENT_CONFIRM_ENABLED,
            confirm_direction=ADJACENT_CONFIRM_DIRECTION,
            confirm_n=ADJACENT_CONFIRM_N_FRAMES,
            confirm_threshold=ADJACENT_CONFIRM_CONFIDENCE,
            confirm_strategy=ADJACENT_CONFIRM_STRATEGY
        )
        r2_frame = r2[0] if r2 else None
        r2_conf = r2[1] if r2 else None
        if r2 and save_frames:
            out = os.path.join(
                frames_folder,
                f"{os.path.splitext(video_file)[0]}_r2_{r2_frame}_c{_format_conf(r2_conf)}.jpg"
            )
            save_frame(r2[2], out)

        r3_frame = None
        r3_conf = None
        if num_rounds == 3:
            if r2_frame is not None:
                r3_start = r2_frame + ROUND3_OFFSET_AFTER_R2
                r3 = infer_video_optimized(
                    vp, model, transform, start_frame=r3_start, device=device,
                    use_adjacent_confirm=ADJACENT_CONFIRM_ENABLED,
                    confirm_direction=ADJACENT_CONFIRM_DIRECTION,
                    confirm_n=ADJACENT_CONFIRM_N_FRAMES,
                    confirm_threshold=ADJACENT_CONFIRM_CONFIDENCE,
                    confirm_strategy=ADJACENT_CONFIRM_STRATEGY
                )
                if r3 is None:
                    print(f"Round 3 not found at {r3_start}, trying {ROUND3_EARLY_FALLBACK} frames earlier...")
                    r3_fallback_start = max(0, r3_start - ROUND3_EARLY_FALLBACK)
                    r3 = infer_video_optimized(
                        vp, model, transform, start_frame=r3_fallback_start, device=device,
                        use_adjacent_confirm=ADJACENT_CONFIRM_ENABLED,
                        confirm_direction=ADJACENT_CONFIRM_DIRECTION,
                        confirm_n=ADJACENT_CONFIRM_N_FRAMES,
                        confirm_threshold=ADJACENT_CONFIRM_CONFIDENCE,
                        confirm_strategy=ADJACENT_CONFIRM_STRATEGY
                    )
                if r3:
                    r3_frame = r3[0]
                    r3_conf = r3[1]
                    if save_frames:
                        out = os.path.join(
                            frames_folder,
                            f"{os.path.splitext(video_file)[0]}_r3_{r3_frame}_c{_format_conf(r3_conf)}.jpg"
                        )
                        save_frame(r3[2], out)
            else:
                print(f"Cannot search for Round 3: Round 2 not found in {video_file}")

        rows.append({
            'filename': video_file,
            'frame_r1': r1_frame if r1_frame is not None else '',
            'Round 1': '',
            'frame_r2': r2_frame if r2_frame is not None else '',
            'Round 2': '',
            'frame_r3': r3_frame if r3_frame is not None else '',
            'Round 3': ''
        })

        if device == 'cuda':
            torch.cuda.empty_cache(); gc.collect()

    df = pd.DataFrame(rows, columns=[
        'filename', 'frame_r1', 'Round 1', 'frame_r2', 'Round 2', 'frame_r3', 'Round 3'
    ])
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("\n" + "="*60)
    print(f"Concluído. Saída: {output_csv}")
    print(f"Tempo total: {(datetime.datetime.now() - t0).total_seconds():.2f}s")

if __name__ == "__main__":
    print(f"Device: {DEVICE} | Threshold: {CONFIDENCE_THRESHOLD}")
    main(VIDEO_FOLDER, MODEL_PATH)
