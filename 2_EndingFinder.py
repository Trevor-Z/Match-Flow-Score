from __future__ import annotations

import cv2
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# --------------------------- DEFAULT CONFIG --------------------------- #

BASE_DIR = Path(__file__).resolve().parent
DIGIT_MODEL = BASE_DIR / "model.tflite"

REGION_CONFIGS: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {
    "P1": {
        # (x, y, w, h)
        "round_end_check_region": (67, 233, 12, 22),
    },
    "P2": {
        "round_end_check_region": (1854, 233, 12, 22),
    },
}

DEFAULT_TAIL_CONSECUTIVE_NONE = 10
COARSE_SCAN_STEP = 10


# --------------------------- IMPLEMENTATION --------------------------- #


_digit_logger = logging.getLogger("RoundEndFinder.DigitClassifier")


class BaseClassifier(ABC):
    def __init__(
        self,
        model_path: str,
        img_width: int,
        img_height: int,
        color_mode: str,
        class_names: List[str],
    ):
        self.model_path = model_path
        self.img_width = img_width
        self.img_height = img_height
        self.color_mode = color_mode
        self.class_names = class_names
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self._validate_inputs()
        self._load_model()

    def _validate_inputs(self) -> None:
        if not self.model_path:
            raise ValueError("Model path cannot be empty")
        if self.img_width <= 0 or self.img_height <= 0:
            raise ValueError("Image dimensions must be positive")
        if self.color_mode not in {"rgb", "grayscale"}:
            raise ValueError("Color mode must be 'rgb' or 'grayscale'")
        if not self.class_names:
            raise ValueError("Class names cannot be empty")

    def _load_model(self) -> None:
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file does not exist: {self.model_path}")
            _digit_logger.info("Loading TFLite model from %s", self.model_path)
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            num_classes = self.output_details[0]["shape"][-1]
            if num_classes != len(self.class_names):
                _digit_logger.warning(
                    "Model reports %s classes but %s class names provided",
                    num_classes,
                    len(self.class_names),
                )
        except Exception as exc:
            _digit_logger.error("Error loading TFLite model: %s", exc)
            raise

    @abstractmethod
    def _preprocess_image(self, image_crop: np.ndarray) -> np.ndarray:
        ...

    def _infer(self, img_array: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input_details[0]["index"], img_array)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])

    def _post_process_predictions(self, class_name: str) -> str:
        return class_name

    def classify(self, image_crop: np.ndarray, return_top_k: int = 2) -> Tuple:
        try:
            if image_crop is None or image_crop.size == 0:
                raise ValueError("Invalid image crop")

            processed = self._preprocess_image(image_crop)
            prediction = self._infer(processed)

            top_indices = np.argsort(prediction[0])[::-1][:return_top_k]
            results = [
                (
                    self._post_process_predictions(self.class_names[i]),
                    float(prediction[0][i]),
                )
                for i in top_indices
            ]

            while len(results) < return_top_k:
                results.append(results[-1] if results else ("unknown", 0.0))

            flat = [v for pair in results for v in pair]
            return tuple(flat[: return_top_k * 2])
        except Exception as exc:
            _digit_logger.error("Error in classification: %s", exc)
            return tuple(["error", 0.0] * return_top_k)


class DigitClassifier(BaseClassifier):
    def __init__(
        self,
        model_path: str,
        img_width: int = 12,
        img_height: int = 22,
        color_mode: str = "rgb",
    ):
        super().__init__(
            model_path,
            img_width,
            img_height,
            color_mode,
            [str(i) for i in range(10)] + ["10"],
        )

    def _preprocess_image(self, image_crop: np.ndarray) -> np.ndarray:
        image = cv2.resize(image_crop, (self.img_width, self.img_height))
        if self.color_mode == "grayscale":
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[..., np.newaxis]
        else:
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        return image[np.newaxis, ...]

    def _post_process_predictions(self, class_name: str) -> str:
        return "none" if class_name == "10" else class_name


def create_digit_classifier(
    model_path: str, img_width: int = 12, img_height: int = 22, color_mode: str = "rgb"
) -> DigitClassifier:
    return DigitClassifier(model_path, img_width, img_height, color_mode)


def classify_image_region(
    frame: np.ndarray,
    classifier: BaseClassifier,
    region: Tuple[int, int, int, int],
    return_top_k: int = 2,
) -> Tuple:
    try:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame")
        if len(region) != 4:
            raise ValueError("Region must be a tuple of four values")

        x, y, w, h = region
        h_frame, w_frame = frame.shape[:2]

        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)

        if w <= 0 or h <= 0:
            raise ValueError("Region dimensions are invalid")

        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            raise ValueError("Extracted ROI is empty")

        return classifier.classify(roi, return_top_k)
    except Exception as exc:
        _digit_logger.error("Error in classify_image_region: %s", exc)
        return tuple(["error", 0.0] * return_top_k)


logger = logging.getLogger("RoundEndFinder")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RoundEndFinder:
    """
    Detects the last active frame of a round by monitoring a HUD digit until it
    shows 'none' for a configured tail length.
    """

    def __init__(
        self,
        digit_model_path: Union[str, Path] = DIGIT_MODEL,
        region_configs: Dict[str, Dict[str, Tuple[int, int, int, int]]] = REGION_CONFIGS,
        tail_consecutive_none: int = DEFAULT_TAIL_CONSECUTIVE_NONE,
        coarse_scan_step: int = COARSE_SCAN_STEP,
        debug_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        digit_model_path = Path(digit_model_path)
        if not digit_model_path.is_absolute():
            digit_model_path = BASE_DIR / digit_model_path
        if not digit_model_path.exists():
            raise FileNotFoundError(f"Digit model not found: {digit_model_path}")

        self.digit_clf = create_digit_classifier(str(digit_model_path))
        self.region_configs = region_configs
        self.tail_none = max(1, int(tail_consecutive_none))
        self.coarse_step = max(1, int(coarse_scan_step))
        self.debug_dir = None
        if debug_dir is not None:
            dbg_path = Path(debug_dir)
            if not dbg_path.is_absolute():
                dbg_path = BASE_DIR / dbg_path
            self.debug_dir = dbg_path
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _probe_end_digit(self, frame, side: str) -> str:
        side = side.upper()
        if side not in self.region_configs:
            raise ValueError(f"Unknown side '{side}'. Expected one of {tuple(self.region_configs.keys())}.")
        region = self.region_configs[side]["round_end_check_region"]
        digit, _, _, _ = classify_image_region(frame, self.digit_clf, region)
        return str(digit)

    def _fine_scan(
        self,
        video_path: Path,
        start_frame: int,
        limit: int,
        side: str,
        fallback_frame: int,
    ) -> int:
        cap_fine = cv2.VideoCapture(str(video_path))
        if not cap_fine.isOpened():
            logger.error("Cannot reopen video for fine scan: %s", video_path)
            return fallback_frame

        try:
            cap_fine.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            round_active = False
            consecutive_none = 0
            first_none_frame: Optional[int] = None
            last_non_none = fallback_frame
            last_non_none_frame_img = None
            fine_total = max(1, limit - start_frame + 1)
            fine_bar = tqdm(total=fine_total, desc=f"RoundEnd fine {side}", unit="frame", leave=False)

            while frame_idx <= limit:
                ret, frame = cap_fine.read()
                if not ret:
                    break

                fine_bar.update(1)
                digit = self._probe_end_digit(frame, side)

                if digit != "none":
                    round_active = True
                    consecutive_none = 0
                    first_none_frame = None
                    last_non_none = frame_idx
                    last_non_none_frame_img = frame.copy()
                else:
                    if round_active:
                        if first_none_frame is None:
                            first_none_frame = frame_idx
                        consecutive_none += 1
                        if consecutive_none >= self.tail_none:
                            fine_bar.close()
                            candidate = first_none_frame - 1
                            if candidate < start_frame:
                                candidate = fallback_frame
                            if self.debug_dir and last_non_none_frame_img is not None:
                                out_name = f"{video_path.stem}_round_end_{last_non_none}_{side}.png"
                                cv2.imwrite(str(self.debug_dir / out_name), last_non_none_frame_img)
                                for offset in (1, 2):
                                    next_idx = last_non_none + offset
                                    if next_idx > limit:
                                        break
                                    cap_next = cv2.VideoCapture(str(video_path))
                                    if not cap_next.isOpened():
                                        break
                                    try:
                                        cap_next.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
                                        ret_next, frame_next = cap_next.read()
                                        if not ret_next:
                                            break
                                        next_name = f"{video_path.stem}_round_end_{next_idx}_{side}_+{offset}.png"
                                        cv2.imwrite(str(self.debug_dir / next_name), frame_next)
                                    finally:
                                        cap_next.release()
                            return candidate
                    else:
                        # still before round actually started
                        consecutive_none = 0
                        first_none_frame = None

                frame_idx += 1

            fine_bar.close()
            return last_non_none
        finally:
            cap_fine.release()

    def _search_side(
        self,
        video_path: Path,
        start_frame: int,
        max_frames: int,
        side: str,
    ) -> Optional[int]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return None

        try:
            limit = start_frame + max(1, int(max_frames))
            coarse_step = max(1, int(self.coarse_step))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            last_non_none = start_frame
            round_active = False

            total_frames = max(1, limit - start_frame + 1)
            progress = tqdm(total=total_frames, desc=f"RoundEnd {side}", unit="frame", leave=False)

            while frame_idx <= limit:
                sample_idx = frame_idx
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached at frame %s", frame_idx)
                    break

                progress.update(1)
                digit = self._probe_end_digit(frame, side)

                if digit != "none":
                    round_active = True
                    last_non_none = sample_idx
                else:
                    if round_active:
                        progress.close()
                        window_start = sample_idx - (2 * coarse_step) + 1
                        fine_start = max(start_frame, window_start)
                        return self._fine_scan(
                            video_path,
                            fine_start,
                            limit,
                            side,
                            last_non_none,
                        )

                frame_idx += 1

                skip = max(0, min(coarse_step - 1, limit - frame_idx + 1))
                for _ in range(skip):
                    if not cap.grab():
                        break
                    progress.update(1)
                    frame_idx += 1

            progress.close()

            if not round_active:
                logger.warning("Round end probe never activated for %s (%s)", video_path, side)
                return None
            return last_non_none
        finally:
            cap.release()

    def find_round_end(
        self,
        video_path: Union[str, Path],
        start_frame: int,
        max_frames: int,
        side: str = "auto",
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Returns (end_frame, resolved_side). If side == 'auto', tests P1 then P2.
        """
        video_path = Path(video_path)
        side = side.upper()
        if side not in ("P1", "P2", "AUTO"):
            raise ValueError("side must be 'P1', 'P2', or 'auto'")

        sides = ("P1", "P2") if side == "AUTO" else (side,)
        for candidate in sides:
            end_frame = self._search_side(video_path, start_frame, max_frames, candidate)
            if end_frame is not None:
                return end_frame, candidate
        return None, None


__all__ = [
    "RoundEndFinder",
    "DIGIT_MODEL",
    "REGION_CONFIGS",
    "DEFAULT_TAIL_CONSECUTIVE_NONE",
    "COARSE_SCAN_STEP",
]
