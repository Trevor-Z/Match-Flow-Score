From SF6 replay video data, we compute a 100-point Match Flow Score showing how even (a win with few points) or lopsided (a win with many points) a round was. Or even whether the losing side actually played better and only lost because it was a very close call (a win with negative points).

The information used: round duration, winner’s health at KO (or margin at timeout), time each player went without being hit, and the number of times each was hit.

These become four weighted terms (see match_flow_score_from_series in Match_Flow_Score_tool.py):

duration_term: Rewards quick wins. Take the round length, subtract it from the full 100-second timer, and square what’s left. Short rounds push the score up, drawn-out rounds push it down.

health_term: Rewards finishing with a health lead. Measure the winner’s final health minus the loser’s, and square it to emphasize big gaps.

streak_term: Rewards good defense. Find the longest damage-free streak for each side, convert those to fractions of the round, and take the difference.

tempo_term: Rewards offensive pressure. Count how many times each side’s health dropped, the side that scored more hits gets credit.

Their sum is the final match flow score. Default weights are 0.35, 0.35, 0.20, 0.10, but this is a fairly arbitrary choice.

Streak and tempo are not perfect metrics, because in extreme cases one can two-touch kill at the very end of the round while taking six jabs evenly spaced throughout.

This is a proof-of-concept tool, I make no claims about its accuracy or reliability. The parts were taken from the already brittle research pipeline created for another project, Frankensteined together with vibecode, and had a basic GUI slapped on top for minimal practicity. Everything is provided as-is. 

Dammit, Jim, I'm a psychologist, not a programmer.

* * * * *

Instructions:

1 – Record one or more replay matches without interruptions or skips from anywhere before the “Fight” sign appears on the first round until the moment the “xyz wins” sign appears on the final round. Input History Display must be turned on for both players. A sample video is provided with the release.

2 – Load a video or a folder with several of these videos.

3 – Press the button.

When it finishes processing, it will show the MF Score and also generate a .csv report with details in the output folder.

Extracting the health-bar data visually is not a flawless process right now, but it mostly works if the video has all needed parts, has enough resolution (preferably 1080p) and is not overly compressed. The choice of stage or any visual obstruction of the health bar at KO might have adverse effects, though.
