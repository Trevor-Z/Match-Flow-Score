The idea: by combining the information we can extract from an SF6 replay video, we can generate a 100-point score that tells us how even (a win with few points) or lopsided (a win with many points) a round was. Or even whether the losing side actually played better and only lost because it was a very close call (a win with negative points).

The information being used: how long a round lasted; how much health the winner had at KO, or the health margin in a timeout; how long each player went without being hit; and how many times each one was hit.

These are converted into four factors:

w_duration: Rewards quick wins. Take the round length in seconds, subtract it from the full 100-second timer, and square what’s left. Short rounds push the score up; drawn-out rounds push it down.

w_health: Rewards finishing with a health lead. Measure the winner’s final health minus the loser’s, and square it to emphasize big gaps.

w_streak: Rewards sustained momentum. Find the longest damage-free streak for each side, convert those to fractions of the round, and take the difference.

w_tempo: Rewards offensive pressure. Count how many times each side’s health dropped; the side that caused more of those “damage events” gets credit.

Their sum generates the final score. By default, the weights are, respectively, 0.35, 0.35, 0.20, and 0.10, but this is arbitrary.

Streak and tempo are not perfect metrics, because in extreme cases one can two-touch kill while taking six jabs evenly spaced throughout the round.

This is a proof-of-concept scoring system and tool, I make no claims about its accuracy or balance. I took the parts out of the already brittle research pipeline created for another project, Frankensteined them together with vibecode, made a GUI and compiled a Windows .exe for minimal ease of use, and present it here as-is. It's ugly. Dammit, Jim, I'm a psychologist, not a programmer.

* * * * *

Instructions:

1 – Record one or more replay matches without interruptions or skips from anywhere before the “Fight” sign appears on the first round until the moment the “xyz wins” sign appears on the final round. Input History Display must be turned on for both players. A sample video is provided.

2 – Load a video or a folder with several of these videos.

3 – Press the button.

When it finishes processing, it will show the MF Score and also generate a .csv report with details in the output folder.

Extracting the health-bar data visually is not a flawless process right now, but it mostly works if the video has all needed parts, has enough resolution (preferably 1080p) and is not overly compressed.
