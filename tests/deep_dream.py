"""
This tests the deep-dream (weird dreamy effect that uses Tenserflow)
Usage : python preprocessor.py --video <file_path>>
"""
from constants import *
from tqdm import tqdm

import argparse
import cv2
import input
import fmethods
import numpy as np
from output import *

_HEIGHT = _WIDTH = 500


# Parse terminal arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

# PREPARE VIDEO SOURCE
if args.get("video", None) is None:
    raise ValueError("Must provide video file using '--video' option!")
else:
    vs = input.VideoFile(args["video"])

# TODO For testing purposes, skip the fade-in and title card
for _ in range(1200):
    _ = vs.get_frame()

deepDream = fmethods.DeepDream()
outfile_DD = ProcessWrite("test_dd_video_output" + VIDEO_EXT, _HEIGHT, _WIDTH)

# TODO for testing purposes, save <1s
for _ in tqdm(range(8)):
    frame = vs.get_frame()
    if frame is None:
        break  # End of video

    deepDream.update(frame)
    outfile_DD.write(deepDream.get_display(dims=(_HEIGHT, _WIDTH)))

# Cleanup the camera and close any open windows
vs.close()
cv2.destroyAllWindows()
outfile_DD.close()
