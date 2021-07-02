"""
This does off-line processing to save optic flow data (Takes a lot of time on CPU!)
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

opticFlow = fmethods.RaftModel(iterations=1)
deepDream = fmethods.DeepDream()
outfile_OF1 = ProcessWrite("test_cropped_video_output" + VIDEO_EXT, opticFlow.height, opticFlow.width)
outfile_OF2 = ProcessWrite("test_of_video_output" + VIDEO_EXT, opticFlow.height, opticFlow.width)
outfile_DD = ProcessWrite("test_dd_video_output" + VIDEO_EXT, opticFlow.height, opticFlow.width)
outfile_OFnumpy = NumpyWrite("test_numpy_output" + NUMPY_EXT)  # this is a function not class
flow_frames = []

# TODO for testing purposes, save <1s
for _ in tqdm(range(5)):
    frame = vs.get_frame()
    if frame is None:
        break  # End of video

    opticFlow.update(frame)
    deepDream.update(frame)

    outfile_OF1.write(opticFlow.cropped_frame)
    outfile_OF2.write(opticFlow.get_display())
    outfile_DD.write(deepDream.get_display(dims=(opticFlow.height, opticFlow.width)))
    outfile_OFnumpy.write([opticFlow.flow_low, opticFlow.flow_up])

    """
    TODO: OF2.write() does not work but works in realtime.py. Also output looks blue??
    The following is from cv2.imshow() docs...
     
    .   The function imshow displays an image in the specified window. If the window was created with the
    .   cv::WINDOW_AUTOSIZE flag, the image is shown with its original size, however it is still limited by the screen resolution.
    .   Otherwise, the image is scaled to fit the window. The function may scale the image, depending on its depth:
    .   
    .   -   If the image is 8-bit unsigned, it is displayed as is.
    .   -   If the image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. That is, the
    .       value range [0,255\*256] is mapped to [0,255].
    .   -   If the image is 32-bit or 64-bit floating-point, the pixel values are multiplied by 255. That is, the
    .       value range [0,1] is mapped to [0,255].
    
    TODO - Something similar when writing to file!
    """


# Cleanup the camera and close any open windows
vs.close()
cv2.destroyAllWindows()
for file in [outfile_OF1, outfile_OF2, outfile_DD, outfile_OFnumpy]:
    file.close()
