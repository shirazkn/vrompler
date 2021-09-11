"""
This does off-line processing
Uses optic flow data (generated using the RAFT neural net) to slow down the video by a specified factor
Takes a lot of time on CPU!
Usage : python preprocessor.py --video <file_path>>
"""
from constants import *
from hidden_constants import *
from tqdm import tqdm

import argparse
import cv2
import input
import fmethods
import numpy as np
from output import *


SLOW_DOWN_FACTOR = 2  # Number of frames inserted between adjacent frames
GENERATE_MASKS = True  # Exports a video that can serve as a mask for source footage
OPTIC_FLOW_ITERATIONS = 10
opticFlow = fmethods.RaftModel(iterations=OPTIC_FLOW_ITERATIONS)
# (e.g. to extract interesting objects from a scene)


# Parse terminal arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to the video file")
args = vars(ap.parse_args())

# PREPARE VIDEO SOURCE
if args.get("video", None) is None:
    raise ValueError("Must provide video file using '--video' option!")
else:
    vs = input.VideoFile(args["video"])
    source_file, extension = os.path.splitext(args["video"])
    filename = os.path.split(source_file)[-1]
    filepath = os.path.join(SAVE_DIRECTORY, filename)

file_info = {
    "filename": filename, "filepath": filepath, "extension": extension,
    "source_framerate": vs.framerate, "output_framerate": SAVE_FRAMERATE,
    "slowdown_factor": SLOW_DOWN_FACTOR
             }

# Ensure that source footage is at least as big as optic flow frame
# else the logic for cropping_frame might fail below
if vs.height < opticFlow.height or vs.width < opticFlow.width:
    def get_frame():
        return fmethods.c_resize(vs.get_frame(), dims=(opticFlow.height, opticFlow.width), cropping=False)
else:
    def get_frame():
        return vs.get_frame()

# TODO For testing purposes, skip the fade-in and title card
for _ in range(1200):
    _ = vs.get_frame()

frame = get_frame()
cropped_frame = fmethods.crop_from_aspect_ratio(frame, new_asp_ratio=opticFlow.aspect_ratio)
opticFlow.set_display_dims(example_frame=cropped_frame)
opticFlow.update(cropped_frame)
dims = np.shape(cropped_frame)
outfiles = [
    ProcessWrite(filepath + "_cropped" + VIDEO_EXT, dims[HEIGHT], dims[WIDTH], framerate=vs.framerate),
    ProcessWrite(filepath + "_flow_u" + VIDEO_EXT, opticFlow.height, opticFlow.width, framerate=vs.framerate),
    ProcessWrite(filepath + "_flow_v" + VIDEO_EXT, opticFlow.height, opticFlow.width, framerate=vs.framerate),
    ProcessWrite(filepath + "_slowed" + str(int(SLOW_DOWN_FACTOR)) + "x" + VIDEO_EXT,
                 opticFlow.height, opticFlow.width, framerate=vs.framerate + 10),
    # The +10 is a slight speed-up for smoothing/giving ffmpeg more frames to work with
    NumpyWrite(filepath + "_info" + NUMPY_EXT)
            ]

last_cropped_frame = cropped_frame
last_flow = None
flow = None
cv2.imwrite("test_image_write" + IMAGE_EXT, cropped_frame)

outfiles[0].write(cropped_frame)

# TODO for testing purposes, save <1s | After testing: while frame
for i in tqdm(range(7)):
    opticFlow.update(frame)

    if i > 3:
        cv2.imwrite("frame" + str(i-1) + IMAGE_EXT, last_cropped_frame)
        outfiles[3].write(last_cropped_frame)
        interp_frames = opticFlow.interpolate_frames(last_cropped_frame, cropped_frame,
                                                     opticFlow.last_upsampled_avg_flow, opticFlow.upsampled_avg_flow,
                                                     n_iterpolations=SLOW_DOWN_FACTOR-1)
        # TODO Test this using a test-file and 2 simple images

        for j, _if in enumerate(interp_frames):
            cv2.imwrite("frame" + str(i) + "-" + str(j+1) + IMAGE_EXT, cropped_frame)
            outfiles[3].write(_if)

        cv2.imwrite("frame" + str(i) + IMAGE_EXT, cropped_frame)

        thresh_frame = opticFlow.threshold_image_from_flow(frame=cropped_frame, flow=opticFlow.upsampled_avg_flow,
                                                           threshold=(opticFlow.flow_limits["flow_speed_min"]
                                                                      + opticFlow.flow_limits["flow_speed_max"])*0.5)
        cv2.imwrite("threshframe" + str(i) + IMAGE_EXT, thresh_frame)

    last_cropped_frame = cropped_frame
    cropped_frame = fmethods.crop_from_aspect_ratio(frame, new_asp_ratio=opticFlow.aspect_ratio)

    # Testing
    flow_uv = opticFlow.flow
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    print("Max/min flow is", np.array(rad).max(), " and ", np.array(rad).min())
    flow_u, flow_v = opticFlow.get_images_from_flow(frame)
    outfiles[0].write(cropped_frame)

    frame = get_frame()

outfiles[3].write(last_cropped_frame)
outfiles[-1].write(file_info)

# Cleanup the camera and close any open windows
vs.close()
cv2.destroyAllWindows()
for file in outfiles:
    file.close()


