"""
This does real-time processing
(It's SUPER SLOW on cpu with optical flow turned on. Change device to 'cuda' in constants.py)
Usage : python preprocessor.py --video <file_path>>, not specifying video defaults to using webcam

Note:
Use this to convert a video file from the 20 to 80 second marks in a different format
ffmpeg -i input.avi -c:v h264 -ss 20 -t 80 output.mp4
Video output resolution is handled by constants.PROCESSING_FRAME_SIZE

TODO : Make processing frame resolution separate from display resolution
TODO : Add a display/output class that can redo aspect ratio using black bars/some other method
TODO : Write a file that pre-processes optical flow on a video file and generates a dataset.
^Saves FPS, original file path, original file name, FPS of data
^Advantage : Data can be at a different FPS after smoothing,
or can be interpolated at higher FPS to slow down original video
Do this after you figure out how to upscale/downscale so that flow data can be compared one-to-one againt pixels
"""
import time
import argparse
import cv2
import input
import fmethods


# Parse terminal arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

# PREPARE VIDEO SOURCE
if args.get("video", None) is None:
	vs = input.WebCam()
else:
	vs = input.VideoFile(args["video"])

objDetection = fmethods.ObjectDetection()
opticFlow = fmethods.RaftModel(iterations=1)

while True:
	timer = time.time()
	frame = vs.get_frame()
	if frame is None:  # End of video
		break

	# OBJECT DETECTION USING OPENCV
	frame_overlay, thresh_frame, delta_frame = objDetection.update(frame)

	# OPTIC FLOW USING RAFT
	flow_frame = opticFlow.update(frame)

	# Show the frame
	cv2.imshow("Raw Footage", frame_overlay)
	cv2.imshow("Threshold Footage", thresh_frame)
	cv2.imshow("Frame Delta", delta_frame)
	cv2.imshow("Optic Flow", flow_frame)

	# If display rate is faster than frame rate, wait a bit to match playback speed of source
	# TODO This is done inside a 'display' method within FMethod (and update timer inside it)
	elapsed_time = (time.time() - timer) * 1000.0
	wait_time = max(vs.framelength_ms - elapsed_time, 1)
	key = cv2.waitKey(int(wait_time)) & 0xFF
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
vs.close()
cv2.destroyAllWindows()
