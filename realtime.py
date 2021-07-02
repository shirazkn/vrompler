"""
This does real-time processing
(It's SUPER SLOW on cpu with optical flow turned on. Change device to 'cuda' in constants.py)
Usage : python preprocessor.py --video <file_path>>, not specifying video defaults to using webcam

Note:
Use this to convert a video file from the 20 to 80 second marks in a different format
ffmpeg -i input.avi -c:v h264 -ss 20 -t 80 output.mp4
Video output resolution is handled by constants.PROCESSING_FRAME_SIZE
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
	objDetection.update(frame)
	frame_overlay = objDetection.get_display()

	# OPTIC FLOW USING RAFT
	opticFlow.update(frame)
	flow_frame = opticFlow.get_display()

	# Show the frame
	cv2.imshow("Raw Footage", frame_overlay)
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
