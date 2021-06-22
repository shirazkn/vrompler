"""
Use this to convert a video file from the 20 to 80 second marks in a different format
ffmpeg -i input.avi -c:v h264 -ss 20 -t 80 output.mp4
"""
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import ffmpeg
import time

# Parse terminal arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
# Note: Contours smaller than min-area are ignored
args = vars(ap.parse_args())

# If the video argument is None, read from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
# else, read from file
else:
	vs = cv2.VideoCapture(args["video"])
	vs_data = ffmpeg.probe(args["video"])
	assert vs_data["format"]["probe_score"] > 90  # Else ffmpeg doesn't know this file (bad)!
	for stream in vs_data["streams"]:
	    if stream["codec_type"] == "video":
	        vs_data["frameduration_ms"] = (1.0/stream["r_frame_rate"])*1000.0

vs_data += 1

firstFrame = None

while True:
	timer = time.time()
	# Get current frame
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	text = "Unoccupied"
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break
	# Resize the frame (no need to process raw image!)
	frame = imutils.resize(frame, width=500)
	# Convert it to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Gaussian Blur (across a 21x21 range)
	# because identical frames will not be identical
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	# If the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# Difference between frames
	frameDelta = cv2.absdiff(firstFrame, gray)

	# Set a lower threshold to fill in the white 'patches' [25-255 is set to white]
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	# Dilate the white pixels to fill in the black 'holes'
	thresh = cv2.dilate(thresh, None, iterations=2)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	for c in cnts:
		# Ignore small contours
		if cv2.contourArea(c) < args["min_area"]:
			continue
		# Draw Bounding Box on the frame,
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		text = "Occupied"

	# Draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
				(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)

	elapsed_time = (time.time() - timer)*1000.0
	wait_time = min(vs_data["frameduration_ms"] - elapsed_time, 1)
	key = cv2.waitKey(wait_time) & 0xFF
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()