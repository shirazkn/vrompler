import cv2
import ffmpeg
import time
from constants import PROCESSING_FRAME_WIDTH
from imutils import resize
from imutils.video import VideoStream


class VideoFile:
    def __init__(self, filename):
        self.file = cv2.VideoCapture(filename)
        vs_data = ffmpeg.probe(filename)
        assert vs_data["format"]["probe_score"] > 90  # Else ffmpeg doesn't know this file (bad)!
        self.framelength_ms = None  # Used to match playback speed of file
        for stream in vs_data["streams"]:
            if stream["codec_type"] == "video":
                self.framelength_ms = (1.0 / eval(stream["r_frame_rate"])) * 1000.0
                break

    def get_frame(self):
        frame = self.file.read()[1]
        return resize(frame, width=PROCESSING_FRAME_WIDTH) if frame is not None else None

    def close(self):
        self.file.release()


class WebCam:
    def __init__(self):
        self.stream = VideoStream(src=0).start()
        time.sleep(2.0)  # Not sure if this (slow wakeup) is needed?
        self.framelength_ms = 1  # WebCam playback speed does not need compensation (see VideoFile)

    def get_frame(self):
        frame = self.stream.read()
        return resize(frame, width=PROCESSING_FRAME_WIDTH) if frame is not None else None

    def close(self):
        self.stream.stop()
