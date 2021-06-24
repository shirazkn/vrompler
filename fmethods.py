import datetime
import numpy as np
from argparse import Namespace

import cv2
import imutils

import torch
from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz
# from RAFT.core.utils.utils import InputPadder

from constants import DEVICE, RAFT_MODEL_FILE


class FMethod:
    # Class that processes frame(s)
    def __init__(self, **kwargs):
        # Not sure what I'd use this for yet, maybe to normalize aspect ratio/frame size amongst different effects
        pass


class RaftModel(FMethod):
    def __init__(self, iterations=20, **kwargs):
        # Load trained nn model
        model_args = Namespace(alternate_corr=False, mixed_precision=False, small=False)
        neural_net = torch.nn.DataParallel(RAFT(model_args))
        if DEVICE == 'cuda':
            neural_net.load_state_dict(torch.load(RAFT_MODEL_FILE))
        else:
            neural_net.load_state_dict(torch.load(RAFT_MODEL_FILE, map_location=torch.device('cpu')))
        self.model = neural_net.module
        self.model.to(DEVICE)
        self.model.eval()

        self.iterations = iterations
        self.frame = None
        self.last_frame = None
        self.flow = None

        super().__init__(**kwargs)

    def prepare_frame(self, _frame):
        #  TODO Make note of input frame, so we can upscale back to it (See notes at the top of realtime.py)
        _ret_frame = np.array(_frame[:440, :, :]).astype(np.uint8)  # TODO Replace this with logic (both aspect ratio and resize should be done here)
        _ret_frame = torch.from_numpy(_ret_frame).permute(2, 0, 1).float()
        return _ret_frame[None].to(DEVICE)

    def unprepare_frame(self, _out_frame):
        _frame = flow_viz.flow_to_image(_out_frame[0].permute(1, 2, 0).cpu().numpy())
        return _frame  # TODO convert frame back to display specs

    def update(self, _frame):
        self.frame = self.prepare_frame(_frame)
        if self.last_frame is None:
            self.last_frame = self.frame
            return _frame * 0.0

        with torch.no_grad():
            flow_low, flow_up = self.model(self.last_frame, self.frame, iters=self.iterations, test_mode=True)

        self.flow = flow_up
        self.last_frame = self.frame
        display_frame = self.unprepare_frame(self.flow)

        return display_frame[:, :, [2, 1, 0]] / 255.0


class ObjectDetection:
    """
    Logic copied from an opencv tutorial
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first_frame = None
        self.min_contour_area = 500

    def update(self, frame):
        text = "False"
        # Convert it to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gaussian Blur (across a 21x21 range)
        # because nearly-identical frames will not be identical
        frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
        # If the first frame is None, initialize it
        if self.first_frame is None:
            first_frame = frame_gray
            return frame * 0.0, frame * 0.0, frame * 0.0

        # Difference between frames
        frame_delta = cv2.absdiff(self.first_frame, frame_gray)

        # Set a lower threshold to fill in the white 'patches' [25-255 is set to white]
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        # Dilate the white pixels to fill in the black 'holes'
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            # Ignore small contours
            if cv2.contourArea(c) < self.min_contour_area:
                continue

            # Draw Bounding Box on the frame,
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text = "True"

        # Draw the text and timestamp on the frame
        cv2.putText(frame, "Objects in frame: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        return frame, thresh, frame_delta
