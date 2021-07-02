import os
import datetime
import numpy as np
from argparse import Namespace

# For OpticFlow (RAFT)
import torch
from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz
# from RAFT.core.utils.utils import InputPadder

# For DeepDream
import zipfile
import tensorflow as tf
import tensorflow.compat.v1 as tfc

import cv2
import imutils
from imutils import resize
import constants
from hidden_constants import *


def crop(_frame, new_dim, along=HEIGHT, align=ALIGN_CENTER):
    start_loc = 0
    old_dim = np.shape(_frame)[along]
    if align == ALIGN_CENTER:
        start_loc = (old_dim - new_dim)//2
    elif align == ALIGN_DOWN:
        start_loc = old_dim - new_dim

    slice_list = [slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    slice_list[along] = slice(start_loc, start_loc + new_dim, 1)
    return _frame[tuple(slice_list)]


def c_resize(_frame, dims: tuple, align=ALIGN_CENTER):
    # Constrained Resize - Requires both dims to be specified
    old_dims = np.shape(_frame)[:2]
    new_asp_rat = dims[WIDTH] / dims[HEIGHT]
    old_asp_rat = old_dims[WIDTH] / old_dims[HEIGHT]

    if new_asp_rat < old_asp_rat:
        _frame = resize(_frame, height=dims[HEIGHT])
        # There's no functionality to crop outside a frame (i.e. add black bars) since I don't need it here currently
        return crop(_frame, dims[WIDTH], along=WIDTH, align=align)
    else:
        _frame = resize(_frame, width=dims[WIDTH])
        return crop(_frame, dims[HEIGHT], along=HEIGHT, align=align)


def quick_resize(_frame, dims):
    if np.any(dims is None):
        return resize(_frame, height=dims[HEIGHT], width=dims[WIDTH], inter=cv2.INTER_NEAREST)


class FMethod:
    # Class that processes frame(s)
    def __init__(self, **kwargs):
        # Not sure what I'd use this for yet, maybe to normalize aspect ratio/frame size amongst different effects
        pass


class RaftModel(FMethod):
    """
    See readme for downloading trained model
    """
    def __init__(self, iterations=20, **kwargs):
        # Load trained nn model
        model_args = Namespace(alternate_corr=False, mixed_precision=False, small=False)
        neural_net = torch.nn.DataParallel(RAFT(model_args))
        if constants.DEVICE == 'cuda':
            neural_net.load_state_dict(torch.load(constants.RAFT_MODEL_FILE))
        else:
            neural_net.load_state_dict(torch.load(constants.RAFT_MODEL_FILE, map_location=torch.device('cpu')))
        self.model = neural_net.module
        self.model.to(constants.DEVICE)
        self.model.eval()
        self.iterations = iterations

        self.frame = None
        self.last_frame = None
        self.cropped_frame = None
        self.display_frame = None
        self.flow_up = None
        self.flow_low = None

        self.aspect_ratio = constants.RAFT_MODELS[constants.RAFT_MODEL_IND]["aspect_ratio"]
        self.width = constants.RAFT_MODELS[constants.RAFT_MODEL_IND]["width"]
        self.height = int(self.width/self.aspect_ratio)
        self.display_dims = None

        super().__init__(**kwargs)

    def prepare_frame(self, _frame):
        self.cropped_frame = c_resize(_frame, (self.height, self.width))
        _ret_frame = np.array(self.cropped_frame).astype(np.uint8)
        _ret_frame = torch.from_numpy(_ret_frame).permute(2, 0, 1).float()
        return _ret_frame[None].to(constants.DEVICE)

    def unprepare_frame(self, _out_frame):
        _frame = flow_viz.flow_to_image(_out_frame[0].permute(1, 2, 0).cpu().numpy())
        return c_resize(_frame, dims=self.display_dims)[:, :, [2, 1, 0]] / 255.0

    def set_display_dims(self, dims=None, example_frame=None):
        if dims is not None:
            self.display_dims = dims
        elif example_frame is not None:
            self.display_dims = np.shape(example_frame)[:2]
        else:
            raise ValueError("Both arguments cannot be None.")

    def update(self, _frame):
        self.frame = self.prepare_frame(_frame)
        if self.last_frame is None:
            self.set_display_dims(example_frame=_frame)
            self.last_frame = self.frame
            self.display_frame = _frame * 0.0
            return

        with torch.no_grad():
            self.flow_low, self.flow_up = self.model(self.last_frame, self.frame, iters=self.iterations, test_mode=True)

        self.last_frame = self.frame
        self.display_frame = self.unprepare_frame(self.flow_up)

    def get_display(self):
        return self.display_frame


class ObjectDetection:
    """
    Logic copied from an opencv tutorial
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first_frame = None
        self.min_contour_area = 500
        self.display_frame = None

    def update(self, frame):
        text = "False"
        # Convert it to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gaussian Blur (across a 21x21 range)
        # because nearly-identical frames will not be identical
        frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
        # If the first frame is None, initialize it
        if self.first_frame is None:
            self.first_frame = frame_gray
            self.display_frame = frame * 0.0
            return

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

        self.display_frame = frame

    def get_display(self):
        return self.display_frame


class DeepDream(object):
    """
    DeepDream implementation adapted from official tensorflow deepdream tutorial,
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream
    and from ffmpeg-python/examples/tensorflow_stream.py
    Credit: Alexander Mordvintsev
    """

    _DOWNLOAD_URL = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    _ZIP_FILENAME = 'deepdream_model.zip'
    _MODEL_FILENAME = 'tensorflow_inception_graph.pb'

    @staticmethod
    def _download_model():
        try:
            from urllib.request import urlretrieve  # python 3
        except ImportError:
            from urllib import urlretrieve  # python 2
        urlretrieve(DeepDream._DOWNLOAD_URL, DeepDream._ZIP_FILENAME)
        zipfile.ZipFile(DeepDream._ZIP_FILENAME, 'r').extractall('.')

    @staticmethod
    def _tffunc(*argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See `_resize` function below.
        '''
        placeholders = list(map(tfc.placeholder, argtypes))

        def wrap(f):
            out = f(*placeholders)

            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

            return wrapper

        return wrap

    @staticmethod
    def _base_resize(img, size):
        '''Helper function that uses TF to resize an image'''
        img = tf.expand_dims(img, 0)
        return tfc.image.resize_bilinear(img, size)[0, :, :, :]

    def __init__(self):
        if not os.path.exists(DeepDream._MODEL_FILENAME):
            self._download_model()

        self._graph = tf.Graph()
        self._session = tfc.InteractiveSession(graph=self._graph)
        self._resize = self._tffunc(np.float32, np.int32)(self._base_resize)
        with tf.gfile.GFile(DeepDream._MODEL_FILENAME, 'rb') as f:  # or tfc.gfile.FastGFile
            graph_def = tfc.GraphDef()
            graph_def.ParseFromString(f.read())
        self._t_input = tfc.placeholder(np.float32, name='input')  # define the input tensor
        imagenet_mean = 117.0
        t_preprocessed = tf.expand_dims(self._t_input - imagenet_mean, 0)
        tf.import_graph_def(graph_def, {'input': t_preprocessed})

        self.t_obj = self.T('mixed4d_3x3_bottleneck_pre_relu')[:, :, :, 139]
        # self.t_obj = tf.square(self.T('mixed4c'))
        self.display_frame = None

    def T(self, layer_name):
        '''Helper for getting layer output tensor'''
        return self._graph.get_tensor_by_name('import/%s:0' % layer_name)

    def _calc_grad_tiled(self, img, t_grad, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h - sz // 2, sz), sz):
            for x in range(0, max(w - sz // 2, sz), sz):
                sub = img_shift[y:y + sz, x:x + sz]
                g = self._session.run(t_grad, {self._t_input: sub})
                grad[y:y + sz, x:x + sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    def update(self, frame, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        t_score = tf.reduce_mean(self.t_obj)  # defining the optimization objective
        t_grad = tf.gradients(t_score, self._t_input)[0]  # behold the power of automatic differentiation!

        # split the image into a number of octaves
        img = frame
        octaves = []
        for i in range(octave_n - 1):
            hw = img.shape[:2]
            lo = self._resize(img, np.int32(np.float32(hw) / octave_scale))
            hi = img - self._resize(lo, hw)
            img = lo
            octaves.append(hi)

        # generate details octave by octave
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = self._resize(img, hi.shape[:2]) + hi
            for i in range(iter_n):
                g = self._calc_grad_tiled(img, t_grad)
                img += g * (step / (np.abs(g).mean() + 1e-7))
                # print('.',end = ' ')
        self.display_frame = img

    def get_display(self, dims=None):
        if dims is None:
            return self.display_frame
        return c_resize(self.display_frame, dims=dims)
