import os
import datetime
import numpy as np
from argparse import Namespace
from copy import deepcopy

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

EPSILON = 1e-2  # Assumes we're not upsampling by 100x!!


def crop(_frame, new_dim, along=HEIGHT, align=ALIGN_CENTER):
    # Crops along dimension specified by `along`
    start_loc = 0
    old_dim = np.shape(_frame)[along]
    if align == ALIGN_CENTER:
        start_loc = (old_dim - new_dim)//2
    elif align == ALIGN_DOWN:
        start_loc = old_dim - new_dim

    slice_list = [slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    slice_list[along] = slice(start_loc, start_loc + new_dim, 1)
    return _frame[tuple(slice_list)]


def crop_from_aspect_ratio(_frame, new_asp_ratio, align=ALIGN_CENTER):
    # Preserves at least one dimension of frame, sets aspect ratio
    old_dims = np.shape(_frame)[:2]
    old_asp_ratio = old_dims[WIDTH] / old_dims[HEIGHT]
    if new_asp_ratio < old_asp_ratio:
        return crop(_frame, int(new_asp_ratio*old_dims[HEIGHT]), along=WIDTH, align=align)
    else:
        return crop(_frame, int(old_dims[WIDTH]/new_asp_ratio), along=HEIGHT, align=align)


def c_resize(_frame, dims: tuple, align=ALIGN_CENTER, cropping=True):
    # Constrained resize - Resize and crop to new dimensions
    old_dims = np.shape(_frame)[:2]
    new_asp_rat = dims[WIDTH] / dims[HEIGHT]
    old_asp_rat = old_dims[WIDTH] / old_dims[HEIGHT]

    if new_asp_rat < old_asp_rat:
        _frame = resize(_frame, height=dims[HEIGHT])
        # There's no functionality to crop outside a frame (i.e. add black bars) since I don't need it here currently
        return crop(_frame, dims[WIDTH], along=WIDTH, align=align) if cropping else _frame
    else:
        _frame = resize(_frame, width=dims[WIDTH])
        return crop(_frame, dims[HEIGHT], along=HEIGHT, align=align) if cropping else _frame


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
        self.source_frame = None
        self.last_source_frame = None  # Possibly not needed
        self.flow = None
        self.upsampled_flow = None
        self.last_upsampled_flow = None
        self.upsampled_avg_flow = None  # Flow averaged to get the exact flow at -1 timestep
        self.last_upsampled_avg_flow = None  # Flow averaged to get the exact flow at -2 timestep
        self.flow_limits = {
            "flow_u_min": 100.0, "flow_v_min": 100.0,
            "flow_u_max": -100.0, "flow_v_max": -100.0,
            "flow_speed_min": 100.0,
            "flow_speed_max": 0.0
                         }
        self.upsampled_flow_limits = {
            "flow_u_min": 100.0, "flow_v_min": 100.0,
            "flow_u_max": -100.0, "flow_v_max": -100.0,
            "flow_speed_min": 100.0,
            "flow_speed_max": 0.0
        }

        self.aspect_ratio = constants.RAFT_MODELS[constants.RAFT_MODEL_IND]["aspect_ratio"]
        self.width = constants.RAFT_MODELS[constants.RAFT_MODEL_IND]["width"]
        self.height = int(self.width/self.aspect_ratio)
        self.display_dims = None

        super().__init__(**kwargs)

    def prepare_frame(self, _frame):
        cropped_frame = c_resize(_frame, (self.height, self.width))
        _ret_frame = np.array(cropped_frame).astype(np.uint8)
        _ret_frame = torch.from_numpy(_ret_frame).permute(2, 0, 1).float()
        return _ret_frame[None].to(constants.DEVICE)

    def raft_display_from_flow(self, _out_frame):
        _frame = flow_viz.flow_to_image(_out_frame[0].permute(1, 2, 0).cpu().numpy())
        return c_resize(_frame, dims=self.display_dims)[:, :, [2, 1, 0]]

    def set_display_dims(self, dims=None, example_frame=None):
        if dims is not None:
            self.display_dims = dims
        elif example_frame is not None:
            self.display_dims = np.shape(example_frame)[:2]
        else:
            raise ValueError("Both arguments cannot be None.")

    def update(self, _frame):
        self.last_source_frame = self.source_frame
        self.last_frame = self.frame

        self.source_frame = _frame
        self.frame = self.prepare_frame(_frame)

        if self.last_frame is None:
            self.set_display_dims(example_frame=_frame)
            return

        with torch.no_grad():
            _, t_flow = self.model(self.last_frame, self.frame, iters=self.iterations, test_mode=True)

        self.flow = t_flow[0].permute(1, 2, 0).cpu().numpy()
        self.flow_limits = self.get_flow_limits(self.flow_limits, self.flow)

        self.upsample_flow()
        self.upsampled_flow_limits = self.get_flow_limits(self.upsampled_flow_limits, self.upsampled_flow)

    def get_display(self):
        # Uses the visualization method in raft to generate colormap from flow data
        return self.raft_display_from_flow(self.flow) if self.flow is not None else None

    def get_images_from_flow(self, frame):
        # Generates images from flow data so that flow data can be saved 'efficiently'
        # TODO
        return None, None

    def interpolate_frames(self, frame_1, frame_2, flow_1, flow_2, n_iterpolations):
        interpolated_source_frames = []
        intervals = np.linspace(0, 1, n_iterpolations+2)[1:-1]
        # 1 unit of dt is the time elapsed between source frames
        for dt in intervals:
            mid_1 = 1.0 - 0.5*dt  # Flow 1 is linearly interpolated with this weight
            mid_2 = 1.0 - 0.5*(1+dt)  # dt + (1-dt)/2 = (1+dt)/2
            frame_pushed_front = self.push_frame_to_flow(frame_1, flow_1*mid_1 + (1-mid_1)*flow_2, dt)
            frame_pushed_back = self.push_frame_to_flow(frame_2, flow_1*mid_2 + (1-mid_2)*flow_2, -(1-dt))
            interpolated_source_frames.append((1-dt)*frame_pushed_front + dt*frame_pushed_back)

        return interpolated_source_frames

    def push_frame_to_flow(self, frame, flow, dt):
        dims = np.shape(frame)[0:2]
        pushed_frame = deepcopy(frame)
        for i in range(dims[0]):
            for j in range(dims[1]):
                pushed_frame[i][j][0] = self.get_value_at_xy(frame, i+0.5-flow[i][j][0]*dt, j+0.5-flow[i][j][1]*dt, 0)
                pushed_frame[i][j][1] = self.get_value_at_xy(frame, i+0.5-flow[i][j][0]*dt, j+0.5-flow[i][j][1]*dt, 1)
                pushed_frame[i][j][2] = self.get_value_at_xy(frame, i+0.5-flow[i][j][0]*dt, j+0.5-flow[i][j][1]*dt, 2)
        return pushed_frame

    def upsample_flow(self):
        self.last_upsampled_avg_flow = self.upsampled_avg_flow
        self.last_upsampled_flow = self.upsampled_flow

        scale_factor = self.display_dims[HEIGHT]/self.height  # >=1
        flow_up = np.zeros([self.display_dims[HEIGHT], self.display_dims[WIDTH], 2])
        for i in range(self.display_dims[HEIGHT]):
            for j in range(self.display_dims[WIDTH]):
                # Get corresponding coordinates in low res. flow
                x, y = [(i+0.5)/scale_factor, (j+0.5)/scale_factor]
                flow_up[i][j][0] = self.get_value_at_xy(self.flow, x, y, 0) * scale_factor
                flow_up[i][j][1] = self.get_value_at_xy(self.flow, x, y, 1) * scale_factor

        self.upsampled_flow = flow_up
        if self.last_upsampled_flow is None:
            self.last_upsampled_flow = deepcopy(self.upsampled_flow)

        self.upsampled_avg_flow = (self.last_upsampled_flow + self.upsampled_flow) * 0.5

    @staticmethod
    def get_flow_limits(flow_limits, flow):
        flow_limits["flow_u_min"] = min(flow_limits["flow_u_min"], np.array(flow[:, :, 0]).min())
        flow_limits["flow_v_min"] = min(flow_limits["flow_v_min"], np.array(flow[:, :, 1]).min())
        flow_limits["flow_u_max"] = max(flow_limits["flow_u_max"], np.array(flow[:, :, 0]).max())
        flow_limits["flow_v_max"] = max(flow_limits["flow_v_max"], np.array(flow[:, :, 1]).max())

        flow_speed = np.sqrt(np.square(flow[:, :, 0]) + np.square(flow[:, :, 1]))
        flow_limits["flow_speed_min"] = min(flow_limits["flow_speed_min"], flow_speed.min())
        flow_limits["flow_speed_max"] = max(flow_limits["flow_speed_max"], flow_speed.max())
        return flow_limits

    @staticmethod
    def get_value_at_xy(data, x, y, value_ind):
        # TODO Account for height and width interchange!!
        # Upsamples value of data[:, :, value_ind] at float coordinates x, y (1 unit = 1 pixel of data)
        x_lim, y_lim = np.shape(data)[0:2]
        floor_x = min( max(int(np.floor(x - 0.5)), 0) , x_lim-1)
        floor_y = min( max(int(np.floor(y - 0.5)), 0) , y_lim-1)
        ceil_x = max( min(int(np.ceil(x - 0.5)) , x_lim-1), 0)
        ceil_y = max( min(int(np.ceil(y - 0.5)) , y_lim-1), 0)
        th_x = 1 - min(np.modf(x-0.5)[0], EPSILON)
        th_y = 1 - min(np.modf(y-0.5)[0], EPSILON)

        # Lower left, upper left ...clockwise
        vals = [data[floor_x][floor_y], data[floor_x][ceil_y], data[ceil_x][ceil_y], data[ceil_x][floor_y]]
        vals = [val[value_ind] for val in vals]
        vals1 = [vals[0]*th_y + vals[1]*(1-th_y), vals[1]*th_x + vals[2]*(1-th_x),
                 vals[2]*(1-th_y) + vals[3]*th_y, vals[3]*(1-th_x) + vals[0]*th_x]
        vals2 = [vals1[0]*th_x + vals1[2]*(1-th_x), vals1[1]*(1-th_y) + vals1[3]*th_y]
        return 0.5*(vals2[0] + vals2[1])


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
        with tfc.gfile.FastGFile(DeepDream._MODEL_FILENAME, 'rb') as f:
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
