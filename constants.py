import os

NUMPY_EXT = ".npy.gz"
VIDEO_EXT = ".mp4"
IMAGE_EXT = ".jpg"

RAFT_MODELS = [
    {"rel_path": "RAFT/models/raft-things.pth", "width": 1024, "aspect_ratio": 1024.0/440}
]
RAFT_MODEL_IND = 0
RAFT_MODEL_FILE = os.path.join(os.path.dirname(__file__), RAFT_MODELS[RAFT_MODEL_IND]["rel_path"])

DEVICE = 'cpu'  # Can be run with 'cuda' on supporting systems (not tested)
PROCESSING_FRAME_WIDTH = RAFT_MODELS[RAFT_MODEL_IND]["width"]  # Height is scaled to match asp. ratio
ASPECT_RATIO = RAFT_MODELS[RAFT_MODEL_IND]["aspect_ratio"]

SAVE_DIRECTORY = "processed_videos"
SAVE_FRAMERATE = 60
