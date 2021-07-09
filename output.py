from constants import *
import subprocess
import ffmpeg
import gzip
import numpy as np


class ProcessWrite:
    def __init__(self, filename, height, width, framerate):
        _args = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height),
                   framerate=str(framerate))
            .output(filename, pix_fmt='yuv420p', vsync='0', framerate=str(SAVE_FRAMERATE))  # OpenCV uses the BGR format
            .overwrite_output()
            .compile()
        )
        self.p = subprocess.Popen(_args, stdin=subprocess.PIPE)

    def write(self, _frame):
        # TODO Resize here??
        self.p.stdin.write(
            _frame
            .astype(np.uint8)
            .tobytes()
        )

    def close(self):
        self.p.stdin.close()
        self.p.wait()

    def __del__(self):
        self.close()


class NumpyWrite:
    def __init__(self, filename, compression=True):
        if compression:
            self.file = gzip.GzipFile(filename, "w")
        else:
            self.file = filename
        self.data = []
        print("Warning, output.NumpyWrite saves only when file is closed.")

    def write(self, data):
        self.data.append(data)

    def close(self):
        np.save(file=self.file, arr=self.data, allow_pickle=True)
        self.file.close()

    def __del__(self):
        self.close()
