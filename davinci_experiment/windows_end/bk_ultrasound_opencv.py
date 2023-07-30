import time

import numpy as np
from threading import Thread
import cv2

from sksurgerybk.interface.bk5000 import BK5000


class BKOpenCV:
    def __init__(self):
        """ Display BK data using OpenCV."""
        self.default_tcp_ip = '128.16.0.3'
        self.default_tcp_port = 7915
        self.default_timeout = 5
        self.default_fps = 30

        # Define the codec and create VideoWriter object
        self.bk5000 = BK5000(self.default_timeout, self.default_fps)
        self.connect()

        # Get the first frame to initialize self.frame
        self.bk5000.get_frame()
        self.frame: np.array = cv.cvtColor(self.bk5000.img, cv2.COLOR_GRAY2RGB)
        self.stopped: bool = False

    def connect(self):
        """ Start acquisition / streaming. """
        self.bk5000 = BK5000(self.default_timeout, self.default_fps)
        self.bk5000.connect_to_host(self.default_tcp_ip, self.default_tcp_port)
        self.bk5000.image_size = [756, 616]
        self.bk5000.pixels_in_image = 756 * 616
        self.bk5000.start_streaming()

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            self.bk5000.get_frame()
            self.frame = cv2.cvtColor(self.bk5000.img, cv2.COLOR_GRAY2RGB)

    def stop(self):
        """ Stop acquisition/streaming. """
        self.stopped = True

    def disconnect(self):
        """ Disconnect from host"""
        self.bk5000.stop_streaming()
        self.bk5000.disconnect_from_host()
        # Allow enough time to disconnect from host
        time.sleep(0.01)