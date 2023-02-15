from threading import Thread
import cv2

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True

class VideoShowSg(VideoShow):
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, window, frame, video_name='Camera'):
        super().__init__(frame, video_name)
        self.imgbytes = None
        self.window = window

    def show(self):
        while not self.stopped:
            # Convert the image to PNG Bytes
            if self.frame is not None:
                self.imgbytes = cv.imencode('.png', self.frame)[1].tobytes()
                self.window['-IMAGE-'].update(self.imgbytes)


class ToImgBytes:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame, window):
        self.frame = frame
        self.window = window
        self.imgbytes = None
        self.stopped = False

    def start(self):
        Thread(target=self.convert_frame_to_bytes, args=()).start()
        return self

    def convert_frame_to_bytes(self):
        while not self.stopped:
            # Convert the image to PNG Bytes
            if self.frame is not None:
                self.imgbytes = cv.imencode('.png', self.frame)[1].tobytes()
                self.window['-IMAGE-'].update(self.imgbytes)

    def stop(self):
        self.stopped = True