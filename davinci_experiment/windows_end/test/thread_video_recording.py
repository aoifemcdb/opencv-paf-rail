from dataclasses import dataclass, field

from windows_end.video_threading.video_get import VideoGet
from windows_end.video_threading.video_show import VideoShow

from threading import Thread
import cv2.cv2 as cv


@dataclass
class ThreadVideoRecording:
    """
    Class that continuously shows a frame using a dedicated thread.
    """
    data_folder: str
    video_name_webcam: str = ''
    video_getter_webcam: VideoGet = field(init=False)
    video_shower_webcam: VideoShow = field(init=False)
    out_webcam: cv.VideoWriter = field(init=False)
    stopped: bool = False

    def __post_init__(self):
        # self.video_name_bk = '{}/ultrasound.avi'.format(self.data_folder, 'ultrasound')
        self.video_name_webcam = '{}/webcam.avi'.format(self.data_folder, 'webcam')

        # video_getter_bk = BKOpenCV().start()
        # video_shower_bk = VideoShow(video_getter_bk.frame, 'Ultrasound stream').start()
        # out_bk = cv.VideoWriter(video_name_bk, cv.VideoWriter_fourcc(*'XVID'), 30, (892, 728))

    def start(self):
        self.video_getter_webcam = VideoGet('webcam').start()
        self.video_shower_webcam = VideoShow(self.video_getter_webcam.frame, 'Webcam stream').start()
        self.out_webcam = cv.VideoWriter(self.video_name_webcam, cv.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
        Thread(target=self.record(), args=()).start()
        return self

    def record(self):
        while not self.stopped:
            if self.video_getter_webcam.stopped or self.video_shower_webcam.stopped:
                self.stop()
            if cv.waitKey():
                self.stop()

            # print('Not stopped')
            frame_webcam = self.video_getter_webcam.frame
            self.video_shower_webcam.frame = frame_webcam
            self.out_webcam.write(frame_webcam)
        print('Getting out')

    def stop(self):
        """Print setting stop to True"""
        print('Setting stop to True')
        self.stopped = True
        self.video_getter_webcam.stop()
        self.video_shower_webcam.stop()