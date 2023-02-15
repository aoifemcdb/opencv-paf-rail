import cv2.cv2 as cv
from cv2.aruco import drawDetectedMarkers

from camera.aruco import detect_aruco, draw_center
from camera.aruco import load_coefficients
from camera.thread.video_show import VideoShow


class VideoShowAruco(VideoShow):
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None, video_name='output'):
        super().__init__(frame, video_name)
        self.aruco_dict = cv.aruco.DICT_6X6_50
        [self.camera_matrix, self.dist_coeffs] = load_coefficients(
            "calibration_files/integrated_webcam_calibration.yml")
        self.corners = []

    def start(self):
        return super().start()

    def show(self):
        while not self.stopped:
            self.corners, _, _ = detect_aruco(self.frame, cv.aruco.DICT_6X6_50)

            if len(self.corners) > 4:
                cv.imwrite('saved_images/aruco.png', self.frame)

            for marker_corner in self.corners:
                draw_center(self.frame, marker_corner)
                rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(marker_corner, 0.03,
                                                                              self.camera_matrix, self.dist_coeffs)
                # Get rid of that nasty numpy value array error
                (rvec - tvec).any()
                cv.aruco.drawAxis(self.frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.01)  # Draw axis

            drawDetectedMarkers(self.frame, self.corners)
            cv.imshow(self.name, self.frame)

            if cv.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        return super().stop()