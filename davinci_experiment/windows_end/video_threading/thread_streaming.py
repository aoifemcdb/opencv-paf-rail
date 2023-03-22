from datetime import datetime

from davinci_experiment.windows_end.bk_ultrasound_opencv import BKOpenCV
from davinci_experiment.windows_end.video_threading.video_get import VideoGet
from davinci_experiment.windows_end.video_threading.video_show import VideoShow
from windows_end.video_threading.countsPerSecond import CountsPerSec


def put_iterations_per_sec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
               (10, 450), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


def thread_get(video_name='webcam'):
    """
    Dedicated thread for grabbing videos frames with VideoGet object.
    Dedicated thread for showing videos frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(video_name).start()

    while True:
        if video_getter.stopped:
            video_getter.stop()
            break


def thread_both():
    """
    Dedicated thread for grabbing videos frames with VideoGet object.
    Dedicated thread for showing videos frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter_webcam = VideoGet('webcam').start()
    video_shower_webcam = VideoShow(video_getter_webcam.frame, 'Webcam stream').start()

    while True:
        if video_getter_webcam.stopped or video_shower_webcam.stopped:
            video_shower_webcam.stop()
            video_getter_webcam.stop()
            break

        frame = video_getter_webcam.frame
        video_shower_webcam.frame = frame


def thread_bk_get():
    """
    Dedicated thread for grabbing videos frames with VideoGet object.
    Dedicated thread for showing videos frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = BKOpenCV().start()
    cps = CountsPerSec().start()

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('ultrasound.avi', fourcc, 25, (892, 728))

    while True:
        if (cv.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = put_iterations_per_sec(frame, cps.countsPerSec())
        out.write(frame)
        cv.imshow("Video", frame)
        cps.increment()


def thread_bk_both():
    """
    Dedicated thread for grabbing videos frames with VideoGet object.
    Dedicated thread for showing videos frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter_bk = BKOpenCV().start()
    video_shower_bk = VideoShow(video_getter_bk.frame, 'Ultrasound stream').start()

    while True:
        any_thread_stopped = video_getter_bk.stopped or video_shower_bk.stopped
        if cv.waitKey(1) == ord("q") or any_thread_stopped:
            video_shower_bk.stop()
            video_getter_bk.stop()
            break

        frame = video_getter_bk.frame
        video_shower_bk.frame = frame


def thread_multiples(video_names_list: list):
    """
    Dedicated thread for grabbing videos frames with VideoGet object.
    Dedicated thread for showing videos frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """
    video_getters = []
    video_showers = []
    stop = False
    nb_video = len(video_names_list)

    for video_name in video_names_list:
        video_getter = VideoGet(video_name).start()
        video_getters.append(video_getter)
        video_showers.append(VideoShow(video_getter.frame, video_name).start())

    while True:
        for video_thread in video_getters + video_showers:
            if video_thread.stopped:
                stop = True
        if stop is True:
            for video_thread in video_getters + video_showers:
                video_thread.stop()
                break

        for i_video in range(nb_video):
            frame = video_getters[i_video].frame
            video_showers[i_video].frame = frame


def thread_bk_and_webcam():
    """
    Dedicated thread for grabbing videos frames with VideoGet object.
    Dedicated thread for showing videos frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """
    saving_folder = 'videos'
    date_time = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
    video_name_bk = '{}/{}_{}.avi'.format(saving_folder, date_time, 'ultrasound')
    video_name_webcam = '{}/{}_{}.avi'.format(saving_folder, date_time, 'webcam')

    video_getter_bk = BKOpenCV().start()
    video_shower_bk = VideoShow(video_getter_bk.frame, 'Ultrasound stream').start()
    out_bk = cv.VideoWriter(video_name_bk, cv.VideoWriter_fourcc(*'XVID'), 30, (756, 616))

    video_getter_webcam = VideoGet('webcam').start()
    video_shower_webcam = VideoShow(video_getter_webcam.frame, 'Webcam stream').start()
    out_webcam = cv.VideoWriter(video_name_webcam, cv.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

    while True:
        any_thread_stopped = video_shower_bk.stopped or video_getter_bk.stopped or \
                             video_shower_webcam.stopped or video_getter_webcam.stopped
        if cv.waitKey(1) == ord("q") or any_thread_stopped:
            video_shower_webcam.stop()
            video_getter_webcam.stop()
            video_shower_bk.stop()
            video_getter_bk.stop()
            break

        frame_bk = video_getter_bk.frame
        video_shower_bk.frame = frame_bk
        out_bk.write(frame_bk)

        frame_webcam = video_getter_webcam.frame
        video_shower_webcam.frame = frame_webcam
        out_webcam.write(frame_webcam)


if __name__ == "__main__":
    # noThreading()
    # thread_bk_get()
    # thread_bk_both()
    # thread_multiples(['output1', 'output2'])
    thread_bk_and_webcam()