import multiprocessing
import us_test
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "dummy"

def run_in_parallel(*fns):
    proc = []
    for fn in fns:
        p = multiprocessing.Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

if __name__ == '__main__':
    run_in_parallel(apprixmate_spline_video.main, us_test.main)

