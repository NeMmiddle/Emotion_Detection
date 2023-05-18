import time
import cv2


def calculate_fps(start_time=None, fps=None):
    if start_time is None:
        start_time = time.time()
    if fps is None:
        fps = 0

    fps += 1
    if (time.time() - start_time) > 1:
        fps_value = fps / (time.time() - start_time)
        fps_text = "FPS: {:.2f}".format(fps_value)
        start_time = time.time()
        fps = 0
    else:
        fps_text = None

    return start_time, fps, fps_text


def fps_coords():
    fps_x, fps_y = 10, 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    return fps_x, fps_y, font, font_scale, font_thickness
