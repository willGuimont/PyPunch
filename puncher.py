import cv2
import numpy as np
from detector import Detector
from data_saver import DataSaver
from math_utils import MathUtils


class Puncher:

    __detector = Detector()
    __g = 9.8

    def __init__(self):
        self.__px_per_m_values = []
        self.__px_per_m = None

    def calibrate(self, path, save_path="", verbose=False):
        a = self.__get_acceleration_px(path, save_path)
        px_per_m = a / Puncher.__g
        self.__px_per_m_values.append(px_per_m)
        self.__update_px_per_m()
        if verbose:
            print(self.__px_per_m)

    def punch(self, path, save_path=""):
        return self.__get_initial_speed(path, save_path)

    def __get_coefficients(self, path, save_path=""):
        video_capture = cv2.VideoCapture(path)
        positions = self.__detector.detect(video_capture)

        if save_path != "":
            DataSaver.save_to_file(positions, "test.csv")

        fps = video_capture.get(cv2.CAP_PROP_FPS)

        movement = Puncher.__get_displacement(positions)
        a, b, c = MathUtils.get_polynomial_coefficients(fps, movement, 2)
        return a, b, c

    @staticmethod
    def __get_displacement(positions):
        if len(positions) == 0:
            raise ValueError("Position is empty")
        x = np.array([d[0] for d in positions])
        y = np.array([d[1] for d in positions])
        dx = x - x[0]
        dy = y - y[0]

        return np.sqrt(np.power(dx, 2) + np.power(dy, 2))

    def __update_px_per_m(self):
        if len(self.__px_per_m_values) != 0:
            self.__px_per_m = np.mean(self.__px_per_m_values)

    def __get_acceleration_px(self,path, save_path=""):
        a, b, c = self.__get_coefficients(path, save_path)
        return 2 * a

    def __get_initial_speed(self, path, save_path=""):
        if self.__px_per_m is None:
            raise ValueError("Puncher must be calibrated before using... Dumbass")
        a, b, c = self.__get_coefficients(path, save_path)

        return b / self.__px_per_m

if __name__ == '__main__':
    puncher = Puncher()
    base_path = "videos/{type}_{index}.mp4"
    # TODO parse filenames
    # Calibration
    fall_path = base_path.format(type="fall", index='{index}')
    fall_num = 5
    for i in range(fall_num):
        puncher.calibrate(fall_path.format(index=i + 1), verbose=True)

    print(puncher.punch('videos/punch_1.mp4'))
