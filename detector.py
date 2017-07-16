import cv2
import numpy as np

class Detector:

    __slider_window_name = "HSV range"

    __min_h_slider_name = "Min Hue"
    __max_h_slider_name = "Max Hue"

    __min_s_slider_name = "Min Saturation"
    __max_s_slider_name = "Max Saturation"

    __min_v_slider_name = "Min Value"
    __max_v_slider_name = "Max Value"

    __structure_elem_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    __kernel = np.ones((5, 5), np.uint8)

    __min_area_contour = 10

    def __init__(self):
        # Init sliders
        cv2.namedWindow(Detector.__slider_window_name)

        cv2.createTrackbar(Detector.__min_h_slider_name, Detector.__slider_window_name, 0, 151, self.__nothing)
        cv2.createTrackbar(Detector.__max_h_slider_name, Detector.__slider_window_name, 0, 151, self.__nothing)

        cv2.createTrackbar(Detector.__min_s_slider_name, Detector.__slider_window_name, 0, 256, self.__nothing)
        cv2.createTrackbar(Detector.__max_s_slider_name, Detector.__slider_window_name, 0, 256, self.__nothing)

        cv2.createTrackbar(Detector.__min_v_slider_name, Detector.__slider_window_name, 0, 256, self.__nothing)
        cv2.createTrackbar(Detector.__max_v_slider_name, Detector.__slider_window_name, 0, 256, self.__nothing)

        # Defaults value determined by experience
        cv2.setTrackbarPos(Detector.__min_h_slider_name, Detector.__slider_window_name, 27)
        cv2.setTrackbarPos(Detector.__max_h_slider_name, Detector.__slider_window_name, 151)

        cv2.setTrackbarPos(Detector.__min_s_slider_name, Detector.__slider_window_name, 0)
        cv2.setTrackbarPos(Detector.__max_s_slider_name, Detector.__slider_window_name, 256)

        cv2.setTrackbarPos(Detector.__min_v_slider_name, Detector.__slider_window_name, 0)
        cv2.setTrackbarPos(Detector.__max_v_slider_name, Detector.__slider_window_name, 256)

    # Empty callback for the sliders
    @staticmethod
    def __nothing(var):
        pass

    @staticmethod
    def __cost_contour(cnt):
        area = cv2.contourArea(cnt)

        # Discard contour too small
        if area < Detector.__min_area_contour:
            return -1

        perimeter = cv2.arcLength(cnt, True)
        _, radius = cv2.minEnclosingCircle(cnt)
        try:
            cost = abs(
                area / (perimeter * radius) - 1)  # must be close to 0 # so that area /(perimeter * radius) = 1
            return cost
        except ZeroDivisionError:
            return -1

    def __get_best_contour(self, contours):
        best_cnt = None
        best_cost = -1

        for cnt in contours:
            cost = self.__cost_contour(cnt)

            if (best_cost == -1 or cost < best_cost) and cost != -1:
                best_cost = cost
                best_cnt = cnt

        return best_cnt

    # Path may be a video file path or an integer for live cam
    def detect(self, video_capture, verbose=False):
        finished = False
        positions = []

        if verbose:
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            print("FPS: {0}".format(fps))

        while video_capture.isOpened() and not finished:
            ret, frame = video_capture.read()

            # If no image
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            while True:
                draw = frame.copy()

                min_h = cv2.getTrackbarPos(Detector.__min_h_slider_name, Detector.__slider_window_name)
                min_s = cv2.getTrackbarPos(Detector.__min_s_slider_name, Detector.__slider_window_name)
                min_v = cv2.getTrackbarPos(Detector.__min_v_slider_name, Detector.__slider_window_name)

                max_h = cv2.getTrackbarPos(Detector.__max_h_slider_name, Detector.__slider_window_name)
                max_s = cv2.getTrackbarPos(Detector.__max_s_slider_name, Detector.__slider_window_name)
                max_v = cv2.getTrackbarPos(Detector.__max_v_slider_name, Detector.__slider_window_name)

                in_range = cv2.inRange(hsv, (min_h, min_s, min_v), (max_h, max_s, max_v))
                in_range = cv2.morphologyEx(in_range, cv2.MORPH_CLOSE, Detector.__structure_elem_ellipse) # kernel?
                in_range = cv2.erode(in_range, Detector.__kernel, iterations=1)

                im2, contours, hierarchy = cv2.findContours(in_range, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                best_contour = self.__get_best_contour(contours)
                pos = (0, 2)

                if best_contour is not None:
                    pos, radius = cv2.minEnclosingCircle(best_contour)
                    pos = (int(pos[0]), int(pos[1]))
                    cv2.circle(draw, pos, int(radius), (0, 255, 0), 2)

                cv2.imshow("range", in_range)
                cv2.imshow("frame", draw)

                key = cv2.waitKey(10)

                # TODO HERE
                if key == 32: # Space to save the position
                    if verbose:
                        print(pos)
                    positions.append([pos[0], pos[1]])
                    break
                elif key == 27: # Escape to stop the detection
                    if verbose:
                        print("Finished")
                    finished = True
                    break
                elif key == ord('s'): # S to skip the frame, useful is the first frame isn't good
                    if verbose:
                        print("Skip")
                    break

        return positions


if __name__ == '__main__':
    pass
