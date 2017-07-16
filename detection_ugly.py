import cv2
import numpy as np


def nothing(var):
    pass

track_win_name = "Range"
cv2.namedWindow(track_win_name)

cv2.createTrackbar("minH", track_win_name, 0, 150, nothing)
cv2.createTrackbar("maxH", track_win_name, 0, 150, nothing)

cv2.createTrackbar("minS", track_win_name, 0, 255, nothing)
cv2.createTrackbar("maxS", track_win_name, 0, 255, nothing)

cv2.createTrackbar("minV", track_win_name, 0, 255, nothing)
cv2.createTrackbar("maxV", track_win_name, 0, 255, nothing)

cv2.setTrackbarPos("minH", track_win_name, 27)
cv2.setTrackbarPos("maxH", track_win_name, 255)
cv2.setTrackbarPos("minS", track_win_name, 0)
cv2.setTrackbarPos("maxS", track_win_name, 255)
cv2.setTrackbarPos("minV", track_win_name, 0)
cv2.setTrackbarPos("maxV", track_win_name, 255)

video_name = 'punch_3.mp4'
filename = 'csv/' + video_name.replace('mp4', 'csv') + 'a'

file = open(filename, 'w')

cap = cv2.VideoCapture('videos/' + video_name)

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS: {0}".format(fps))

last_frame = None

ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel = np.ones((5,5),np.uint8)

while cap.isOpened():
    ret, frame = cap.read()

    # if empty, then video finished
    try:
        pass # frame = cv2.pyrDown(frame)
    except:
        file.close()
        exit()
    # frame = frame[100:-100, :]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    found = False

    while not found:
        draw = frame.copy()

        minH = cv2.getTrackbarPos("minH", track_win_name)
        minS = cv2.getTrackbarPos("minS", track_win_name)
        minV = cv2.getTrackbarPos("minV", track_win_name)

        maxH = cv2.getTrackbarPos("maxH", track_win_name)
        maxS = cv2.getTrackbarPos("maxS", track_win_name)
        maxV = cv2.getTrackbarPos("maxV", track_win_name)

        inRange = cv2.inRange(hsv, (minH, minS, minV), (maxH, maxS, maxV))
        inRange = cv2.morphologyEx(inRange, cv2.MORPH_CLOSE, kernel)
        inRange = cv2.erode(inRange, kernel, iterations=1)

        im2, contours, hierarchy = cv2.findContours(inRange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_index = -1
        best_fitness = -1
        best_pos = (0, 0)
        best_radius = 0

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 10:
                continue
            perimeter = cv2.arcLength(cnt, True)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            try:
                fitness = abs(area / (perimeter * radius) - 1) # must be close to 0 # so that area /(perimeter * radius) = 1
            except ZeroDivisionError:
                continue
            if best_fitness == -1 or fitness < best_fitness:
                best_fitness = fitness
                best_index = i
                best_pos = (int(x), int(y))
                best_radius = radius

        cv2.circle(draw, best_pos, int(best_radius), (0, 255, 0), 2)

        cv2.imshow("range", inRange)
        cv2.imshow("frame", draw)

        key = cv2.waitKey(10)

        if key == 32:
            file.write('{x},{y}\n'.format(x=best_pos[0], y=best_pos[1]))
            found = True
            print('{x},{y}'.format(x=best_pos[0], y=best_pos[1]))
            print("Radius: {r}".format(r=best_radius))
        elif key == 27:
            file.close()
            exit()

    found = False
