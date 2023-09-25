import  cv2
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider

cap = cv2.VideoCapture(0)
cap.set(15,0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_SATURATION,50)
# Make a horizontal slider to control the frequency.
def empty(i):
    pass

cv2.namedWindow("TrackedBars")
cv2.resizeWindow("TrackedBars", 640, 280)

cv2.createTrackbar("Hue Min", "TrackedBars", 164, 179, empty)
cv2.createTrackbar("Hue Max", "TrackedBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackedBars", 35, 255, empty)
cv2.createTrackbar("Sat Max", "TrackedBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackedBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackedBars", 255, 255, empty)

while True:
    _, frame = cap.read()
    height, width = frame.shape[0:2]  # получаем разрешение кадра

    cut_frame = frame[80:560, 150:560]

    hue_min = cv2.getTrackbarPos("Hue Min", "TrackedBars")
    hue_max = cv2.getTrackbarPos("Hue Max", "TrackedBars")
    sat_min = cv2.getTrackbarPos("Sat Min", "TrackedBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackedBars")
    val_min = cv2.getTrackbarPos("Val Min", "TrackedBars")
    val_max = cv2.getTrackbarPos("Val Max", "TrackedBars")

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    hvs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hvs_frame,lower,upper)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    if len(contours) != 0:  # если найден хоть один контур
        maxc = max(contours, key=cv2.contourArea)  # находим наибольший контур
        moments = cv2.moments(maxc)  # получаем моменты этого контура

        if moments["m00"] > 100:  # контуры с площадью меньше 20 пикселей не будут учитываться
            cx = int(moments["m10"] / moments["m00"])  # находим координаты центра контура (найденного объекта) по x
            cy = int(moments["m01"] / moments["m00"])  # находим координаты центра контура (найденного объекта) по y

            iSee = True  # устанавливаем флаг, что контур найден
            controlX = 2 * (
                        cx - width / 2) / width  # находим отклонение найденного объекта от центра кадра и нормализуем его (приводим к диапазону [-1; 1])



            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            perimeter = cv2.arcLength(cnt, True)

            if perimeter > 500:
                print(perimeter)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawContours(frame, maxc, -1, (0, 255, 0), 2)  # рисуем контур
                cv2.line(frame, (cx, 0), (cx, height), (0, 255, 0), 2)  # рисуем линию линию по x
                cv2.line(frame, (0, cy), (width, cy), (0, 255, 0), 2)  # линия по y
                cv2.putText(frame, 'iSee: {};'.format(iSee), (width - 370, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)  # текст
                cv2.putText(frame, 'controlX: {:.2f}'.format(cx), (width - 200, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)  # текст




    cv2.imshow('Tracked Object', frame)
    cv2.imshow('Crop', cut_frame)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
