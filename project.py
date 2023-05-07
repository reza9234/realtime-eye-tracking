import cv2
import dlib
import numpy as np
import winsound
import time


def beep(number):
    frequency = 2200
    duration = 100  #ms
    for i in range(number):
        winsound.Beep(frequency, duration)
        time.sleep(0.05)


def shape_to_np(shape, dtype='int'):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        N = cv2.moments(cnt)
        cx = int(N['m10']/N['m00'])
        cy = int(N['m01']/N['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 255, 0), 2)
        return cx
    except:
        pass


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass

cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

prev_frame_time = 0
new_frame_time = 0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.medianBlur(thresh, 3)
        thresh = cv2.bitwise_not(thresh)


        try:
           if (contouring(thresh[:, 0:mid], mid, img)) <= shape[36][0] + 5:
               beep(2)
               cv2.arrowedLine(img, (250, 50), (400, 50), (255, 0, 0), thickness=7, tipLength=0.3)
           elif contouring(thresh[:, mid:], mid, img, True) + 5 >= shape[45][0]:
               beep(3)
               cv2.arrowedLine(img, (400, 50), (250, 50), (255, 0, 0), thickness=7, tipLength=0.3)
        except:
            pass

        # for (x, y) in shape[36]:
        cv2.circle(img, (shape[36][0], shape[36][1]), 2, (0, 255, 0), -1)
        cv2.circle(img, (shape[45][0], shape[45][1]), 2, (0, 255, 0), -1)
        #     print(x, y)
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = round(fps, 1)
        fps = str(fps)
        cv2.putText(img, fps, (7, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)


    cv2.imshow('eyes', img)
    cv2.imshow('image', thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
