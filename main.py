import cv2
from roboflow import Roboflow
from vehicle import Vehicle
from correction import conv
from envoriments import display, fps, stackImages
import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
deadZone = 100
threshold1 = 50
threshold2 = 100

rf = Roboflow(api_key="rjIESI1n63uJUu9nRKyg")
project = rf.workspace().project("circle-finder")
model = project.version(3).model

cap = cv2.VideoCapture(0)
dir = 0

if not cap.isOpened():
    raise IOError("Cannot open webcam")


while True:
    timer = cv2.getTickCount()
    ret, frame = cap.read()

    img = cv2.resize(frame, (frameWidth, frameHeight))
    imgAs = img.astype("float") / 255.0
    imgDim = np.expand_dims(imgAs, axis=0)
    imgCorr = conv(img)
    imgHsv = cv2.cvtColor(imgCorr, cv2.COLOR_BGR2HSV)
    imgContour = imgCorr.copy()
    # Classes
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    response = model.predict(img, confidence=40, overlap=30).json()

    if (response["predictions"]):
        for bounding_box in response["predictions"]:
            w = bounding_box['width']
            h = bounding_box['height']
            x = bounding_box['x']
            y = bounding_box['y']
            x0 = x - w / 2
            x1 = x + w / 2
            y0 = y - h / 2
            y1 = y + h / 2

            start_point = (int(x0), int(y0))
            end_point = (int(x1), int(y1))

            if (x < int(frameWidth/2)-deadZone):
                cv2.putText(imgContour, " GO LEFT ", (15, frameHeight-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (100, 255, 0), 2)
                cv2.rectangle(imgContour, (0, int(frameHeight/2-deadZone)), (int(
                    frameWidth/2)-deadZone, int(frameHeight/2)+deadZone), (0, 0, 255), cv2.FILLED)
                dir = 1
            elif (x > int(frameWidth / 2) + deadZone):
                cv2.putText(imgContour, " GO RIGHT ", (15, frameHeight-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (100, 255, 0), 2)
                cv2.rectangle(imgContour, (int(frameWidth/2+deadZone), int(frameHeight/2-deadZone)),
                              (frameWidth, int(frameHeight/2)+deadZone), (0, 0, 255), cv2.FILLED)
                dir = 2
            elif (y < int(frameHeight / 2) - deadZone):
                cv2.putText(imgContour, " GO UP ", (15, frameHeight-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (100, 255, 0), 2)
                cv2.rectangle(imgContour, (int(frameWidth/2-deadZone), 0), (int(
                    frameWidth/2+deadZone), int(frameHeight/2)-deadZone), (0, 0, 255), cv2.FILLED)
                dir = 3
            elif (y > int(frameHeight / 2) + deadZone):
                cv2.putText(imgContour, " GO DOWN ", (15, frameHeight-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (100, 255, 0), 2)
                cv2.rectangle(imgContour, (int(frameWidth/2-deadZone), int(frameHeight/2)+deadZone),
                              (int(frameWidth/2+deadZone), frameHeight), (0, 0, 255), cv2.FILLED)
                dir = 4
            else:
                dir = 0

            cv2.putText(
                imgContour,
                bounding_box["class"],
                (int(x0), int(y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2)
            cv2.line(imgContour, (int(frameWidth/2),
                     int(frameHeight/2)), (int(x), int(y)), (0, 0, 255), 3)
            cv2.rectangle(imgContour, start_point, end_point, (255, 0, 0), 2)
            cv2.putText(imgContour, " " + str(int(x)) + " " + str(int(y)),
                        (int(x - 20), int(y - 45)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (100, 255, 0), 2)

    else:
        cv2.putText(
            imgContour,
            "OBJECT LOST",
            (15, frameHeight - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2)
        dir = 0

    display(imgContour, frameWidth, frameHeight, deadZone)

    stackImgs = stackImages(
        0.8, ([imgContour, imgCorr, imgHsv], [imgGray, imgCanny, imgDil]))

    fps(stackImgs, timer)
    # Show
    cv2.imshow('AUV', stackImgs)

    if dir == 1:
        # SOL
        pass
    elif dir == 2:
        # SAĞ
        pass
    elif dir == 3:
        # YUKARI
        pass
    elif dir == 4:
        # AŞŞAGI
        pass
    else:
        # DUR
        pass

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
