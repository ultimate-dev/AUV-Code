import cv2
import numpy as np


def display(img, frameWidth, frameHeight, deadZone):
    cv2.line(img, (int(frameWidth/2)-deadZone, 0),
             (int(frameWidth/2)-deadZone, frameHeight), (255, 255, 0), 3)
    cv2.line(img, (int(frameWidth/2)+deadZone, 0),
             (int(frameWidth/2)+deadZone, frameHeight), (255, 255, 0), 3)
    cv2.circle(img, (int(frameWidth/2), int(frameHeight/2)), 5, (0, 0, 255), 5)
    cv2.line(img, (0, int(frameHeight / 2) - deadZone),
             (frameWidth, int(frameHeight / 2) - deadZone), (255, 255, 0), 3)
    cv2.line(img, (0, int(frameHeight / 2) + deadZone),
             (frameWidth, int(frameHeight / 2) + deadZone), (255, 255, 0), 3)


def fps(img, timer):
    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img, "FPS: "+str(int(fps)), (15, 30),  cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (100, 255, 0), 2)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
