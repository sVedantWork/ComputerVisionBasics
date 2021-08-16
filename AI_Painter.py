import autopy.mouse
import cv2
import numpy as np
import time
import HandTrackingBasics_Module as htm
import os

#######
wCam = 1280
hCam = 720
#######
# Import Images
folderPath = "Painting_Headers"
myList = os.listdir(folderPath)
# print(myList) Check to see if all images are available.

overlayList = []
for imgPath in myList:
    # cv2.imread() --> loads an image from the specified file and returns it
    image = cv2.imread(f'{folderPath}/{imgPath}')  # act path to import specific images from
    overlayList.append(image)  # storing the images in a list.
# print(len(overlayList)) # check if all images have been imported.

header = overlayList[0]
drawColor = (255, 0, 80) #DEFAULT Blue

# Required to use webcam and set its frame window to a specific size.
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.85)  # To get good painting we need high confidence

xPre, yPre = 0, 0
# Canvas to draw on.
#np.zeros takes height before width, we have 3 channels(b,g,r), np.uint8 means we get 0-255 values
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    success, img = cap.read()

    # Flips image horizontally so that drawing becomes more intuitive.
    img = cv2.flip(img, 1)

    # Detects the 21 3D landmarks on the hand.
    img = detector.findHands(img)

    # Get all landmark positions.
    lmList, _ = detector.findPosition(img, idx=1, draw=False)
    if len(lmList) != 0:
        # print(lmList)

        # Tips of index and middle finger.
        x1_cord, y1_cord = lmList[8][1:]
        x2_cord, y2_cord = lmList[12][1:]

        fingers = detector.fingersUp()
        # print(fingers) Check if all values are visible as needed.

        if fingers[1] and fingers[2]:
            xPre, yPre = 0, 0
            # Check for click
            if y1_cord < 130:
                if 100 < x1_cord < 220:
                    header = overlayList[0]
                    drawColor = (255, 0, 80)
                    print("DRAW1")
                elif 260 < x1_cord < 380:
                    header = overlayList[1]
                    drawColor = (0, 255, 255)
                    print("DRAW2")
                elif 470 < x1_cord < 700:
                    header = overlayList[2]
                    drawColor = (0, 120, 139)
                    print("DRAW3")
                elif 780 < x1_cord < 890:
                    header = overlayList[3]
                    drawColor = (0, 252, 124)
                    print("DRAW4")
                elif 1050 < x1_cord < 1160:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)
                    print("DRAW5")
            cv2.rectangle(img, (x1_cord, y1_cord - 25), (x2_cord, y2_cord + 25), drawColor, cv2.FILLED)

        elif fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1_cord, y1_cord), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            # At the initial stage when we don't have any previous points; we want to build a point here instead of
            # a line from 0,0 to curr x,y.
            if xPre == 0 and yPre == 0:
                xPre, yPre = x1_cord, y1_cord

            if drawColor == (0, 0, 0): #Eraser
                cv2.line(img, (xPre, yPre), (x1_cord, y1_cord), drawColor, 60)
                cv2.line(imgCanvas, (xPre, yPre), (x1_cord, y1_cord), drawColor, 60)
            else:
                # Draw a line.
                cv2.line(img, (xPre, yPre), (x1_cord, y1_cord), drawColor, 15)
                cv2.line(imgCanvas, (xPre, yPre), (x1_cord, y1_cord), drawColor, 15)

            # Update the locations
            xPre, yPre = x1_cord, y1_cord

    # To Draw on the display window {Complex Method}:
    # convert the imgCanvas from BGR color config to a greyscale config.
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    # Invert the greyscale image color config i.e. instead of drawing on a black canvas with white; we draw on
    # a white canvas with black.
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) #?
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    # bitwise 'and' of display window with canvas so that we can draw on it.
    img = cv2.bitwise_and(img, imgInv)
    # bitwise 'or' of display window with canvas so that we can draw on it with the specific color
    # that we have selected. We need this because otherwise it will draw using the selected colors on the canvas
    # instead of drawing with those colors on the display window.
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting header image.
    img[0:125, 0:1280] = header  # renders image on the display window.
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0) #Add and blend the images {SIMPLE WAY}
    cv2.imshow("Image", img)

    # waitkey() waits until a key is pressed here (q) to exit from creating continuous frames.
    if cv2.waitKey(1) & 0xFF == ord('q'):  # very important if you want a continuous array of frames
        break
