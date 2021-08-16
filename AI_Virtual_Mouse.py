# Credits: https://www.youtube.com/watch?v=8gPONnGIPgw
import cv2
import mediapipe as mp
import numpy as np
import time
import HandTrackingBasics_Module as htm
import autopy

#Base Tuning Params
########################################################################
wCam = 1280
hCam = 720
# Used for frame reduction. To scale mouse movement proportionally.
frameRed = 120
wScreen, hScreen = autopy.screen.size()
# print(wScreen, "THIS IS WIDTH")
# print(hScreen, "THIS IS HEIGHT")
# For scaling the mouse. NOTE: Normalization can be used as well.
smoothening = 10 # found by trial-error method. Mostly, 7 - 12 is a good range.
#########################################################################

# To smoothen mouse clicking.
prelocX, prelocY = 0, 0
currlocX, currlocY = 0, 0

# Required to use webcam of the device. (s1)
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# To track framerate(S1)
pTime = 0
cTime = 0

detector = htm.handDetector(maxHands=1, detectionCon=0.4)

while True:
    # Required to use webcam of the device.(s2)
    success, img = cap.read()

    # Find HandLandMarks.
    img = detector.findHands(img)
    # boundary box -->bbox
    lmList, bbox = detector.findPosition(img, idx=1) #RED Module

    # Get tips of index and middle finger:
    if len(lmList) != 0:
        x1_cord, y1_cord = lmList[8][1:]
        x2_cord, y2_cord = lmList[12][1:]

        # Check which fingers are up.
        fingers = detector.fingersUp()
        # Hyperparam adjustments make movement better.
        # Box and interp have to change accordingly so that mouse input stays within bounds.
        cv2.rectangle(img, (frameRed + 40, frameRed - 60), (wCam - frameRed, hCam - frameRed), (0, 0, 255), 3)

        # If only Index finger is up: MOVING MODE
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert co-ordinates obtained into pixel values used to navigate the screen.
            # interp--> Used to interpolate length.
            x3 = np.interp(x1_cord, (frameRed + 40, wCam - frameRed), (0, wScreen))  # These hyperparam adjustments can
            y3 = np.interp(y1_cord, (frameRed - 60, hCam - frameRed), (0, hScreen))  # make detection smoother

            # Smoothen the Values:
            currlocX = prelocX + (x3 - prelocX) / smoothening
            currlocY = prelocY + (y3 - prelocY) / smoothening

            # Move Mouse:
            autopy.mouse.move(wScreen - currlocX, currlocY)
            cv2.circle(img, (x1_cord, y1_cord), 15, (255, 0, 255), cv2.FILLED)
            prelocX, prelocY = currlocX, currlocY

        # Else: Both Index and Middle fingers are up: CLICKING MODE
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between fingers.
            # Click mouse if its a short distance.
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)
            if length < 60:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # To track framerate(S2)
    cTime = time.time()  # current time
    fps = 1 / (cTime - pTime)  # fps calc math formula
    pTime = cTime

    # To display the framerate on the screen.
    # cv2.putText(image =img, value_to_disp = fps, position_disp = (x,y), text_font = cv2.FONT,
    #           color_text = RBG scale color, thickness of text =3)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3,
                (255, 0, 255), 3)

    # Required to use webcam of the device.(s3,s4)
    cv2.imshow("Image", img)
    # waitkey() waits until a key is pressed here (q) to exit from creating continuous frames.
    if cv2.waitKey(1) & 0xFF == ord('q'):  # very important if you want a continuous array of frames
        break
