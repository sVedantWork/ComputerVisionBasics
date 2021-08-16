# CREDITS: Murtaza's WorkShop Youtube Channel.
#       : https://github.com/AndreMiras/pycaw --pyclaw lib.
import cv2
import time

import numpy
import numpy as np
import HandTrackingBasics_Module as htm
import math
# Imports for volume control.
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#############################################
wCam, hCam = 1280, 720 #display window width and height params.
#############################################

# Required to use webcam of the device. (s1)
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# To track framerate(S1)
cTime = 0
pTime = 0
# We need to create an object of the module to use its functions here.
detector = htm.handDetector(detectionCon=0.7)  # Higher confidence level criteria for better volume control.

# Volume(S2)
# Using standard pycaw template for getting audio control.
# Initialization:
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()  # To get range of the volume on the device. Max= 0 and Min=-63.5
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 0
volPercentage = 0

while True:
    # Required to use webcam of the device.(s2)
    success, img = cap.read()

    # REF htm Module.
    img = detector.findHands(img)

    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # https://google.github.io/mediapipe/solutions/hands.html --> 21 points mapping the hand.
        # print(lmList[4], lmList[8])  # We need thumb_tip and indexFinger_tip

        # Obtain X, Y co-ordinates for the required positions.
        # lmList[attribute_num(0-21)][x_value(1)/ y_value(2)/ z_value(3)]
        x1_cord, y1_cord = lmList[4][1], lmList[4][2]
        x2_cord, y2_cord = lmList[8][1], lmList[8][2]
        x_centre = (x1_cord + x2_cord) // 2
        y_centre = (y1_cord + y2_cord) // 2

        # Draw circles around the required points to indicate points used.
        cv2.circle(img, (x1_cord, y1_cord), 15, (255, 0, 100), cv2.FILLED)  # REF HandTrackingBasics.py
        cv2.circle(img, (x2_cord, y2_cord), 15, (240, 0, 80), cv2.FILLED)

        # Draw a connected line between the 2 selected points.
        # cv2.line(image, 1st set of points, 2nd set of points, color of line, thickness of line)
        cv2.line(img, (x1_cord, y1_cord), (x2_cord, y2_cord), (200, 100, 200), 3)

        # Draw a centre point on the line above.
        cv2.circle(img, (x_centre, y_centre), 15, (255, 255, 0), cv2.FILLED)
        # The volume can be changed based on the length of the line. Here, we calculate the length of the line.
        # math.hypot returns the euclidean/ L2 norm.
        length = math.hypot(x2_cord - x1_cord, y2_cord - y1_cord)
        # print(length)

        if length < 50:
            cv2.circle(img, (x_centre, y_centre), 15, (0, 255, 0), cv2.FILLED)  # optional

        # Create a volume box to display on screen.
        # wBar = 85-50, lBar = 450-250
        # cv2.rectangle(image, initial_pos(width, height), final_pos(width, height), color, thickness)
        cv2.rectangle(img, (50, 250), (85, 450), (0, 100, 255), 4)
        cv2.rectangle(img, (50, int(volBar)), (85, 450), (0, 100, 255), cv2.FILLED)

        # To display the volume percentage on the screen.
        # cv2.putText(image =img, value_to_disp = fps, position_disp = (x--> left/right, y--> up/down),
        # text_font = cv2.FONT, color_text = RBG scale color, thickness of text =3)
        cv2.putText(img, f'VOL: {int(volPercentage)} %', (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (255, 0, 255), 3)

        # Volume(S3) ***
        # Hand Range: 50 to 300 #Print(cv2.line to check)
        # Vol Range: -63.5 to 0 #Print(volRange to check)
        # interp--> Used to interpolate length of range (50, 300) w.r.t vol of range(-63.5 to 0)
        # makes conversion easy and effective.
        vol = numpy.interp(length, [50, 300], [minVol, maxVol])  # For volume control
        print(int(length), vol)

        # interp--> Used to interpolate length of range (50, 350) w.r.t volBar of range(500 to 300)
        volBar = numpy.interp(length, [50, 300], [450, 250])  # For volume bar's appropriate display

        # interp--> Used to interpolate length of range (50, 300) w.r.t percentage of range(0 to 100)
        volPercentage = numpy.interp(length, [50, 300], [0, 100])  # For volume percentage display

        # Volume(S4)
        # To set device's master volume.
        volume.SetMasterVolumeLevel(vol, None)

    # To track framerate(S2)
    cTime = time.time()  # current time
    fps = 1 / (cTime - pTime)  # fps calc math formula
    pTime = cTime

    # To display the framerate on the screen.
    # cv2.putText(image =img, value_to_disp = fps, position_disp = (x,y), text_font = cv2.FONT,
    #           color_text = RBG scale color, thickness of text =3)
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3,
                (255, 0, 255), 3)

    # Required to use webcam of the device.(s3,s4)
    cv2.imshow("Image", img)
    # waitkey() waits until a key is pressed here (q) to exit from creating continuous frames.
    if cv2.waitKey(1) & 0xFF == ord('q'):  # very important if you want a continuous array of frames
        break
