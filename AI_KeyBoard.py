# Credits: Murtaza's Workshop #

import cv2
import mediapipe as mp
from HandTrackingBasics_Module import handDetector
from time import sleep
from pynput.keyboard import Key, Controller

# Use to access laptop webcam and capture video.
cap = cv2.VideoCapture(0)

# Screen Control Variables:
wCam = 1280  # width
hCam = 720  # height

cap.set(3, wCam)
cap.set(4, hCam)

# To detect hands.
detector = handDetector(detectionCon=0.95)  # high weight to detect hand better, needed for this project.

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

textToPrint = ""

keyboard = Controller()


def drawALL(img, buttonList):
    # To draw on the given image.
    for button in buttonList:
        x, y = button.pos
        width, height = button.size
        # cv2.rectangle(img= image to draw on, (x,y)--start pts., (x,y)--end pts., (RGB color), thickness)
        cv2.rectangle(img, button.pos, (x + width, y + height), (255, 0, 255), cv2.FILLED)
        # cv2.putText(img= image to draw on, "TEXT_TO_PUT", (x,y)--pos of text, Font, Font_size, Color, Thickness)
        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img


class Button():
    # Initialize with user provided attributes
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text


# List containing all button positions.
buttonList = []
# Iterate through the key list and add all the keys to the button list so that they can be drawn onto
# the image.
for i in range(len(keys)):  # How to improve this, quadratic implementation always sucks??
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

while True:
    # cap.read() use to grab data from videofiles in realtime.
    success, img = cap.read()

    # Detect the hands.
    img = detector.findHands(img)

    # Collect the 21 hand landmarks and the boundary box details.
    lmList, bbox = detector.findPosition(img, idx=1)

    # draw all the buttons on the image in each frame.
    img = drawALL(img, buttonList)

    if lmList:
        for button in buttonList:
            # stores position of a button
            x, y = button.pos
            # stores dimensions of a selected button
            width, height = button.size
            # print(lmList[8][id num][x-pos][y-pos])

            # checks if tip of index finger is on button position in both horizontal and vertical sense; it
            # highlights that specific button.
            if x < lmList[8][1] < x + width and y < lmList[8][2] < y + height:
                # cv2.rectangle(img= image to draw on, (x,y)--start pts., (x,y)--end pts., (RGB color), thickness)
                cv2.rectangle(img, button.pos, (x + width, y + height), (175, 0, 175), cv2.FILLED)
                # cv2.putText(img= image to draw on, "TEXT_TO_PUT", (x,y)--pos of text, Font, Font_size, Color,
                # Thickness)
                cv2.putText(img, button.text, (x + 20, y + 65),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                # calculates the distance between tips of index and middle finger as they appear in the image.
                distance1, _, _ = detector.findDistance(8, 12, img, draw=False)
                # print(distance1)
                distance2, _, _ = detector.findDistance(12, 0, img, draw=False)
                #print(distance2)
                distance3, _, _ = detector.findDistance(20, 4, img, draw=False)
                print(distance3)

                # if distance is less than a particular value we want it to click. This can be fine-tuned to the
                # distance between the webcam and the hand.
                if distance1 < 50:
                    keyboard.press(button.text)
                    # cv2.rectangle(img= image to draw on, (x,y)--start pts., (x,y)--end pts., (RGB color), thickness)
                    cv2.rectangle(img, button.pos, (x + width, y + height), (0, 200, 0), cv2.FILLED)
                    # cv2.putText(img= image to draw on, "TEXT_TO_PUT", (x,y)--pos of text, Font, Font_size, Color,
                    # Thickness)
                    cv2.putText(img, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    # text to be printed on the screen
                    textToPrint += button.text
                    # force a small delay in printing to make it a bit more natural.
                    sleep(0.30)

                elif 95 < distance2 < 125:
                    keyboard.press(Key.backspace)
                    sleep(0.25)

                elif 360 < distance3 < 450:
                    keyboard.press(Key.space)
                    sleep(0.50)

    # display the image being captured
    cv2.imshow("Image", img)

    # Required to use webcam of the device.
    cv2.imshow("Image", img)
    # waitkey() waits until a key is pressed here (q) to exit from creating continuous frames.
    if cv2.waitKey(1) & 0xFF == ord('q'):  # very important if you want a continuous array of frames
        break
