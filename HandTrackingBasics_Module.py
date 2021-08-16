# CREDITS: https://www.youtube.com/watch?v=NZde8Xt78Iw

import cv2
import mediapipe as mp
import time
import math


# Used to detect the hand and define functions to aid the detection process and retrieve valuable info
# from detection.
class handDetector():
    # Basic params required for mp.hands. These values are provided by user.
    def __init__(self, mode=False, maxHands=3, detectionCon=0.5, trackCon=0.5):
        # Represented in same order as order of params in hands pipeline.
        """When we use self.abc --> Basically means that we want to create an object of the class with its
        own variable."""

        self.mode = mode  # We assign value of mode passed by user to the object of the class we work with.
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        """Initializing some values required for hand detection from the mediapipe lib"""

        # mp.solutions.hands is an entire pipeline that can be used to detect 21
        # 3D landmarks of an hand from an single frame
        self.mpHands = mp.solutions.hands

        # mediapipe.Hands(obj params)
        # static_image_mode: false--> auto decision making for tracking and
        # detecting a hand based on some confidence levels.
        # when true, it always detects which makes the program slow.
        # max_num_hands: How many hands will it detect at a time.
        # min_detection_confidence and min_tracking_confidence: Confidence levels for image modes.
        # self.params are set similar to their counterparts in the .Hands pipeline.
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)

        # Use red dots to help us visualize what points its detecting and tracking.
        self.mpDraw = mp.solutions.drawing_utils

        # Tip id numbers for thumb, index, middle, ring, pinky fingers resp.
        self.tipIds = [4, 8, 12, 16, 20]  # REF: https://google.github.io/mediapipe/solutions/hands.html

    def findHands(self, img, draw=True):  # img to find a hand on.
        # As hands object only uses RGB images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # hands.process() will process the image under certain criteria.
        self.results = self.hands.process(imgRGB)  # To use object defined in class above. As def is within same class.

        # Check if we have multiple hands and if so, extract them one-by-one.
        if self.results.multi_hand_landmarks:  # loop will execute when results.multi_hand_landmarks is true.
            for handLms in self.results.multi_hand_landmarks:

                # .draw_landmarks(img = image to draw points on, handLms = the particular
                #               hand if more than 1 to draw on, To show the connections between points)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum=0, draw=True, idx=0):
        if idx == 1:
            self.lmList = []
            xList = []
            yList = []
            bbox = []
            if self.results.multi_hand_landmarks:  # loop will execute when results.multi_hand_landmarks is true.
                myHand = self.results.multi_hand_landmarks[handNum]

                for id, lm in enumerate(myHand.landmark):  # enumerate saves us from keep count, track of each process.

                    # We use x, y landmark(lm) values to find the location of points on the hand.
                    # The x, y obtained are not pixels but ratio's of the image. We multiply them with the
                    # width and length resp; to get a proper pixel location we can work with.
                    height_img, width_img, channels_img = img.shape
                    cord_x, cord_y = int(lm.x * width_img), int(lm.y * height_img)
                    xList.append(cord_x)
                    yList.append(cord_y)
                    self.lmList.append([id, cord_x, cord_y])

                    # To search and detect a specific point on the hand that we've mapped.
                    if draw:
                        # cv2.circle(image=img, centre_at_pt:(x,y)pixel loc, radius, color_circle, shade in circle)
                        cv2.circle(img, (cord_x, cord_y), 8, (255, 0, 200), cv2.FILLED)

            # Have a box on the screen helps user see the area in which the finger can be moved to expect a response.
            if len(self.lmList) != 0:
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

                # Drawing a boundary box around the detected hand. Makes it better to visualize.
                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                                  (0, 255, 0), 2)

            return self.lmList, bbox
        else:
            lmList = []
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[handNum]
                for id, lm in enumerate(myHand.landmark):
                    # We use x, y landmark(lm) values to find the location of points on the hand.
                    # The x, y obtained are not pixels but ratio's of the image. We multiply them with the
                    # width and length resp; to get a proper pixel location we can work with.
                    height_img, width_img, channels_img = img.shape
                    cord_x, cord_y = int(lm.x * width_img), int(lm.y * height_img)
                    lmList.append([id, cord_x, cord_y])
                    if draw:
                        cv2.circle(img, (cord_x, cord_y), 15, (255, 0, 255), cv2.FILLED)

            return lmList

    def fingersUp(self):
        fingers = []
        # Right Thumb
        # In mediapipe pipeline, for 21 hand landmarks, for thumb value decreases from right to left with highest at 0.
        # lmList --> List of attributes.
        # lmList[tipIds[id]--> attribute/ landmark number] [X=1/Y=2/Z=3 positions of landmark]
        # Here we want the X landmark.
        # For example lets consider index finger with tip id number 8. #REF DOCS DIAGRAM
        # Essentially, if the y-value of tip num 8 is less than y-value of tip num 6, the index finger is open.
        # else its close.
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers (Works for both hands)
        for id in range(1, 5):
            # In mediapipe pipeline, for 21 hand landmarks, the highest attribute is at 0 and decrease as we go upwards.
            # lmList --> List of attributes.
            # lmList[tipIds[id]--> attribute/ landmark number] [X=1/Y=2/Z=3 positions of landmark]
            # Here we want the Y landmark.
            # For example lets consider index finger with tip id number 8. #REF DOCS DIAGRAM
            # Essentially, if the y-value of tip num 8 is less than y-value of tip num 6, the index finger is open.
            # else its close.
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)  # appends 1 to the list for each finger that is open in order.
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1) #

        return fingers

    # A basic function used to calculate the distance between two points and display it on the circle or line.
    def findDistance(self, p1, p2, img, draw=True, radius=15, thickness=3):
        x1_cord, y1_cord = self.lmList[p1][1:]
        x2_cord, y2_cord = self.lmList[p2][1:]
        cord_x, cord_y = (x1_cord + x2_cord) // 2, (y1_cord + y2_cord) // 2

        if draw:
            cv2.line(img, (x1_cord, y1_cord), (x2_cord, y2_cord), (255, 0, 255), thickness)
            cv2.circle(img, (x1_cord, y1_cord), radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2_cord, y2_cord), radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cord_x, cord_y), radius, (0, 0, 255), cv2.FILLED)
            #finds hypotenuse of right angled triangle.
        length = math.hypot(x2_cord - x1_cord, y2_cord - y1_cord)

        return length, img, [x1_cord, y1_cord, x2_cord, y2_cord, cord_x, cord_y]


# dummy code to run this.

def main():
    # To track framerate(S1)
    pTime = 0
    cTime = 0
    # Required to use webcam of the device. (s1)
    cap = cv2.VideoCapture(0)
    detector = handDetector()  # obj points to handDetector class

    while True:
        # Required to use webcam of the device.(s2)
        success, img = cap.read()
        img = detector.findHands(img)  # we pass obtained image to findHands in handDetector class.
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[0])  # Instead of 0 any other value for 0-20 can be used to get a diff landmark position

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


if __name__ == "__main__":
    main()
