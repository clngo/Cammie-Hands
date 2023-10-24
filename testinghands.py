#https://youtu.be/vQZ4IvB07ec
#https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/
#https://google.github.io/mediapipe/solutions/hands.html

import copy
import mediapipe as mp
import cv2
import numpy as np
import os
import time
from pynput.keyboard import Key, Controller
keyboard = Controller()
counter = 0


#https://youtu.be/EgjwKM3KzGU this is the angles

#literally just labels, it's not even correct
def get_label(index, hand, results): #index is the number of hands (mainly just index 0 & 1 because two hands)
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        #print(f"index: {index}")
        label = results.multi_handedness[index].classification[0].label #left or right hand
        
        if label == "Left":
            cv2.putText(image, "Left", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if label == "Right":
            cv2.putText(image, "Right", (580, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if index == 1: #ONLY Two hands, but could be Right and Right :/
            cv2.putText(image, "Both", (295, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        score = classification.classification[0].score #estimated probability of predicted handedness
        text = "{} {}".format(label, round(score, 2)) #just converting the label and score into a string

        #Extract Coordinates
        coords = tuple(np.multiply(
            np.array((hand.landmark[mphandss.HandLandmark.WRIST].x, hand.landmark[mphandss.HandLandmark.WRIST].y)), 
            [640, 480]).astype(int))
            #[640, 480] is multiplying the coords so it somehow fits into the dimensions of webcam

        output = text, coords
        return output

thumb = [4, 3, 2]
index = [8, 7, 6]
middle = [12, 11, 10]
ring = [16, 15, 14]
pinky = [20, 19, 18]

joint_list = [thumb, index, middle] # first element is the finger, and each finger has 3 joints. 
ref = False
reference = []

#also drawing bruh
def draw_finger_angles(image, results, joint_list):
    for landmark in results.multi_hand_landmarks: #loop through hands
        for joint in joint_list: #loop through the joints
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) #1st joint
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) #2nd joint
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) #3rd joint

            
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180/np.pi)
            
            if angle > 180:
                angle = 360 - angle

            text = str(round(angle, 2))
            coords = tuple(np.multiply(b, [640, 480]).astype(int))
            cv2.putText(image, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 
            #cv2 font requirements: font size, font color, line width, line type
    return image

#relative graph for the hand
def handgraph(hand, ref):
        
    temphand = copy.deepcopy(hand)
    #print(f"temp_landmark_list: {temp_landmark_list}")

    # Convert to relative coordinates
    #wrist joint x: {hand.landmark[0].x}
    base_x, base_y = 0, 0

    xtempmax = []
    ytempmax = []

    for i in range(21):
        if temphand.landmark[i].x == temphand.landmark[0].x and temphand.landmark[i].y == temphand.landmark[0].y:
            base_x, base_y = temphand.landmark[i].x, temphand.landmark[i].y
        temphand.landmark[i].x -= base_x
        temphand.landmark[i].y -= base_y
        xtempmax.append(temphand.landmark[i].x)
        ytempmax.append(temphand.landmark[i].y)
    #print(f"temphand: {temphand.landmark[1].y}")

    # Normalization
    xmax = [abs(x) for x in xtempmax]
    ymax = [abs(y) for y in ytempmax]
    xmax = max(xmax)
    ymax = max(ymax)
    #print(xmax, ymax)

    for i in range(21):
        temphand.landmark[i].x = (temphand.landmark[i].x / xmax)
        temphand.landmark[i].y = (temphand.landmark[i].y /ymax)
    if ref and len(reference) <= 15: #Getting x and y results to save as reference
            joint_coords = [temphand.landmark[8].y]
            print(f"Results: {joint_coords}")
            reference.append([joint_coords])
    return temphand


#aight so get the POSE
def fingers(relativehand, indexhands, results):
    thumb_joint = relativehand.landmark[4]
    index_joint = relativehand.landmark[8]
    middle_joint = relativehand.landmark[12]
    ring_joint = relativehand.landmark[16]
    pinky_joint = relativehand.landmark[20]

    if thumb_joint.y <= -0.8:
        cv2.putText(image, "thumb", (295, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    elif index_joint.y <= -0.8:
        cv2.putText(image, "index", (295, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    elif middle_joint.y <= -0.8:
        cv2.putText(image, "middle", (295, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    elif ring_joint.y <= -0.7:
        cv2.putText(image, "ring", (295, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    elif pinky_joint.y <= -0.8:
        cv2.putText(image, "pinky", (295, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def numbers(relativehand, indexhands, results):
    for idx, classification in enumerate(results.multi_handedness):
        #print(f"index: {index}")
        label = results.multi_handedness[indexhands].classification[0].label #left or right hand
    thumb_joint = relativehand.landmark[4]
    index_joint = relativehand.landmark[8]
    middle_joint = relativehand.landmark[12]
    ring_joint = relativehand.landmark[16]
    pinky_joint = relativehand.landmark[20]
    count = 0
    if thumb_joint.y <= -0.8 or (thumb_joint.x <= -0.9 and label == "Right") or (thumb_joint.x >= 0.9 and label == "Left"):
        count += 1
    if index_joint.y <= -0.8:
        count += 1
    if middle_joint.y <= -0.8:
        count += 1
    if ring_joint.y <= -0.7:
        count += 1
    if pinky_joint.y <= -0.7:
        count += 1
    return count

def peace(relativehand, indexhands, results):
    for idx, classification in enumerate(results.multi_handedness):
        #print(f"index: {index}")
        label = results.multi_handedness[indexhands].classification[0].label #left or right hand
    thumb = False
    index = False
    middle = False
    ring = False
    pinky = False
    thumb_joint = relativehand.landmark[4]
    index_joint = relativehand.landmark[8]
    middle_joint = relativehand.landmark[12]
    ring_joint = relativehand.landmark[16]
    pinky_joint = relativehand.landmark[20]

    #thumb
    if ((thumb_joint.x >= -0.4 and label == "Right") or (thumb_joint.x <= 0.4 and label == "Left")) and thumb_joint.y >= -0.5:
        #print("thumbx works")
        thumb = True

    #index
    if index_joint.y <= -0.83 and index_joint.y >= -1:
        #print("indexy works")
        index = True
        
    #middle
    if middle_joint.y <= -0.83 and middle_joint.y >= -1:
        #print("middley works")
        middle = True
    
    #ring
    if ring_joint.y >= -0.4:
        ring = True
    
    #pinky
    if pinky_joint.y >= -0.4:
        pinky = True

    if thumb and index and middle and ring and pinky:
        if indexhands == 0 and indexhands != 1:
            cv2.putText(image, "Peace", (295, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

switch = "Right"
def mute(relativehand, indexhands, results, switch):
    
    for idx, classification in enumerate(results.multi_handedness):
        #print(f"index: {index}")
        label = results.multi_handedness[indexhands].classification[0].label #left or right hand
    thumb = False
    index = False
    middle = False
    ring = False
    pinky = False
    thumb_joint = relativehand.landmark[4]
    index_joint = relativehand.landmark[8]
    middle_joint = relativehand.landmark[12]
    ring_joint = relativehand.landmark[16]
    pinky_joint = relativehand.landmark[20]

    #thumb
    if thumb_joint.y >= -1 and thumb_joint.y <= -0.6 and ((thumb_joint.x >= 0 and label == "Right") or (thumb_joint.x <= 0 and label == "Left")):
        #print("thumb works")
        thumb = True

    #index
    if index_joint.y >= -1 and index_joint.y <= -0.7:
        #print("indexy works")
        index = True
        
    #middle
    if middle_joint.y >= -0.4:
        #print("middley works")
        middle = True
    
    #ring
    if ring_joint.y >= -0.4:
        ring = True
    
    #pinky
    if pinky_joint.y >= -0.4:
        pinky = True

    if thumb and index and middle and ring and pinky:
        cv2.putText(image, "mute", (295, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if label == "Right" and switch == "Right":
            switch = "Left"
            return switch
        if label == "Left" and switch == "Left":
            switch = "Right"
            return switch

#Real Time Feed
mp_drawing = mp.solutions.drawing_utils #draw out the detections
mphandss = mp.solutions.hands
pTime = 0
use_brect = True

captureeee = cv2.VideoCapture(0) #from the camera; number could be different 

with mphandss.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5, max_num_hands = 10) as hands:
    while captureeee.isOpened():
        data, frame = captureeee.read() #reading each frame from the camera

        #BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Flip on the Horizontal, easier for me to read, MIRRORS; without = camera view, u can comment this
        image = cv2.flip(image, 1) 

        #FPS 
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS:{int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Sets flag
        image.flags.writeable = False

        #Detections
        results = hands.process(image) #very important, making detections

        #Sets flag to True
        image.flags.writeable = True
        
        #RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

       
        #Detections
        #print(results)

        #Rendering Results
        if results.multi_hand_landmarks: #landmarks are just joints; a hand has 21 landmarks
            for num, hand in enumerate(results.multi_hand_landmarks):
                #print(f"thumb joint x: {hand.landmark[4].x}")
                #print(results.multi_hand_landmarks)
                #print(f"num: {num}") number of hands (0 is 1 handx)
                mp_drawing.draw_landmarks(image, hand, mphandss.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 4),
                mp_drawing.DrawingSpec(color = (250, 44, 250), thickness = 2, circle_radius = 2))
                #Drawing Spec is like the colors of the markings basically
                #https://www.rapidtables.com/web/color/RGB_Color.html for colors
                

                # Render left or right direction
                if get_label(num, hand, results): #Checking if we actually have results
                    text, coords = get_label(num, hand, results)
                    cv2.putText(image, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 
                    #cv2 font requirements: font size, font color, line width, line type

                if cv2.waitKey(1) & 0xFF == ord("t"):
                    ref = True
                relativehand = handgraph(hand, ref)
                #print(f"indexy: {relativehand.landmark[8].y}")


                #poses
                
                count = numbers(relativehand, num, results)
                peace(relativehand, num, results)
                
                if switch == "Right" and mute(relativehand, num, results, switch):
                    mute(relativehand, num, results, switch)
                    switch = mute(relativehand, num, results, switch)
                    print(f"to unmute, use {switch}")


                    # name = "working"
                    # image_name = os.path.join("C:\zstuff\AAACodesVS\\ngoarchives\\cammiehands\\hcammie", f"{name} {str(counter)}.jpg")
                    # cv2.imwrite(image_name, image)
                    # print("snap")
                    counter += 1
                    keyboard.press("`")
                    keyboard.release("`")
                if switch == "Left" and mute(relativehand, num, results, switch):
                    mute(relativehand, num, results, switch)
                    switch = mute(relativehand, num, results, switch)
                    print(f"to mute, use {switch}")

                    
                    # name = "working"
                    # image_name = os.path.join("C:\zstuff\AAACodesVS\\ngoarchives\\cammiehands\\hcammie", f"{name} {str(counter)}.jpg")
                    # cv2.imwrite(image_name, image)
                    # print("snap")
                    counter += 1
                    keyboard.press("`")
                    keyboard.release("`")
                    
            #Draws angles
            if count != 0:
                    cv2.putText(image, str(count), (295, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            draw_finger_angles(image, results, joint_list)


        
        #cv2.imshow("OG", frame) #you can comment this to remove hand landmarks
        cv2.imshow("cammiehandss", image) #shows hand landmarks

        #Saving Image, ig, in the future for some reason
        # counter = 0 gonna try the format counter another time
        # uuid is a unique identifier number for the thing whatever

        if cv2.waitKey(1) & 0xFF == ord("s"):
            name = "working"
            image_name = os.path.join("C:\zstuff\AAACodesVS\\ngoarchives\\cammiehands\\hcammie", f"{name} {str(counter)}.jpg")
            cv2.imwrite(image_name, image)
            print("snap")
            counter += 1

        if cv2.waitKey(1) & 0xFF == ord("q"): #closing by pressing q
            print("closed")
            break




captureeee.release()
cv2.destroyAllWindows()
print(reference)
