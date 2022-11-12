import mediapipe as mp
import cv2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
#Set Frame Size
cap.set(3, 1280)
cap.set(4, 650)
joint_list = [[2,3,4], [8,7,6]]

def fingerAngles(image, results, joint_list):
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        result = []
        for idx, joint in enumerate(joint_list):
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = round(np.abs(radians*180.0/np.pi), 2)
            
            if angle > 180.0:
                angle = 360-angle

            if idx == 0:
                result.append([angle, 'thumb'])
                colour = (0, 255, 0)
            else:
                result.append([angle, 'index'])
                colour = (0, 0, 255)

    return result
    
#Set the detection confidence and tracking confidence for better result
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(150, 22, 150), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(150, 200, 250), thickness=2, circle_radius=2),
                                        )
                        
            # Draw angles to image from joint list
            val = fingerAngles(image, results, joint_list)
            if abs(val[0][0] - val[1][0]) < 6:
                cv2.putText(image, "shooting", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        #Showing the camera
        cv2.imshow('Finger Angles', image)

cap.release()
cv2.destroyAllWindows()