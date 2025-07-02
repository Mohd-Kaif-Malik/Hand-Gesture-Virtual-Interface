import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGER_TIPS = [8, 12, 16, 20]
THUMB_TIP = 4

def count_fingers(hand_landmarks):
    fingers = []
    if hand_landmarks.landmark[THUMB_TIP].x < hand_landmarks.landmark[THUMB_TIP - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for tip in FINGER_TIPS:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        finger_count = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_count = count_fingers(hand_landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.rectangle(frame, (0,0), (150,100), (0,0,0), -1)
        cv2.putText(frame, str(finger_count), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 5)
        cv2.imshow("Virtual Hand Counting", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
