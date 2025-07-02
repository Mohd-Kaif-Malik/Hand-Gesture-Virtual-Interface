import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

buttons = [
    {"pos": (100, 100), "size": (150, 100), "color": (0, 0, 255)},
    {"pos": (300, 100), "size": (150, 100), "color": (0, 0, 255)},
    {"pos": (500, 100), "size": (150, 100), "color": (0, 0, 255)},
]

def is_inside(x, y, btn):
    bx, by = btn["pos"]
    bw, bh = btn["size"]
    return bx <= x <= bx + bw and by <= y <= by + bh

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1) as hands:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)

                cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)

                for btn in buttons:
                    if is_inside(x, y, btn):
                        btn["color"] = (0, 255, 0)
                    else:
                        btn["color"] = (0, 0, 255)

        for i, btn in enumerate(buttons):
            x, y = btn["pos"]
            w_, h_ = btn["size"]
            cv2.rectangle(frame, (x, y), (x + w_, y + h_), btn["color"], -1)
            cv2.putText(frame, f"Button {i+1}", (x + 10, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Virtual Buttons", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
