
import cv2
import mediapipe as mp
import time
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera index or drivers.")

pTime = 0 

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )
                h, w, _ = frame.shape
                idx_finger_tip = hand_landmarks.landmark[8]
                cx, cy = int(idx_finger_tip.x * w), int(idx_finger_tip.y * h)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 0), -1)
                cv2.putText(frame, handedness.classification[0].label, (cx+10, cy+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow('Hand Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
