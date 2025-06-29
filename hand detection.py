import cv2
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture('abc.mp4')
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror image for better UX
    hands, img = detector.findHands(img)  # Detect hands

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        # Display number of fingers
        totalFingers = fingers.count(1)
        cv2.putText(img, f'Fingers: {totalFingers}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Hand Detection", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
