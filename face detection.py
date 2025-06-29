import cv2
from cvzone.FaceDetectionModule import FaceDetector


cap = cv2.VideoCapture('anime.mp4')
detector = FaceDetector(minDetectionCon=0.5)

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect faces
    img, bboxs = detector.findFaces(img)

    # If faces are found
    if bboxs:
        for bbox in bboxs:
            # bbox['bbox'] gives (x, y, w, h)
            x, y, w, h = bbox['bbox']
            center = bbox['center']
            score = bbox['score'][0]
            print(f"Center: {center}, Score: {score:.2f}")

    # Show the image
    cv2.imshow("Face Detection", img)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
