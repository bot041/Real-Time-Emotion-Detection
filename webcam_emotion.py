import cv2
from hsemotion.facial_emotions import HSEmotionRecognizer

# Load emotion model
fer = HSEmotionRecognizer(
    model_name='enet_b2_8',
    device='cpu'
)

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

# Fullscreen window
cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Emotion Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        try:
            emotion, scores = fer.predict_emotions(
                face,
                logits=False
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            # Show emotion only
            cv2.putText(
                frame,
                emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        except Exception as e:
            print(e)

    cv2.imshow(
        "Emotion Detection",
        frame
    )

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
