import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def detect_faces_with_gender(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_count = 0

    for (x, y, w, h) in faces:
        face_count += 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if w > 100:
            gender = "Male"
        else:
            gender = "Female"

        cv2.putText(image, f'Gender: {gender}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(image, f'Faces detected: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = detect_faces_with_gender(frame)
    cv2.imshow('Face Detection and Gender Estimation', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
