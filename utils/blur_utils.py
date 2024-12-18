import cv2

def blur_faces_and_plates(frame):
    # 加载 Haar 级联分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    # 转为灰度图像以进行检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # 模糊人脸区域
        face_roi = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(face_roi, (99, 99), 30)

    # 检测车牌
    plates = plate_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in plates:
        # 模糊车牌区域
        plate_roi = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(plate_roi, (99, 99), 30)

    return frame
