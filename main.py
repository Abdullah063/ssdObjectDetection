from nms import *
from objectDetectionClass import objrctDetect

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# cap.set(10,70)

while True:
    success, img = cap.read()
    sonuc,objectInfo = objrctDetect(img, 0.45, objects=["portakal", "cep telefonu"])

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()