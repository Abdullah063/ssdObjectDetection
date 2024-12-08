import cv2
import numpy as np
thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 150)

classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("Loaded Classes:", classNames)

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    # Konsola tespit edilen nesne bilgilerini yazdır
    print("\nDetected Objects:")
    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            label = classNames[classIds[i][0] - 1].upper()
            confidence = confs[i]
            print(f"Class: {label}, Confidence: {confidence:.2f}, Box: {box}")
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x + 10, y + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        print("No objects detected.")

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basıldığında çık
        break

cap.release()
cv2.destroyAllWindows()