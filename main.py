import imutils
import cv2
import numpy as np
from imutils.video import FPS
import time



#Initialize video stream and FPS counter
vs = cv2.VideoCapture("C:\\Users\\Koussay\\Desktop\\Proj\\Car tracking and speed estimation\\Object detection\\cars.mp4")
fps=None

#Initialize the model
prototxt_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"
net=cv2.dnn.readNetFromCaffe(prototxt_path,model_path)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

#Initialize frame dimensions
W=None
H=None

#Initialize the total frames which will help us decide
#wether to go for detection or tracking
totalFrames=0

#Initialize tracker
tracker = cv2.TrackerMOSSE_create()


while (vs.isOpened()):
    time.sleep(0.2)
    ret,frame = vs.read()
    # Start the fps estimator
    fps = FPS().start()
    frame = imutils.resize(frame,width=500,height=300)
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if H is None or W is None:
        (H, W) = frame.shape[:2]
    totalFrames+=1

    if totalFrames % 10 == 0 :
        blob= cv2.dnn.blobFromImage(frame,size=(300,300), ddepth=cv2.CV_8U)
        net.setInput(blob,scalefactor=1.0/127.5, mean=[127.5, 127.5, 127.5])
        detections=net.forward()

        cols = frame.shape[1]
        rows = frame.shape[0]
        tracker = cv2.TrackerMOSSE_create()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                label = int(detections[0, 0, i, 1])
                # if the class label is not a car, ignore it
                if CLASSES[label] != "car":
                    continue
                # grab the bounding box points and multiply them by images size because the points are normalized
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                tracker.init(frame,(startX, startY, endX, endY))


    else:
        (success,box)= tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            w=int(w/2.8)
            h=int(h/2.8)
            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (0, 255, 0), 2)

    fps.stop()
    fps.update()

    # Display information on the screen
    info = [
        ("Tracking Success", "Yes" if success else "No"),
        ("FPS", "{:.4}".format(fps.fps())),
    ]


    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (20,i*30+35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imshow("Video", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()


