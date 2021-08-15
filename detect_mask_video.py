# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import utils.video_loader as video_loader
import numpy as np
import imutils
# import time
import cv2
# import os


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (400, 400),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    # classes = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (400, 400))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        # classes = classNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# load our serialized face detector model from disk
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector_testing6.tf")
# maskNet = load_model("mask_detector_classifier.model")
# classNet = load_model("mask_classifier.model")

# initialize the video stream
vs = video_loader.load_video()

# setup counter
# frame_history = {}
# frame_history['n95_mask'] = []
# frame_history['no_mask'] = []
# frame_history['op_mask'] = []
# n_mask = {}
# n_mask['n95_mask'] = 0
# n_mask['no_mask'] = 0
# n_mask['op_mask'] = 0
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = video_loader.transform(frame)
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    # found_mask = {}
    # found_mask['n95_mask'] = 0
    # found_mask['no_mask'] = 0
    # found_mask['op_mask'] = 0
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (n95_mask, no_mask, op_mask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        mask = max(n95_mask, no_mask, op_mask)
        if(mask == n95_mask):
            label = "N95"
            color = (0, 255, 0)
            # found_mask['n95_mask'] += 1
        elif(mask == no_mask):
            label = "No Mask"
            color = (0, 0, 255)
            # found_mask['no_mask'] += 1
        else:
            label = "OP"
            color = (0, 255, 255)
            # found_mask['op_mask'] += 1

        print("n95 {:.2f}%, no_mask {:.2f}%, op {:.2f}%".format(n95_mask*100, no_mask*100, op_mask*100))
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(n95_mask, no_mask, op_mask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # for mask_type in ['n95_mask', 'no_mask', 'op_mask']:
    #     if len(frame_history[mask_type])<2:
    #         frame_history[mask_type].append(found_mask[mask_type])
    #     else:
    #         frame_history[mask_type].pop(0)
    #         frame_history[mask_type].append(found_mask[mask_type])
    #     diff = max(frame_history[mask_type]) - min(frame_history[mask_type])
    #     n_mask[mask_type] += diff
    # print('n95: {}, none: {}, op: {}'.format(n_mask['n95_mask'], n_mask['no_mask'], n_mask['op_mask']))

    # show the output frame
    cv2.imshow('Mask Detector', frame)

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
