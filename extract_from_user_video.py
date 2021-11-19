# Import Libraries
from scipy.spatial import distance as dist
import numpy as np
import pandas as pd
import progressbar
import cv2


def output_user_keypoints(video_path, npy_path, out_path, threshold, model):
    if model=="body":
        proto_file = "/home/itsc/openpose-1.7.0/models/pose/body_25/pose_deploy.prototxt"
        weights_file = "/home/itsc/openpose-1.7.0/models/pose/body_25/pose_iter_584000.caffemodel"
        BODY_PARTS = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                        5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                        10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                        15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                        20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}
        POSE_PAIRS = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                        [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                        [11, 24], [22, 24], [23, 24]]

    elif model=="coco":
        proto_file = "/home/itsc/openpose-1.7.0/models/pose/coco/pose_deploy_linevec.prototxt"
        weights_file = "/home/itsc/openpose-1.7.0/models/pose/coco/pose_iter_440000.caffemodel"
        BODY_PARTS = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                    10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                    15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}
        POSE_PAIRS = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                    [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

    else : 
        protoFile = "/home/itsc/openpose-1.7.0/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "/home/itsc/openpose-1.7.0/models/pose/mpi/pose_iter_160000.caffemodel"
        BODY_PARTS = {0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                    10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest",
                    15: "Background"}
        POSE_PAIRS = [[0, 1], [1, 2], [1, 5], [1, 14], [2, 3], [3, 4], [5, 6],
                        [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [14, 8], [14, 11]]

    print(video_path, "\n")
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ok, frame = cap.read()

    # receive height and width of the original image
    (frameHeight, frameWidth) = frame.shape[:2]
    h = 500
    w = int((h/frameHeight) * frameWidth)

    # Set up the progressbar
    widgets = ["--[INFO]-- Analyzing Video: ", progressbar.Percentage(), " ",
            progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval = n_frames,
                                widgets=widgets).start()
    p = 0

    # Load the model and the weights
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # Dimensions for inputing into the model
    inHeight = 368
    inWidth = 368

    # Define the output
    output = cv2.VideoWriter(out_path, 0, fps, (w, h))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    (f_h, f_w) = (h, w)
    zeros = None

    data = []
    if (model == "MPII"):
        previous_x, previous_y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif (model == "COCO"):
        previous_x, previous_y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    else:
        previous_x, previous_y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    while True:
        ok, frame = cap.read()
        if ok != True:
            break

        frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)    
        frame = np.copy(frame)
        
        # Input the frame into the model
        inputBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inputBlob)
        output = net.forward()
        # The third dimension is the height of the output map.
        H = output.shape[2]
        # The fourth dimension is the width of the output map.
        W = output.shape[3]

        points = []
        x_data, y_data = [], []
        minX, minY = 1000, 1000
        maxX, maxY = 0, 0

        for i in range(len(BODY_PARTS)):
            # confidence map of body parts
            probMap = output[0, i, :, :]
            # min, max, min position, max position
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            # position points to match the original image
            x = (w * point[0]) / W
            x = int(x)
            y = (h * point[1]) / H
            y = int(y)

            if prob > threshold: # [pointed]
                cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

                points.append((x, y))

                if(x<minX): minX = x
                if(x>maxX): maxX = x
                if(y<minY): minY = y
                if(y>maxY): maxY = y

                # normalize value based on max and min from each frame
                x = ((x-minX)/(maxX-minX)) if (maxX-minX!=0) else 0
                y = ((y-minY)/(maxY-minY)) if (maxY-minY!=0) else 0

                x_data.append(x)
                y_data.append(y)
                # print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

            else: # [not pointed]
                cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

                points.append(None)
                x_data.append(previous_x[i])
                y_data.append(previous_y[i])
                # print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

        for pair in POSE_PAIRS:
            partA = pair[0]  # 0 (Head)
            partB = pair[1]  # 1 (Neck)
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 0), 3)

        cv2.rectangle(frame, (minX, minY), (maxX, maxY), (255, 0, 0), 2)

        if writer is None:
            writer = cv2.VideoWriter(out_path, fourcc, fps,
                                    (f_w, f_h), True)
            zeros = np.zeros((f_h, f_w), dtype="uint8")
        writer.write(cv2.resize(frame,(f_w, f_h)))
        
        
        cv2.imshow('frame' ,frame)

        data.append(x_data + y_data)
        previous_x, previous_y = x_data, y_data

        p += 1
        pbar.update(p)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Save the output data from the video in .npy format
    np.save(npy_path, data)
    print('save complete')


    pbar.finish()
    cap.release()
    cv2.destroyAllWindows()            
