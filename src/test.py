# This file is part of the FPLNet project.
# It is subject to the license terms in the LICENSE file found in the repository root directory.

__author__= 'MSch8791'

import os
import random
import argparse

import numpy as np
import cv2

from utils.ONNXModel import ONNXModel
from face_detection.yunet.FaceDetectorYuNet import FaceDetectorYuNet

from FPLNetInference import FPLNetInference


def getArguments():
    parser = argparse.ArgumentParser(description='FPLNet test program')
    parser.add_argument('--image', type=str, help='The path to an image file')
    parser.add_argument('--output', type=str, help='The path of the file where to save the output image')

    args = parser.parse_args()

    return args

def drawSemanticRegions(predictions, lut):
    nbClasses = min(predictions.shape[-1], len(lut))

    retImg = np.zeros((predictions.shape[0], predictions.shape[1], 3))

    indices = np.argmax(predictions, axis=2)

    for i in range(0, indices.shape[0]):
        for j in range(0, indices.shape[1]):
            if indices[i, j] < nbClasses:
                retImg[i, j, :] = lut[indices[i, j]]
    
    retImg = retImg.astype(dtype=np.uint8)

    return retImg

def postprocess(img, outSize):
    retImg = cv2.resize(img, outSize, cv2.INTER_NEAREST)
    retImg = cv2.blur(retImg, (3, 3))

    return retImg


if __name__ == "__main__":
    np.random.seed(123456)
    random.seed(123456)

    args = getArguments()

    strImageFile            = args.image
    strOutputFile           = args.output
    iNbFaceRegionClasses    = 10
    iNbC                    = 4
    inputShape              = (256, 256, 3)

    # define face parsing regions colors lookup table
    lut = []
    lut.append(np.asarray([0, 0, 0]))
    for i in range(1, iNbFaceRegionClasses):
        color = np.asarray([random.randrange(50, 220, 1), random.randrange(50, 220, 1), random.randrange(50, 220, 1)] )
        lut.append(color)

    print("Loading the models...")
    try:
        faceDet             = FaceDetectorYuNet("face_detection/yunet/model/face_detection_yunet_2023mar.onnx")
        onnxFPLNet          = ONNXModel("../models/fplnet_256_LaPa_4c_20240517.onnx")
        onnxExecProviders   = onnxFPLNet.getAvailableExecutionProviders()
        onnxFPLNet.load(onnxExecProviders)
        FPLNet              = FPLNetInference(onnxFPLNet, inputShape, iNbC)
    except cv2.error as cv2Err:
        print("Failed to load the models. {0}. Terminated.".format(cv2Err))
        exit(1)
    except Exception as err:
        print("Failed to load the models. {0}. Terminated.".format(err))
        exit(1)
    except:
        print("Failed to load the models. An unexpected error occured. Terminated.")
        exit(1)
    print("Models loaded successfully.")

    print("Loading the image...")
    try:
        image = cv2.imread(strImageFile)
    except cv2.error as cv2Err:
        print("Failed to load the image. {0}. Terminated.".format(cv2Err))
        exit(2)
    print("Image loaded successfully.")

    try:
        # detect faces in the given image
        faces = faceDet.predict(image)
    except cv2.error as cv2Err:
        print("An error occured while detecting the faces in the given image. {0}. Terminated.".format(cv2Err))
        exit(3)
    except:
        print("An error occured while detecting the faces in the given image. An unexpected error occured. Terminated.")
        exit(3)

    if len(faces) == 0:
        print("No faces found in the given image. Terminated.")
        exit(4)
    
    faces = [face[0:4].astype(np.int32) for face in faces]

    # predict parsing and landmarks for detected faces. 
    # The method returns also a modified version of the faces bounding boxes.
    facesParsing, facesLandmarks, bboxes = FPLNet.predict(image, faces)
    
    # Draw face parsing segmentation results using the color lookup table for display purpose
    segImg          = np.zeros(image.shape, dtype=np.uint8)
    for i in range(len(bboxes)):
        segFace = postprocess(drawSemanticRegions(facesParsing[i], lut), (bboxes[i][2], bboxes[i][3]))
        roi = segImg[bboxes[i][1]:bboxes[i][1]+bboxes[i][3], bboxes[i][0]:bboxes[i][0]+bboxes[i][2], ...]
        roi = np.where(roi==(0, 0, 0), segFace, roi)
        segImg[bboxes[i][1]:bboxes[i][1]+bboxes[i][3], bboxes[i][0]:bboxes[i][0]+bboxes[i][2], ...] = roi

    # merge image with segmentation image
    outputImage = cv2.addWeighted(image, 0.4, segImg, 0.6, 0)
    
    for i in range(len(faces)):
        # draw original detected face bounding box
        cv2.rectangle(outputImage, (int(faces[i][0]), int(faces[i][1])), (int(faces[i][0]+faces[i][2]), int(faces[i][1]+faces[i][3])), (0, 255, 0), 1)
        # draw face landmarks
        faceLmks = facesLandmarks[i]
        for j in range(len(faceLmks)):
            cv2.circle(outputImage, (int(faceLmks[j, 0]), int(faceLmks[j, 1])), 1, (0, 0, 255), 1)
    
    cv2.imwrite(strOutputFile, outputImage)
    
    print("Done.")