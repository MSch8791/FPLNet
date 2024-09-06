# This file is part of the FPLNet project.
# It is subject to the license terms in the LICENSE file found in the repository root directory.

__author__= 'MSch8791'

import cv2
import numpy as np

from utils.ONNXModel import ONNXModel

class FPLNetInference:
    def __init__(self, onnxModel, modelInputShape=(256, 256, 3), nbC=4):
        self.model              = onnxModel
        self.modelInputShape    = modelInputShape
        self.nbC                = nbC
    
    def __checkAndFitBBoxInBoundaries(self, imgW, imgH, bbox, acceptableOffset=0.75):
        retBbox = bbox
        acceptableOffsetAmountX = bbox[2] * acceptableOffset
        acceptableOffsetAmountY = bbox[3] * acceptableOffset
        
        if (bbox[2] == 0 or bbox[3] == 0):
            raise ValueError('Dimension W and/or H is 0')

        if (retBbox[0] < 0 and retBbox[0] + retBbox[2] < 0): raise ValueError('Bbox fully out of bounds')
        if (retBbox[1] < 0 and retBbox[1] + retBbox[3] < 0): raise ValueError('Bbox fully out of bounds')
        if (retBbox[0] > imgW): raise ValueError('Bbox fully out of bounds')
        if (retBbox[1] > imgH): raise ValueError('Bbox fully out of bounds')
        
        if (retBbox[0] < 0 and -retBbox[0] <= acceptableOffsetAmountX): retBbox[0] = 0
        elif (retBbox[0] < 0 and -retBbox[0] > acceptableOffsetAmountX): raise ValueError('Bbox not valid, beyond acceptable offset')
        
        if (retBbox[1] < 0 and -retBbox[1] <= acceptableOffsetAmountY): retBbox[1] = 0
        elif (retBbox[1] < 0 and -retBbox[1] > acceptableOffsetAmountY): raise ValueError('Bbox not valid, beyond acceptable offset')

        if (retBbox[0] + retBbox[2] > imgW and retBbox[0] + retBbox[2] - imgW <= acceptableOffsetAmountX): retBbox[2] -= (retBbox[0] + retBbox[2] - imgW)
        if (retBbox[1] + retBbox[3] > imgH and retBbox[1] + retBbox[3] - imgH <= acceptableOffsetAmountY): retBbox[3] -= (retBbox[1] + retBbox[3] - imgH)

        return retBbox

    def __cropFace(self, image, faceBbox, marginX=0.0, marginY=0.0):
        cx = int(faceBbox[0] + (faceBbox[2] / 2))
        cy = int(faceBbox[1] + (faceBbox[3] / 2))
        w = faceBbox[2] + int(faceBbox[2] * marginX)
        h = faceBbox[3] + int(faceBbox[3] * marginY)
        x = int(cx - w // 2)
        y = int(cy - h // 2)

        bbox = self.__checkAndFitBBoxInBoundaries(image.shape[1], image.shape[0], [x, y, w, h])

        cropImg = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], ...]
        
        return cropImg, bbox
    
    def __preprocess(self, image, boNormalize, boGrayscale):
        retImg = cv2.resize(image, (self.modelInputShape[1], self.modelInputShape[0]))

        if boGrayscale == True and retImg.shape[2] == 3:
            retImg = cv2.cvtColor(retImg, cv2.COLOR_BGR2GRAY)
            retImg = np.stack([retImg, retImg, retImg], axis=2)
        elif boGrayscale == False and retImg.shape[2] == 1:
            retImg = cv2.cvtColor(retImg, cv2.COLOR_GRAY2BGR)

        if boNormalize == True:
            retImg = retImg / 127.5 - 1.0

        retImg = retImg.astype(dtype=np.float32)

        return retImg
    
    def __extractLandmarksFromHeatmaps(self, tensorLM, faceBbox):
        landmarks = np.zeros((tensorLM.shape[-1], 2))

        for i in range(0, tensorLM.shape[-1]):
            _, _, _, max_loc = cv2.minMaxLoc(np.expand_dims(tensorLM[..., i], axis=2))
            x = max_loc[0] / self.modelInputShape[1]
            y = max_loc[1] / self.modelInputShape[0]
            landmarks[i, 0] = faceBbox[0] + x * faceBbox[2]
            landmarks[i, 1] = faceBbox[1] + y * faceBbox[3]

        return landmarks
    
    def predict(self, image, facesBboxes):
        retFacesBboxes      = []
        retFacesParsing     = []
        retFacesLandmarks   = []

        inputData   = []
        for i in range(0, len(facesBboxes)):
            croppedFaceImg, bbox = self.__cropFace(image, facesBboxes[i], marginX=0.5, marginY=0.5)
            retFacesBboxes.append(bbox)
            inputData.append(self.__preprocess(croppedFaceImg, True, False))
        
        inputData = np.asarray(inputData)
        predictions = self.model.predict([inputData])

        for i in range(0, len(facesBboxes)):
            retFacesParsing.append(predictions[self.nbC-1][i, ...])
            retFacesLandmarks.append(self.__extractLandmarksFromHeatmaps(predictions[self.nbC-2][i, ...], retFacesBboxes[i]))

        return retFacesParsing, retFacesLandmarks, retFacesBboxes
    