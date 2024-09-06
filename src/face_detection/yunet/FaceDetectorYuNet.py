# This file is part of the FPLNet project.
# It is subject to the license terms in the LICENSE file found in the repository root directory.

__author__= 'MSch8791'

import cv2

class FaceDetectorYuNet:
    def __init__(self, strModelFile):
        self.detModel = cv2.FaceDetectorYN_create(strModelFile, "", (0, 0))
    
    def predict(self, image):
        img = image.copy()

        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        height, width, _ = img.shape
        self.detModel.setInputSize((width, height))
        
        _, faces = self.detModel.detect(img)
        faces = faces if faces is not None else []

        return faces