# This file is part of the FPLNet project.
# It is subject to the license terms in the LICENSE file found in the repository root directory.

__author__= 'MSch8791'

import numpy as np
import onnxruntime as onnxrt

# Wrapper class for ONNX model inference
class ONNXModel:
    def __init__(self, strModelFile):
        self.strModelFile   = strModelFile
        self.boLoaded       = False

    def getAvailableExecutionProviders(self):
        return onnxrt.get_available_providers()

    def load(self, executionProviders):
        if(self.boLoaded == False):
            self.onnxSess = onnxrt.InferenceSession(self.strModelFile, providers=executionProviders)

            self.inputDetails    = self.onnxSess.get_inputs()
            self.outputDetails   = self.onnxSess.get_outputs()

            self.boLoaded = True

            return self.inputDetails, self.outputDetails
        
        return None
    
    def predict(self, inputData):
        inputs  = {}
        outputNames = []

        if(self.boLoaded == True):
            for i in range(0, len(self.inputDetails)):
                inputs[self.inputDetails[i].name] = inputData[i]
            
            for i in range(0, len(self.outputDetails)):
                outputNames.append(self.outputDetails[i].name)

            results = self.onnxSess.run(outputNames, inputs)

            return results
        
        return None