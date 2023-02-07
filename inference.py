import onnxruntime as nxrun
import numpy as np


class Network:
    def load_model(self, model_path) -> None:
        """
        Load the ONNX model form path in ONNX Session 
        """
        ## start inference session
        self.sess = nxrun.InferenceSession(model_path)

        ## input, output shape
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape

    def inference(self, input):
        """
        Do ONNXruntime inference on input numpy array
        """
        ## run onnx model with onnx runtime python
        result = self.sess.run(None, {self.input_name: input})
        result = np.array(result).astype(np.float16)
        return result
