import onnxruntime as nxrun
import numpy as np


class Network:
    def load_model(self, model_path) -> None:
        """
        Load the ONNX model form path in ONNX Session
        """

        # define the priority order for the execution providers
        # prefer CUDA Execution Provider over CPU Execution Provider
        providers = [
            # "OpenVINOExecutionProvider",
                     "CPUExecutionProvider"]
        ## start inference session
        self.sess = nxrun.InferenceSession(model_path, providers=providers)

        ## input, output shape
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape
        self.outname = [i.name for i in self.sess.get_outputs()]

    def inference(self, input, output_name=None):
        """
        Do ONNXruntime inference on input numpy array
        """
        ## run onnx model with onnx runtime python
        result = self.sess.run(
            self.outname, {self.input_name: input.astype(np.float32)}
        )[0]
        result = np.array(result).astype(np.float32)
        return result
