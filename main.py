import os
import sys
import time
import cv2
import numpy as np

import torch

from inference import Network
from argparse import ArgumentParser
from utils import non_max_suppression


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="Path to an .onnx model file",
    )
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Path to image or video file"
    )
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for detections filtering" "(0.5 by default)",
    )
    parser.add_argument(
        "-iot",
        "--iou_threshold",
        type=float,
        default=0.5,
        help="io threshold for nvm filtering" "(0.5 by default)",
    )
    return parser


def preprocessing(image, w, h):
    """
    Preprocess input image for inference.

    Changes image shape from [HxWxC] to [1xCxHxW].
    You may need to change code according to your model requirements.
    Parameters:
        image (numpy.ndarray): Image you want to preprocess
        w (int): Width of Image after PreProcessing.
        h (int): Height of Image after PreProcessing.

    Returns:
        image (numpy.ndarray): PreProcessed Image

    """
    img = cv2.resize(image, (w, h))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img)

    # normalize
    im = img.astype(np.float32)
    im /= 255

    return im


def load_class_names(classes_file):
    """
    read file containing classes labels order by the output layer index

    Parameters:
        classes_file (file-like object): class labels file.
    returns:
        classes (list): class labels
    """

    classes = None
    with open(classes_file, "rt") as f:
        classes = f.read().rstrip("\n").split("\n")
    return classes


def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    . Load Model
    . Capture input stream from either camera, video or Image
    . Run Async inference per frame using ONNXRuntime.


    Parameters:
        args: Command line arguments parsed by `build_argparser()`.

    Returns:
        None
    """
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    iou_threshold = args.iou_threshold

    # extract information about model input layer
    (input_height, input_width) = 640, 640

    ### Handle the input stream ###
    # extenstion of input file
    input_extension = os.path.splitext(args.input)[1].lower()
    supported_vid_exts = [".mp4", ".mpeg", ".avi", ".mkv"]
    supported_img_exts = [
        ".bmp",
        ".dib",
        ".jpeg",
        ".jp2",
        ".jpg",
        ".jpe",
        ".png",
        ".pbm",
        ".pgm",
        ".ppm",
        ".sr",
        ".ras",
        ".tiff",
        ".tif",
    ]
    is_image = False
    # if input is camera
    if args.input.upper() == "CAM":
        capture = cv2.VideoCapture(0)

    # if input is video
    elif input_extension in supported_vid_exts:
        capture = cv2.VideoCapture(args.input)

    # if input is image
    elif input_extension in supported_img_exts:
        capture = cv2.VideoCapture(args.input)
        capture.open(args.input)
        is_image = True
    else:
        sys.exit("FATAL ERROR : The format of your input file is not supported")

    # load the onnxruntime model
    net = Network()
    net.load_model(args.model)

    # get classes names
    class_names = load_class_names("bin/classes.txt")

    # OPTIONAL: uncomment for writing output video
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,640))

    # prev_count = 0
    # total_persons = 0
    ### Loop until stream is over ###
    while capture.isOpened():
        ###  Read from the video capture ###
        ret, frame = capture.read()
        if not ret:
            break

        ###  Pre-process the image as needed ###
        image = preprocessing(frame, input_width, input_height)

        start_time = time.time()

        # run inference
        result = net.inference(image)
        print("result: ", result.shape)

        infer_time = time.time() - start_time
        ###  Get the results of the inference request ###

        # show inference time on image
        cv2.putText(
            frame,
            "inference time: {:.5f} ms".format(infer_time),
            (30, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (225, 0, 0),
            1,
        )

        # Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
        out = non_max_suppression(
            torch.from_numpy(result), conf_thres=prob_threshold, iou_thres=iou_threshold
        )
        # define class ids you want to filter from all classes
        required_classes = [0, 1, 2, 3, 5, 7]

        boxed_image = net.apply_threshold(
            frame, out, required_classes, input_width, input_height, class_names
        )
        # show the result image
        cv2.imshow("image", boxed_image)

        if is_image:
            cv2.waitKey(0)

        # OPTIONAL: uncomment for writing output video
        # out.write(ori_images[0])

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    # release resources
    capture.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)


if __name__ == "__main__":
    main()
    sys.exit()
