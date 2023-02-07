import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log

from inference import Network
from argparse import ArgumentParser


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
        help="Path to an xml file with a trained model.",
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
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0).astype(np.float32)

    return img


def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    . Load Model
    . Capture input stream from either camera, video or Image
    . Run Async inference per frame using ONNXRuntime.


    Parameters:
        args: Command line arguments parsed by `build_argparser()`.
        threshold (float): The minimum threshold for detections.

    Returns:
        None
    """
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # extract information about model input layer
    (input_height, input_width) = 640, 640

    ### TODO: Handle the input stream ###
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
    else:
        sys.exit("FATAL ERROR : The format of your input file is not supported")
    print(capture)

    # load the onnxruntime model
    net = Network()
    net.load_model(args.model)
    prev_count = 0
    total_persons = 0
    ### TODO: Loop until stream is over ###
    while capture.isOpened():
        ### TODO: Read from the video capture ###
        ret, frame = capture.read()
        if not ret:
            break
        ### TODO: Pre-process the image as needed ###
        image = preprocessing(frame, input_width, input_height)

        start_time = time.time()
        # run inference
        result = net.inference(image)
        print(result.shape)

        infer_time = time.time() - start_time
        ### TODO: Get the results of the inference request ###

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

       
        cv2.imshow("image", frame)
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
