# People Counter App

| Details            |              |
|-----------------------|---------------|
| Programming Language: |  Python 3.10 |
| Code Formatter: |  black |



## Installation
```
pip install -r requirements.txt
```

## Getting model

```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
python export.py --weights yolov5n.pt --include onnx
```

## Running on the CPU

Though by default application runs on CPU

```
python main.py -i bin\padestrians.mp4 -m bin\yolov5n.onnx -pt 0.4
```

learn more about the above command 

```
usage: main.py [-h] -m MODEL -i INPUT [-pt PROB_THRESHOLD] [-iot IOU_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to an .onnx model file
  -i INPUT, --input INPUT
                        Path to image or video file
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering(0.5 by default)
  -iot IOU_THRESHOLD, --iou_threshold IOU_THRESHOLD
                        io threshold for nvm filtering(0.5 by default)
```