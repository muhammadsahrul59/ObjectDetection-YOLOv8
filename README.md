# Object Detection using YOLOv8✨

This project demonstrates how to use YOLOv8 to perform object detection on a custom dataset. The project uses Label Studio to label the images, and then exports the labels to YOLO format. The model is then trained on the labeled data, and the best model is saved as **_best.pt_**. The **_obj_detection_webcam.py_** script can then be used to perform object detection on a webcam.

## Prerequisites

- Install [Python 3.8 or higher](https://www.python.org/downloads/)
- Install [Anaconda](https://www.anaconda.com/download)
- Install [Label Studio](https://labelstud.io/guide/install)
```
conda create --name label-studio
conda activate label-studio
conda install psycopg2  # required for LS 1.7.2 only
pip install label-studio
```
- Install [Ultralytics YOLOv8](https://docs.ultralytics.com/quickstart/)
```
pip install ultralytics
```
## Labeling Images
To label images with Label Studio, open the Label Studio web application and create a new project. Then, add the images you want to label to the project.

To label an image, click on the image and then click on the object you want to label. You can then draw a bounding box around the object and add a label.

Once you have labeled all of the images in your project, you can export the labels to YOLO format

## Object Detection Images Demos
![val_batch1_pred](https://github.com/muhammadsahrul59/ObjectDetection_YOLOv8/assets/101655285/f034afc8-4217-4df3-abfa-6db4a69f411a)

## Object Decection Videos Demos
![proyekobjectdetectfinal](https://github.com/muhammadsahrul59/ObjectDetection_YOLOv8/assets/101655285/d15f4fe1-7e1a-4fe7-83ac-f17910fa3657)

## Object Detection Webcam Demos

## Run the notebook only on Google Colab
```
- Download the file above
- open https://colab.research.google.com/
- Upload the file obj_detection_yolov8.ipynb
```

or you can open it with this Google Colab Link below :

[![](https://img.shields.io/badge/Google%20colab-Open-FFC000.svg?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAMAAAAolt3jAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAB8lBMVEUAAAD/7QL/6w7/6gz/6w3/6Qj/6Qj/6wj/6gf/6w//6w7/6xP/7gP/2hr/2xX/2jD/z0j/uHX/1zn/3Sz/3Sv/1zn/r4L/zE3/2jL/4ST/4Cf/0RP//zH/3S//2jL/2jL/2zH/3iv/3in/2zD/2jL/2jL/3C///0b/1B3/yxL/xAj/0SH/2DT/2TX/2TX/3C7/2jP/2DX/2DX/2DX/1zD/zRb/YwD/wgf/wQf/xAr/1iX/3C//3C//2zH/2Db/2jP/3C7/2zD/zBf/wgj/wgf/wgf/wQf/wQf/0ST/2y7/5B3/2TT/2TT/3iv/2y//zSD/wQb/wQf/wgf/wgf/wQf/wQf/zx7/3Cz/4x//2TT/2TX/4TD/zRb/wQf/wgf/wQf/wgf/wgf/wQf/wwr/1iT/3S//3C7/2zH/1zH/zBb/wQT/wgf/wQf/wQf/wgf//4r/xAj/0CD/2DT/2TX/2TX/3C7/zBb/wgj/wQf/wQf/wQf/wgf/xgf/0BL/6Rz/3C//2jL/2jP/2zH/3iz/vwD/wgf/wgf/wgf/wgf/xAf/wgf/2Rr/2hf/2TH/t3f//wD/1D//3S7/wgX/wAf/0wj/uQf/wQf/wwf/wgf/5gX/5h//6BT/5xb/6Q//6RD/xAf/xAf/xQf/xAf/xQf/xAf///9tAmifAAAApXRSTlMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATiFmGgWFGaZi0EDAABa6v/5+32y//r/8WoBJ9viYD9wnv+kPlvc5TRg/XsAAArH4yEAAGz+cmL9dgAACsjhHgAAZ/50LOHcUzRbm/+VLUjT6TcAZvH89Pd5uv/x+fRxAgADRJSldRsYcKOVSAQAAAAAAAEAAAAAAQAAAAAAAAAAAAAAAAAAAABydb0CAAAAAWJLR0SlLrlKLwAAAAd0SU1FB+QIEQUaC7jgabcAAAABb3JOVAHPoneaAAAAwUlEQVQI12NgQAOMTMwsrGzsHJxc3DwMDLx8/AKCQsIiomLiEpJSDNIysnLyCopKyiqqauoaDJpa2jq6evoGhkbGJqZmDOYWllbWNrZ29g6OTs4uDK5u7h6eXt4+vn7+AYFBDMEhoWHhEZFR0TGxcfEJDIlJySmpaekZmVnZObl5DPkFhUXFJaVl5RWVVdU1DLV19Q2NTc0trW3tHZ1dDN09vX39EyZOmjxl6rTpMxgYZs6aPWfuvPkLFi5avATdAwCl1jTbaxUL4wAAAF5lWElmSUkqAAgAAAADABIBAwABAAAAAQAAADEBAgANAAAAMgAAAGmHBAABAAAAQAAAAAAAAABQaG90b3MgMi43LjAAAAIAAqAJAAEAAADTAAAAA6AJAAEAAADTAAAAAAAAAJOs9JgAAAAldEVYdGRhdGU6Y3JlYXRlADIwMjAtMDgtMTdUMDU6MjY6MTErMDA6MDBd4jn+AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDIwLTA4LTE3VDA1OjI2OjExKzAwOjAwLL+BQgAAABh0RVh0ZXhpZjpFeGlmSW1hZ2VMZW5ndGgAMjExdjfLjwAAABd0RVh0ZXhpZjpFeGlmSW1hZ2VXaWR0aAAyMTHrqE6dAAAAEnRFWHRleGlmOkV4aWZPZmZzZXQANjTZeQZNAAAAGnRFWHRleGlmOlNvZnR3YXJlAFBob3RvcyAyLjcuMIcagsYAAAAASUVORK5CYII=)](https://colab.research.google.com/drive/1blRBKiPqSeCnm_e9WAHJQKN_8FKo2ugL?usp=sharing)
