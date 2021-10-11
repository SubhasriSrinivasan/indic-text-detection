# Indic Text Detection

This repository contains code to train a model using YOLOv5 to detect text in regional languages.

## Get the Data

1. Download and extract the train (synthetic) and test (real) datasets,
    - Train Set: https://drive.google.com/open?id=1E5kI8CLoC-XffqQMTWwSpBIPp1Wb2tne
    - Test Set: https://drive.google.com/open?id=1Z6Qxr-q-F54iYB2G1AyoDymBh64f5REZ
2. Extracting the train data gives a folder named `Synthetic Train Set - Detection & Recognition/`. Rename this folder to `Train` and move it to the `DataOriginal` folder in this repository.
3. Extracting the test data gives a folder named `real_Image_dataset_Detection/`. Rename this folder to `Test` and move it to the `DataOriginal` folder in this repository.

Your final directory structure should look like this,
```
- DataOriginal/
    - Train/
        - Annotation/
        - Images/
    - Test/
        - Annotation/
        - Images/
```

## Usage

### Convert Data to Yolo

- Our original train data is not in the format as required by YOLO so we run a data convertor script which does this transformation.
- Make sure you have downloaded and restructured the data as mentioned in the [Get Data](#get-data) section above.
- Convert the data with the `data_convertor.py` script,
```bash
python data_convertor.py --source 'DataOriginal' --target 'DataYolo'
```

### Train

1. Open a terminal
2. Clone the [YOLOv5](https://github.com/ultralytics/yolov5/) repository and `cd yolov5`
3. Open the [DataYolo/dataset.yaml](DataYolo/dataset.yaml) file and make sure the `path` is correct according to your system.
4. Run the train script,
```bash
python train.py --img 320 --batch 16 --epochs 3 --data ../DataYolo/dataset.yaml --weights yolov5m.pt
```
5. You can check the training results in the `yolov5/runs/` folder.

For more information on training with yolov5, checkout the [official wiki here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

### Validate on Test Data

Run the `val.py` script to validate the model predictions on test data. _Make sure to replace the weights path_
```bash
python val.py --data ../DataYolo/dataset.yaml --weights path/to/best.pt --img 640 --task test
```
The results are again available under `yolov5/runs/val/` folder

### Test on New Images

To make predictions on new images using the fine-tuned model, run the [Predict.ipynb](Predict.ipynb) notebook.

## References
- Yolo v5: https://github.com/ultralytics/yolov5/