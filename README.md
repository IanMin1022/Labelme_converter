# Labelme_converter
This repo is a utilized version of pylabel and labelme2coco .

## Instruction 
Follow the steps to convert your labelme data either to coco format or yolo format

### 1. Download the git
``` python
  git clone https://github.com/IanMin1022/Labelme_converter.git
```

### 2. From the git directory, use pip to install prerequisites
``` python
  pip install -r requirements.txt
```

### 3. Execute main.py
Arguments are as followed
``` python
  input_dir: path to input data
  output_dir: path for results
  train_rate: train and validation data rate (if train_rate = 0.85, train data is 85%)
  format: COCO or YOLO maybe it will be extended later
```
#### For coco data
``` python
  python3 main.py --i path_to_input_data -o path_to_result  -f "COCO" --train_rate 0.8
  python3 main.py --i path_to_input_data -o path_to_result  -f "COCO" 
```

#### For yolov data
``` python
  python3 main.py --i path_to_input_data -o path_to_result  -f "YOLO" --train_rate 0.8
  python3 main.py --i path_to_input_data -o path_to_result  -f "YOLO" 
```
