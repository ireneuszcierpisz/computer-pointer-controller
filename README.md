# Computer Pointer Controller

This application is build to run multiple models in the **OpenVino toolkit** on the same machine to **control a computer pointer using eye gaze**.

## Project Set Up and Installation

Folder `src` contains seven files. Four files with model class and its methods. Also `mouse_controller.py`, `input_feeder.py` and main file `main.py`.

There are four OpenVino models in the `models` folder:

    - [Face Detection (FP32-INT1)](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
    - [Head Pose Estimation (FP32)](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
    - [Facial Landmarks Detection (FP32)](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
    - [Gaze Estimation (FP32)](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

To build the project I used the **InferenceEngine API from Intel's OpenVino toolkit**.

## Demo

To run the app a basic demo the demo.mp4 file can be used located `bin/demo.mp4` 

Command to run inference on four models:   `python main.py`

The default arguments: --modelF, --modelG, --modelH, --modelL, --input_type "video", --input_file "bin/demo.mp4", --mouse_speed, --mouse_precision, --device "CPU"


## Documentation

- usage: main.py [-h]   [--modelF MODELF]
                        [--modelG MODELG]
                        [--modelH MODELH]
                        [--modelL MODELL]
                        [--device DEVICE]
                        [--input_type INPUT_TYPE]
                        [--input_file INPUT_FILE]
                        [--mouse_speed MOUSE_SPEED]
                        [--mouse_precision MOUSE_PRECISION]
- optional arguments:
  -h, --help           show this help message and exit
  --modelF MODELF      The location of the model XML file
  --modelG MODELG
  --modelH MODELH
  --modelL MODELL
  --device DEVICE
  --input_type INPUT_TYPE
                       'video', 'image' or 'cam'
  --input_file INPUT_FILE
                       The location of the video or image file.
                       'None' for input type: 'cam'
  --mouse_speed MOUSE_SPEED
  --mouse_precision MOUSE_PRECISION


## Benchmarks
*TODO: Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.*

## Results
*TODO: Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.*

## Stand Out Suggestions
*This is where you can provide information about the stand out suggestions that you have attempted.*

### Async Inference
*If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.*

### Edge Cases

There will be certain situations that will break inference flow. For instance, multiple people in the frame. 

In that case if multiple faces in the same input frame is detected this app choose one face that which is detected with bigest confidence.
