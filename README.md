## Introduction

### Goal 

This tutorial will show you how you can use the RealSense camera on MuSHR cars to detect people and visualize their positions in the car's reference frame.
By the end of this tutorial, you should have a working python package that uses SSD MobileNet V2 to perform
high speed inferencing and returns people's 3D coordinates.

### Requirements

- Jetson Nano with JetPack 4.2 or newer (Ubuntu 18.04)
- Intel RealSense Camera

## Setting up Jetson-Inference package

The github repository that needs to be setup for using object detection is [`jetson_inference`](https://github.com/dusty-nv/jetson-inference).
1. Please follow the specific steps for building this project from source: [`instructions`](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md).
   
   **IMPORTANT**: When prompted to choose models to download as part of setup, you only have to choose `SSD-Mobilenet-v2`. 
   
   You can also feel free to skip the **PyTorch** installation for now if running inferences is your only objective.
   See parts of repo that refers to retraining models or using custom models if interested.
   
2. Once done building the project from source, open a python interpreter in any terminal window and enter the following:
    
    ```
    import jetson.inference
    import jetson.utils
    ```
   
   If you were able to import these packages **without** any errors you should be ready to move on! If you do see any import errors, please refer to the repository's setup again.
    
## Getting Started with People Detection

1. Please clone the following repository that contains the python package [`people-detection`](https://github.com/aditjha/people-detection).

    This repo contains a `src` directory that holds the code necessary to run object detections and 3D position estimates.

2. Install the following package used for calculating 3D coordinates:
    ```
    sudo apt-get update
    sudo apt-get install python-image-geometry
    ```
    Once again, you can check if this package is available to you by `import image_geometry`.
 
## Running All Necessary Components
1. Once everything is setup, we should be able to run the source code and watch the car detect people!

    Run the following commands in *separate* terminal windows (or tmux panes):
    ```
    roslaunch mushr_base teleop.launch
    rviz
    ```
    - The realsense camera on the car should be now activated, and the `/camera/color/image_raw` and 
    `/camera/depth/image_rect_raw` topics should be having data published to by the camera. This can be
    checked by running the commands:
    ```
    rostopic echo /camera/color/image_raw
    rostopic echo /camera/depth/image_rect_raw
   ```
    
    - Once rviz is running, add the Robot Model as a topic to see a MuSHR car at (0,0,0), aka the center of the grid.
    
2. In another terminal window:
    ```
     cd people-detection/src/
     python main.py
    ``` 
    *Please be sure to run with *python2.7* for best compatibility.*
    
    - **NOTE**: The *first* time the code is ran, it will take a while for the SSD model to be configured, however
the following times this process is skipped as the model is stored as cache automatically.

    - On your terminal screen, you should start to see continuous output of the current inference FPS along with a set of
coordinates representing any people detected in the camera feed of the car.
    
3. To see the detected bounding boxes with the confidence and labels, please go to rviz and subscribe
to the topic `detected_image` where you should see a live stream of inference results.

{{< figure src="/tutorials/people_detection/sample-inference.jpg" width="800" >}}

4. To see Marker Spheres representing the detected people and their relative poses to the car, please go to rviz
and subscribe to the topic `visualization_markers`. If there are people detected in the frame, you should
be able to see a green sphere that moves as the person moves in front of the camera. 

**NOTE**: Because the inference FPS is slower than the actual RealSense FPS, there is a slight delay in the workflow.

5. At this point, you should have real-time inferencing happening using SSD-Mobilenet-v2 and the camera feed from the RealSense.
You should be seeing output printed on the terminal screen, a car at (0,0,0) and moving green spheres for any detected people!

That's it! Now you have the capability of running real time object detection on MuSHR cars with 3D pose estimates in the 
car's reference frame! 
