
##  Environment configuration
 + *OS version*: Sequoia 15.0
 + *Chip*: M2
 + *Intel RealSense camera model*: Depth Camera D435
 + *Python version*: @3.10
 + *Homebrew version*: 4.4.1
## Prerequisites
 + brew installed.

## Step by step installation
1. `brew install librealsense`
2. `pip install opencv-python`
3. `pip install numpy`

## Important
To run scripts that require connection to the camera, it is essential to run the sudo command, e.g.:

    sudo python3.10 test.py
