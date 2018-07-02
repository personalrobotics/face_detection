# Face Detection
Repository for detecting faces (head pose, mouth etc.)

# Software Versions:

Ubuntu 16.04
C++11
OpenCV 2.7
Dlib 19.4
ROS kinetic 

# Software Dependencies and Installation Instructions:

## 1) OpenCV

### Step 1 : Update packages

sudo apt-get update \
sudo apt-get upgrade 

### Step 2 : Install OS libraries 

#### remove any previous installations of x264 
sudo apt-get remove x264 libx264-dev 

#### we will install dependencies now 

sudo apt-get install build-essential checkinstall cmake pkg-config yasm \
sudo apt-get install git gfortran \
sudo apt-get install libjpeg8-dev libjasper-dev libpng12-dev 

#### If you are using Ubuntu 16.04 

sudo apt-get install libtiff5-dev

sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev \
sudo apt-get install libxine2-dev libv4l-dev \
sudo apt-get install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev \
sudo apt-get install libqt4-dev libgtk2.0-dev libtbb-dev \
sudo apt-get install libatlas-base-dev \
sudo apt-get install libfaac-dev libmp3lame-dev libtheora-dev \
sudo apt-get install libvorbis-dev libxvidcore-dev \
sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev \
sudo apt-get install x264 v4l-utils 

#### Optional dependencies 
sudo apt-get install libprotobuf-dev protobuf-compiler \
sudo apt-get install libgoogle-glog-dev libgflags-dev \
sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen 

### Step 3 : Download opencv and opencv_contrib

We will download opencv and opencv_contrib packages from their github repositories. 

### Step 3.1 : Download opencv from Github

If you have OpenBlas installed on your machine, OpenCV 3.2.0 fails to compile. We will use the 
commit where this bug has been patched. So this is OpenCV 3.2.0 with few bugs patched. 

git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 2b44c0b6493726c465152e1db82cd8e65944d0db
cd ..

### Step 3.2 : Download opencv_contrib from Github

If we use v3.2.0 python module of opencv (cv2) fails to import due to a bug in opencv_contrib. 
Here too we will use a commit where this bug has been patched.

git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout abf44fcccfe2f281b7442dac243e37b7f436d961
cd ..

### Step 4: Compile and install OpenCV with contrib modules 

### Step 4.1 : Create a build directory

cd opencv
mkdir build
cd build

### Step 4.2 : Run CMake

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D WITH_TBB=ON \
-D WITH_V4L=ON \
-D WITH_QT=ON \
-D WITH_OPENGL=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D BUILD_EXAMPLES=ON ..

### Step 4.3 : Compile and Install

### find out number of CPU cores in your machine
nproc
### substitute 4 by output of nproc
make -j4
sudo make install \
sudo sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' \
sudo ldconfig 

## 2) Dlib

### Step 1 : Install OS Libraries

sudo apt-get install build-essential cmake pkg-config \
sudo apt-get install libx11-dev libatlas-base-dev \
sudo apt-get install libgtk-3-dev libboost-python-dev   

### Step 2 : Compile Dlib C++ binaries

wget  http://dlib.net/files/dlib-19.4.tar.bz2
tar xvf dlib-19.4.tar.bz2
cd dlib-19.4/
mkdir build
cd build
cmake ..
cmake --build . --config Release
sudo make install
sudo ldconfig
cd ..

pkg-config --libs --cflags dlib-1

## 3) ROS

Detailed Installation instructions are provided in the link below

http://wiki.ros.org/kinetic/Installation

## 4) Intel RealSense 

Detailed Installation instructions are provided in the link below

https://software.intel.com/en-us/realsense/d400/get-started

## 5) CMake

Detailed Installation instructions are provided in the link below

https://cmake.org/install/

# Pre-trained facial landmark model required

Include the path of this file in the code provided and make changes accroding to where the model is placed. It can be found in the google drive link below.

https://drive.google.com/open?id=1QU6L3vHeN24hwjQ1pVxuNKtbJj-fTvql

