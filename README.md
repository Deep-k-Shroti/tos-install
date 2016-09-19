# tos-install
Installation notes for Asus ROG GL552VW - Ubuntu 16.04, Machine learning libraries etc. I have cited necessary resources from where I used scripts, hints etc.

I shall start with a clean Ubuntu 16.04 LTS installation using a USB bootable version.

GL552VW has GTX960m 4GB graphics card. This is supposed to have had many issues with regards to Linux and nVidia drivers installation. I shall list out all the steps I followed to create a stable Ubuntu installation with machine learning/deep learning libraries like Theano,Caffe, OpenCV etc.


Assume that we have the factory settings for GL552VW with windows installed. 

I. Ubuntu installation

PS: The following step is needed only for the first time. For further installations/reinstallation of Ubuntu, this step is not needed. Just check whether the following is done once.
1.Disable fast booting in the bios and enable booting from other devices.


2. Insert the Ubuntu 16.04 bootable USB into the slot. Press F1/F2/Esc to bring up the list of sources from which to boot. Choose the USB drive.

3. This brings up the Ubuntu installation page. Here, choose option "Install Ubuntu" and press "e" to edit the installation option. This shall open up the page of options for installation. Here enter "nouveau.modeset=0" just before "...quiet...". (Ref:http://askubuntu.com/questions/757573/installing-linux-on-rog-gl552vw-beginner). Then press F10 to reboot.

4. Now Ubuntu shall be installed. Follow the onscreen instructions.

5. Once Ubuntu is installed, login to your account, go to System Settings->Software & Updates-> Additional Drivers. Here choose the NVIDIA binary driver version 361.77 for the GTX960m.

6. Now, the laptop is set for installation of other software/drivers.

PS: Please follow the order mentioned here strictly for a proper installation. 

II. NVIDIA CUDA 8.0RC

7. Since June/July-2016 CUDA 8.0RC is the default cuda toolkit for Ubuntu 16.04. 

8. From "https://developer.nvidia.com/cuda-release-candidate-download" choose the options as shown in figure.


9. Install CUDA 8.0RC as suggested for .deb installation.


III. NVIDIA CuDNN library( 5.0).

10. From https://developer.nvidia.com/rdp/cudnn-download
     Choose v5(May27,2016). I found this didnt create problems unlike the next version(This may have been fixed by the time you need to install this. So choose accordingly-Balaji(10/Sep/2016)).

11. You will probably need to register with Nvidia for the download rights. Please do so and login with your id.

12. Once the library is downloaded, unzipping shall create a folder "cuda".(Ref:http://askubuntu.com/questions/767269/how-can-i-install-cudnn-on-ubuntu-16-04/767270)

13. Copy the files:

$ cd folder/extracted/contents
$ sudo cp -P include/cudnn.h /usr/include
$ sudo cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
$ sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*



No need to install any of the following libs for Anaconda+Spyder. We can edit the pythonpath later in spyder to enable easy importing into spyder. 

 
IV. CAFFE
-----------------------------------------------------------------------------------------------
Follow the installation instructions from https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide for caffe.

14.
sudo apt-get update

sudo apt-get upgrade

sudo apt-get install -y build-essential cmake git pkg-config

sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler

sudo apt-get install -y libatlas-base-dev 

sudo apt-get install -y --no-install-recommends libboost-all-dev

sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev

sudo apt-get install -y python-pip

# (Python 2.7 development files)
sudo apt-get install -y python-dev
sudo apt-get install -y python-numpy python-scipy

# (or, Python 3.5 development files)
sudo apt-get install -y python3-dev
sudo apt-get install -y python3-numpy python3-scipy

# (OpenCV 2.4)
sudo apt-get install -y libopencv-dev


Now go to https://github.com/BVLC/caffe and download the .zip archive and unpack it into a folder  /caffe. In a terminal type

cp Makefile.config.example Makefile.config

#Use the following settings-----just copy the following lines into the Makefile.config file.

//////////////////////////////////////////////////////////////////

## Refer to http://caffe.berkeleyvision.org/installation.html
# Contributions simplifying and improving our build system are welcome!

# cuDNN acceleration switch (uncomment to build with cuDNN).
USE_CUDNN := 1

# CPU-only switch (uncomment to build without GPU support).
# CPU_ONLY := 1

# uncomment to disable IO dependencies and corresponding data layers
# USE_OPENCV := 0
# USE_LEVELDB := 0
# USE_LMDB := 0

# uncomment to allow MDB_NOLOCK when reading LMDB files (only if necessary)
#	You should not set this flag if you will be reading LMDBs with any
#	possibility of simultaneous read and write
# ALLOW_LMDB_NOLOCK := 1

# Uncomment if you're using OpenCV 3
# OPENCV_VERSION := 3

# To customize your choice of compiler, uncomment and set the following.
# N.B. the default for Linux is g++ and the default for OSX is clang++
CUSTOM_CXX := g++

# CUDA directory contains bin/ and lib/ directories that we need.
CUDA_DIR := /usr/local/cuda-8.0
# On Ubuntu 14.04, if cuda tools are installed via
# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
# CUDA_DIR := /usr

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 lines for compatibility.
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
		-gencode arch=compute_20,code=sm_21 \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_50,code=compute_50

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
BLAS := atlas
# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

# Homebrew puts openblas in a directory that is not on the standard search path
# BLAS_INCLUDE := $(shell brew --prefix openblas)/include
# BLAS_LIB := $(shell brew --prefix openblas)/lib

# This is required only if you will compile the matlab interface.
# MATLAB directory should contain the mex binary in /bin.
# MATLAB_DIR := /usr/local
# MATLAB_DIR := /Applications/MATLAB_R2012b.app

# NOTE: this is required only if you will compile the python interface.
# We need to be able to find Python.h and numpy/arrayobject.h.
PYTHON_INCLUDE := /usr/include/python2.7 \
		/usr/lib/python2.7/dist-packages/numpy/core/include
# Anaconda Python distribution is quite popular. Include path:
# Verify anaconda location, sometimes it's in root.
# ANACONDA_HOME := $(HOME)/anaconda
# PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
		# $(ANACONDA_HOME)/include/python2.7 \
		# $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include \

# Uncomment to use Python 3 (default is Python 2)
# PYTHON_LIBRARIES := boost_python3 python3.5m
# PYTHON_INCLUDE := /usr/include/python3.5m \
#                 /usr/lib/python3.5/dist-packages/numpy/core/include

# We need to be able to find libpythonX.X.so or .dylib.
PYTHON_LIB := /usr/lib
# PYTHON_LIB := $(ANACONDA_HOME)/lib

# Homebrew installs numpy in a non standard path (keg only)
# PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
# PYTHON_LIB += $(shell brew --prefix numpy)/lib

# Uncomment to support layers written in Python (will link against Python libs)
WITH_PYTHON_LAYER := 1

# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial

# If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies
# INCLUDE_DIRS += $(shell brew --prefix)/include
# LIBRARY_DIRS += $(shell brew --prefix)/lib

# Uncomment to use `pkg-config` to specify OpenCV library paths.
# (Usually not necessary -- OpenCV libraries are normally installed in one of the above $LIBRARY_DIRS.)
# USE_PKG_CONFIG := 1

# N.B. both build and distribute dirs are cleared on `make clean`
BUILD_DIR := build
DISTRIBUTE_DIR := distribute

# Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
# DEBUG := 1

# The ID of the GPU that 'make runtest' will use to run unit tests.
TEST_GPUID := 0

# enable pretty build (comment to see full commands)
Q ?= @ 

//////////////////////////////////////////////////////////////////////////////////////

15. Now create soft links to the following files as below

cd /usr/lib/x86_64-linux-gnu

sudo ln -s libhdf5_serial.so.10.1.0 libhdf5.so

sudo ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so 


16. Enter into the /python directory within caffe

cd python

for req in $(cat requirements.txt); do pip install $req; done

NOTE: If the Ubuntu operating system was updated, perhaps the Python layer needs to be updated and recompiled, because the Python module no longer works. Perform this step again in that case.

for req in $(cat requirements.txt); do pip install $req; done

In case of any problems, try:

for req in $(cat requirements.txt); do sudo -H pip install $req --upgrade; done


17. Now build caffe


cd ..

(now you are in caffe-master directory)

make all

make test

make runtest

make pycaffe     

make distribute

-----------------------------------------------------------------------------------------

This shall have installed Caffe.


V THEANO

Use the following steps to install Theano....
18.

sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git

sudo pip install Theano



VI OpenCV

19. I used OpenCV 2.4.13 and followed instructions from the following 2 links
https://gist.github.com/dynamicguy/3d1fce8dae65e765f7c4
https://gist.github.com/arthurbeggs/06df46af94af7f261513934e56103b30/

20. Use the script below....to install opencv 2.4.13 in ubuntu 16.04

------------------------------------------------------------------------------------------------

# install dependencies
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y cmake
sudo apt-get install -y libgtk2.0-dev
sudo apt-get install -y pkg-config
sudo apt-get install -y python-numpy python-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev
 
sudo apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2 libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils
 
# download opencv-2.4.13
wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.13/opencv-2.4.13.zip
unzip opencv-2.4.13.zip
cd opencv-2.4.13
mkdir release
cd release
 
# compile and install
cmake -G "Unix Makefiles" -DCMAKE_CXX_COMPILER=/usr/bin/g++ CMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_TBB=ON -DBUILD_NEW_PYTHON_SUPPORT=ON -DWITH_V4L=ON -DINSTALL_C_EXAMPLES=ON -DINSTALL_PYTHON_EXAMPLES=ON -DBUILD_EXAMPLES=ON -DWITH_QT=ON -DWITH_OPENGL=ON -DBUILD_FAT_JAVA_LIB=ON -DINSTALL_TO_MANGLED_PATHS=ON -DINSTALL_CREATE_DISTRIB=ON -DINSTALL_TESTS=ON -DENABLE_FAST_MATH=ON -DWITH_IMAGEIO=ON -DBUILD_SHARED_LIBS=OFF -DWITH_GSTREAMER=ON ..
make all -j2 # 2 cores 
sudo make install

---------------------------------------------------------------------------------------------------

21.This shall install opencv with no problems. If import cv2 appears to have an issue, just run

sudo apt-get install -y libopencv-dev




VII Anaconda + Spyder


22. Now all the necessary libraries have been installed in GL552VW, I shall install Anaconda+Spyder.

23. Download Anaconda for Python 2.7 from https://www.continuum.io/downloads.

24. Install Anaconda by typing 
bash Anaconda2-4.1.1-Linux-x86_64.sh 
in the terminal. Follow instructions to set it up in a folder of your choice.

25. Now, Anaconda is setup, but is not aware of the PYTHONPATH for the above installed libraries.

    i. Open Tools->PYTHONPATH manager.
    ii. Add the following lines to the pythonpath

# The following are locations of the above packages and their .so files
# Caffe and its python related libs are here
      /home/balajitn/caffe/python
      /home/balajitn/caffe/distribute/lib
# Theano is installed here
      /usr/local/lib/python2.7/dist-packages
# OpenCV is installed here(assuming folder opencv24)
      /home/balajitn/opencv24/release/lib       



Now import caffe, import cv2 and import theano all work in Anaconda environment with spyder.


 


