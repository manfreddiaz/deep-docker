# Deep-Docker

**Dockerfile** for creating a docker image with default configurations for Deep Learning tasks, including:

* Ubuntu 16.04
* Nvidia CUDA 8.0 + cuDNN 5.0 ([hub](https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/8.0/devel/cudnn6/Dockerfile))
* Basic Tools
    * cmake
    * curl, wget
    * git
    * nano, vim
    * unzip, tar
    * ffmpeg
* Languages compilers
    * C++ (g++)
    * Python 2.7
    * Python 3.5
    * Fortran (required by OpenCV)
* Python Machine Learning Libraries
    * scipy
	* nose
    * scikit-image
    * matplotlib
    * pandas
    * sklearn
    * sympy
* OpenCV 3.x (compiled with)
    * CUDA 
    * CUBLAS Support
    * V4L
    * TBB
    * PTHREADS
    * OpenGL
    * OpenCL
    * TIFF
    * GDAL
    * VTK
    * IPP
    * cstripes
    * eigen
    * Python 3.5 bindings
* Tensorflow
    * Version: 1.2.1
    * GPU Support: CUDA 8.0
    * Python: 3.5
* PyTorch:
    * Version: 0.1.12
    * GPU Support: CUDA 8.0
    * Python: 3.5
* Keras
    * Version: 2.0.6
    * Theano: YES (v0.9)
    * Tensorflow: YES (1.2.1)
