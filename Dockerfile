FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

MAINTAINER Manfred Diaz <takeitallsource@gmail.com>

ARG TENSORFLOW_VERSION=1.2.1
ARG KERAS_VERSION=2.0.6
ARG PYTORCH_VERSION=0.1.12

RUN echo "Building Deep Learning working Environment..."

RUN echo "Updating & Upgrading sources..." && \
    apt-get -y update && \
    apt-get -y upgrade && \
    echo "Done updating & upgrading"

# Install tools
RUN echo "Installing tools..." && \
    apt-get update && apt-get install -y \
        cmake \
		curl \
		git \
        nano \
		unzip \
		vim \
		wget \
        tar \
        ffmpeg \
        && \
    echo "Tools installed."

# Install languages dependencies
RUN echo "Installing languages dependencies..." && \
    apt-get install -y \
        g++ \
		gfortran \
        python \
        python-dev \
        python-pip \
        python3 \
        python3-dev \
        python3-pip \
        && \
    pip3 install --upgrade pip && \
    echo "alias python='python3'" >> ~/.bash_aliases && \
    echo "alias pip='pip3'" >> ~/.bash_aliases && \
    /bin/bash -c "source ~/.bash_aliases" && \
    echo "Languages dependencies installed."

# Installing commonn python libraries (numpy, scipy, skimage, sklearn, matplotlib, pandas)
RUN echo "Installing common python libraries..." && \
    pip3 install numpy \
		scipy \
		nose \
		scikit-image \
		matplotlib \
		pandas \
		sklearn \
		sympy && \
    echo "Python libraries installed."

# Install TensorFlow
RUN echo "Installing Tensorflow GPU ${TENSORFLOW_VERSION}..." && \
    pip3 install --upgrade \
    https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-${TENSORFLOW_VERSION}-cp35-cp35m-linux_x86_64.whl && \
    echo "Tensorflow GPU installed."

# Install PyTorch
RUN echo "Installing PyTorch ${PYTORCH_VERSION}..." && \
    pip3 install http://download.pytorch.org/whl/cu80/torch-${PYTORCH_VERSION}.post2-cp35-cp35m-linux_x86_64.whl && \
    pip3 install torchvision && \
    echo "PyTorch installed."

# Install Keras (with HDF5 models support)
RUN echo "Installing Keras (with HDF5 support)" && \
    pip3 install keras==${KERAS_VERSION} && \
    apt-get install -y libhdf5-dev && \
    pip3 install h5py && \
    echo "Keras (with HDF5 support) installed."

# Install OpenCV 3.x with CUDA support
RUN echo "Installing OpenCV 3.x (with CUDA support)..." && \
    echo "Installing dependencies..." && \
    apt-get install -y \
        bc \
		build-essential \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libjpeg-dev \
		libpng12-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
        libjasper-dev \
		pkg-config \
		software-properties-common \
		zlib1g-dev \
		qt5-default \
		libvtk6-dev \
		zlib1g-dev \
		libjpeg-dev \
		libwebp-dev \
		libpng-dev \
		libtiff5-dev \
		libjasper-dev \
		libopenexr-dev \
		libgdal-dev \
		libdc1394-22-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libtheora-dev \
		libvorbis-dev \
		libxvidcore-dev \
		libx264-dev \
		yasm \
		libopencore-amrnb-dev \
		libopencore-amrwb-dev \
		libv4l-dev \
		libxine2-dev \
		libtbb-dev \
		libeigen3-dev \
		&& \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3 && \
    echo "Dependencies installed."

RUN echo "Building from OpenCV 3.x sources..." && \
    git clone --depth 1 https://github.com/opencv/opencv.git /root/opencv && \
    git clone --depth 1 https://github.com/opencv/opencv_contrib.git /root/opencv_contrib && \
	cd /root/opencv && \
	mkdir build && \
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DWITH_TBB=ON \
      -DWITH_PTHREADS_PF=ON \
      -DWITH_OPENNI2=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DOPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib/modules \
      -DBUILD_NEW_PYTHON_SUPPORT=ON \
      -DWITH_V4L=ON \
      -DBUILD_TIFF=ON \
      -DENABLE_PRECOMPILED_HEADERS=OFF \
      -DUSE_GStreamer=OFF \
      -DWITH_OPENGL=ON \
      -DWITH_OPENCL=ON \
      -DFORCE_VTK=ON \
      -DWITH_GDAL=ON \
      -DWITH_XINE=ON \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DINSTALL_C_EXAMPLES=OFF \
      -DINSTALL_PYTHON_EXAMPLES=OFF \ 
      -DWITH_IPP=ON \
      -DWITH_CSTRIPES=ON \ 
      -DWITH_EIGEN=ON \
      -DWITH_CUDA=ON \
      -DWITH_CUBLAS=ON \
      -DENABLE_FAST_MATH=1 \
      -DCUDA_FAST_MATH=1 \
      -DWITH_NVCUVID=ON \
      -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" \
      -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
      -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. && \
	make -j"$(nproc)"  && \
	make install && \
	ldconfig && \
	echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc && \
    echo "OpenCV (with CUDA support) built & configured."

# Expose Ports for TensorBoard (6006)
EXPOSE 6006

WORKDIR "/root"
CMD ["/bin/bash"]