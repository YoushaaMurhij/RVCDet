FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y python3-pip apt-transport-https ca-certificates gnupg wget git ninja-build libboost-dev build-essential
RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1 \
    libgomp1 \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Open3D from the pypi repositories
RUN python3 -m pip install --no-cache-dir --upgrade open3d

RUN pip3 install \
	torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
	-f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip install \
    torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# Install CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - > /etc/apt/trusted.gpg.d/kitware.gpg
# RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt install -y cmake

# SpConv installation
RUN git clone -b v1.2.1 https://github.com/traveller59/spconv.git --recursive
RUN cd spconv  \
	# && git checkout 7342772  \
	&& SPCONV_FORCE_BUILD_CUDA=1 python3 setup.py bdist_wheel  \
	&& cd ./dist  \
	&& pip install *  \
	&& cd ../..


RUN apt install -y lsb-release wget software-properties-common
# Install LLVM 10
WORKDIR /code
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 10

# OpenPCDet dependencies fail to install unless LLVM 10 exists on the system
RUN ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config

RUN pip3 install --upgrade pip

ARG TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0+PTX"


# Install rvcdet
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 uninstall opencv-python  --yes
RUN pip3 install opencv-python-headless 

# # Apex installation
WORKDIR /tmp/unique_for_apex
RUN git clone https://github.com/NVIDIA/apex.git  \
 	&& cd apex \
 	&& git checkout 5633f6 \
	#&& python3 setup.py install
	&& pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


