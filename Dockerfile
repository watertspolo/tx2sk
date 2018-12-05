FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04 

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NOWARNINGS yes

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl git pkg-config apt-utils\
                                               python3 python3-dev python3-pip python3-setuptools cmake emacs
RUN apt-get install bsdmainutils # build-essential install llvm

RUN pip3 install --upgrade pip
# RUN pip3 llvmpy cython numba
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision numpy pandas matplotlib Pillow h5py scipy tensorflow-gpu==1.4.1 jupyter scikit-image
RUN pip3 install opencv-python scikit-learn tqdm

WORKDIR /hdgan

EXPOSE 8888
COPY . /hdgan

