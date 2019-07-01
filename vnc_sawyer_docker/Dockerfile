FROM arclab/vnc:cuda9.0-cudnn7
WORKDIR /.

# FOR Mujoco
RUN apt-get update -q \
  && apt-get dist-upgrade -y \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  curl \
  git \
  libgl1-mesa-dev \
  libgl1-mesa-glx \
  libglew-dev \
  libosmesa6-dev \
  software-properties-common \
  net-tools \
  unzip \
  vim \
  virtualenv \
  wget \
  xpra \
  xserver-xorg-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# gym
RUN apt-get update && apt-get dist-upgrade -y && apt-get install -y \
  ffmpeg \
  libboost-all-dev \
  libjpeg-dev \
  libsdl2-dev \
  patchelf \
  python-pip \
  python-pyglet \
  python-setuptools \
  python3-dev \
  python3-opengl \
  unzip \
  xvfb \
  && rm -rf /var/lib/apt/lists/* 

# ========== Special Deps ==========
RUN apt-get update && apt-get -y install \
  make \
  cmake \
  python3-pip

RUN pip3 install awscli
# ALE requires zlib
RUN apt-get -y install zlib1g-dev


RUN pip install imageio tabulate nose
RUN apt-get install -y ack-grep

# usual pip install pygame will fail
RUN pip3 install pygame
RUN pip3 install Pillow

#Mujoco
RUN mkdir /root/.mujoco && \
  cd /root/.mujoco  && \
  wget https://www.roboti.us/download/mjpro150_linux.zip  && \
  unzip mjpro150_linux.zip

RUN touch /root/.mujoco/mjkey.txt

# Set the mujoco path
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin 

# Need this for opengl to work inside docker
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
  ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
  ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && apt-get install mesa-utils

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGLEW.so

ENV HOME=/root
RUN pip3 install robosuite
RUN pip3 install tensorflow==1.12.0
RUN pip3 install tensorflow-gpu==1.12.0
RUN pip3 install pybullet==1.9.5

RUN apt-get install -y python3-tk
WORKDIR /root/
RUN git clone https://github.com/openai/baselines.git && \
  cd baselines && \
  pip3 install -e .

RUN pip3 install gym
RUN pip3 install --upgrade pip
RUN pip3 install atari-py==0.1.14 
RUN pip3 install --no-dependencies stable-baselines 
RUN pip3 install pandas dill seaborn mpi4py zmq glob2 matplotlib pytz setuptools future tqdm
#RUN git clone https://github.com/hill-a/stable-baselines.git && \
#  cd stable-baselines && \
#  pip3 install -e .
CMD ["bash"]


