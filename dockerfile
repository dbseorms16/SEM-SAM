
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities & python prerequisites
RUN apt-get update -y && apt-get install -y --no-install-recommends\
    wget \
    vim \
    curl \
    ssh \
    tree \
    sudo \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up time zone
ENV TZ=Asia/Seoul
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime


RUN python -m pip install --upgrade pip && \
    pip install numpy && \
    pip install Pillow && \
    pip install opencv-python && \
    pip install pytorch-lightning==1.8.2 && \
    pip install segmentation-models-pytorch && \
    pip install transformers && \
    pip install einops && \
    pip install matplotlib && \
    pip install segment-anything  && \
    pip install pytorch-lightning==2.1.2 && \
    pip install ms-python.python 

RUN echo code --install-extension eamodio.gitlens 
RUN echo code --install-extension formulahendry.terminal