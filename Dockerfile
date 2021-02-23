FROM tensorflow/tensorflow:2.4.0-gpu

# (Optional) Install certificates
RUN apt-get install -y \ 
        ca-certificates \
        apt-transport-https \
    && rm -rf /var/lib/apt/lists/*
COPY ./certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates

# Install dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y \
        libsm6 \
        libspatialindex-dev \
        libxext6 \
        libxrender-dev \
        python3.7 \
        python3.7-dev \
        python3.7-venv \
        sudo \
        sumo \
        sumo-doc \
        sumo-tools \
        wget \
        x11-apps \
        xserver-xorg-video-dummy && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Update default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python get-pip.py && \
    pip install --upgrade pip

# Setup DISPLAY
ENV DISPLAY :1
# VOLUME /tmp/.X11-unix
RUN wget -O /etc/X11/xorg.conf http://xpra.org/xorg.conf && \
    cp /etc/X11/xorg.conf /usr/share/X11/xorg.conf.d/xorg.conf

# Setup SUMO
ENV SUMO_HOME /usr/share/sumo

# Setup user
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

# Setup workdir
WORKDIR /src

# Copy source files and install dependencies
COPY . /src
# RUN pip install -r ${workdir}/requirements.txt

# Entrypoint
SHELL ["/bin/bash", "-c", "-l"]
