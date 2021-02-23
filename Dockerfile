FROM tensorflow/tensorflow:2.4.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

# (Optional) Disable certificate verification
RUN touch /etc/apt/apt.conf.d/99verify-peer.conf && \
    echo >>/etc/apt/apt.conf.d/99verify-peer.conf "Acquire { https::Verify-Peer false }"

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

# Setup DISPLAY
ENV DISPLAY :1
# VOLUME /tmp/.X11-unix
RUN wget -O /etc/X11/xorg.conf http://xpra.org/xorg.conf && \
    cp /etc/X11/xorg.conf /usr/share/X11/xorg.conf.d/xorg.conf

# Setup SUMO
ENV SUMO_HOME /usr/share/sumo

# Setup user
# ARG USER_ID
# ARG GROUP_ID
# RUN addgroup --gid $GROUP_ID user && \
#     adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
# USER user

# Install requirements.txt
ENV PATH /root/.local/bin:$PATH
RUN python3.7 -m pip install --user --no-cache --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN python3.7 -m pip install --user --no-cache -r /tmp/requirements.txt

# Copy source files
COPY . /src

# Setup workdir
WORKDIR /src

# Entrypoint
SHELL ["/bin/bash", "-c", "-l"]
