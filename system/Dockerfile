# Build from base image
ARG IMAGE_BASE
FROM ${IMAGE_BASE}

# OpenCV + TF requirements
RUN sudo apt install -y libsm6 libxext6 libxrender-dev libgomp1

# Install requirements as root
ARG HERE
COPY ${HERE}/requirements.txt /tmp/requirements.txt
RUN sudo pip3.6 install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Set user and workspace
ARG USER_NAME
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

# Entry command
CMD ["bash"]