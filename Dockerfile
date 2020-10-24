# Get nvidia/cuda tag to run FROM command
ARG CUDA_TAG
FROM nvidia/cuda:${CUDA_TAG}

# Install dependencies as root for python3.6 and others
RUN apt update && apt install sudo && sudo apt update && \
    sudo apt install -y software-properties-common curl && \
    sudo add-apt-repository ppa:deadsnakes/ppa && \
    sudo apt update && sudo apt -y upgrade && \
    sudo apt install -y python3.6 && \
    curl https://bootstrap.pypa.io/get-pip.py | sudo -H python3.6

# Add user based on host definitions
ARG USER_NAME
ARG USER_ID
ARG GROUP_NAME
ARG GROUP_ID
RUN sudo groupadd -g ${GROUP_ID} ${GROUP_NAME} && \
    sudo useradd -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME}