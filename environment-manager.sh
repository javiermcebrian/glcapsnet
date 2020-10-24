#!/bin/bash

# Constants
image_name=javiermcebrian/glcapsnet
data_folder=TFM
src_folder=tfm_visual_attention

# Args
service=$1
action=$2

# Argument handling
if [ -z "$1" ] && [ -z "$2" ]
  then
    printf "
Usage: ./environment-manager.sh service action\n
Service must be one of the services defined in docker-compose.yml.\n
Action must be one of the following: build, play, stop, clean.\n
"
    exit 1
fi

# Parameters
cuda_tag=$(jq -r --arg service "${service}" '.[$service].cuda_tag' docker-config.json)
dockerfile_path=$(jq -r --arg service "${service}" '.[$service].dockerfile_path' docker-config.json)
image_base="${image_name}:${cuda_tag}"

# Functions
case $action in
  build)
    # Build base image
    sudo docker build \
      -t ${image_base} \
      --file Dockerfile \
      --build-arg CUDA_TAG=${cuda_tag} \
      --build-arg USER_NAME=$(whoami) \
      --build-arg USER_ID=$(id -u) \
      --build-arg GROUP_NAME=$(getent group $(id -g) | cut -d: -f1) \
      --build-arg GROUP_ID=$(id -g) \
      .
    # Build specified service
    sudo docker build \
      -t "${image_base}-${service}" \
      --file "${dockerfile_path}/Dockerfile" \
      --build-arg IMAGE_BASE=${image_base} \
      --build-arg HERE=${dockerfile_path} \
      --build-arg USER_NAME=$(whoami) \
      .
    ;;
  play)
    # Run service with nvidia-docker (--gpus all)
    # Only mount data and code folders (avoid docker to install as user in shared .pip)
    # Mount credential files to enable 'sudo pip3.6 install ...'
    sudo docker run -it --gpus all \
      -v "${HOME}/${data_folder}":"${HOME}/${data_folder}" \
      -v "${HOME}/${src_folder}":"${HOME}/${src_folder}" \
      -v /etc/passwd:/etc/passwd:ro \
      -v /etc/group:/etc/group:ro \
      -v /etc/shadow:/etc/shadow:ro \
      "${image_base}-${service}"
    ;;
  stop)
    sudo docker ps -a | awk '{ print $1,$2 }' | grep "${image_base}-${service}" | awk '{ print $1 }' | xargs -I {} sudo docker stop {}
    ;;
  clean)
    sudo docker ps -a | awk '{ print $1,$2 }' | grep "${image_base}-${service}" | awk '{ print $1 }' | xargs -I {} sudo docker rm {}
    ;;
  *)
    printf "\nUnknown action argument.\n\n"
    ;;
esac