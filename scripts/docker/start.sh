#!/bin/bash

# Start the container with useful privileges.
#
# Author: Helio Perroni Filho

SCRIPT_PATH=$(readlink -f "$0")
export SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

IMAGE='elderlab/cstrack:latest'
NAME=${IMAGE//[:\/]/_}

export DOCKER_CONTAINER_NAME=$NAME

# Create the PulseAudio socket to enable audio access from the Docker container.
pactl load-module module-native-protocol-unix socket=/tmp/pulseaudio.socket 2> /dev/null

# Disable GPU mode if the NVIDIA monitor application is not found.
GPUS="--gpus all"
if ! which nvidia-smi > /dev/null
then
  GPUS=""
fi

if [ "$(docker ps -aq -f name=$NAME -f status=exited)" ]
then
  echo "Restarting container..."
  docker restart $NAME
elif [ "$(docker ps -aq -f name=$NAME)" == "" ]
then
  echo "Starting container..."
  docker run -id \
      -e DISPLAY \
      --net=host \
      --privileged \
      --cap-add=ALL \
      $GPUS \
      --env PULSE_SERVER=unix:/tmp/pulseaudio.socket \
      --env PULSE_COOKIE=/tmp/pulseaudio.cookie \
      --volume="$HOME:/home/user/host" \
      --volume="/dev:/dev" \
      --volume="/lib/modules:/lib/modules" \
      --volume="/media:/home/user/media" \
      --volume="/var/run/dbus:/var/run/dbus" \
      --volume="/run/user/1000/pulse:/run/user/1000/pulse" \
      --volume="/tmp/pulseaudio.socket:/tmp/pulseaudio.socket" \
      --volume="$SCRIPT_DIR/../../cfg/pulseaudio.client.conf:/etc/pulse/client.conf" \
      --name="$NAME" \
      $IMAGE
fi

echo "Connecting to container..."

screen -R -U -c "$SCRIPT_DIR/../../cfg/screenrc" docker exec -it $NAME /bin/bash

echo "Stopping container..."

docker kill $NAME
