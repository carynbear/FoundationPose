docker rm -f foundationpose
DIR=$(pwd)/
xhost +  
docker run \
    --name foundationpose \
    --privileged \
    --gpus all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $DIR:$DIR \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp:/tmp \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/bus/usb:/dev/bus/usb \
    --network=host \
    --ipc=host \
    -e DISPLAY=${DISPLAY} \
    -e GIT_INDEX_FILE \
    foundationpose:live bash -c "cd $DIR && bash"