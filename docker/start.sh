#!/bin/bash

cd "$(dirname "$0")"
cd ..

workspace_dir=$PWD

if [ "$(docker ps -aq -f status=exited -f name=rvcdet)" ]; then
    docker rm rvcdet;
fi

docker run -it -d --rm \
    --gpus all \
    --net host \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -e "DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    --shm-size="45g" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --name rvcdet \
    -v $workspace_dir/:/home/trainer/rvcdet/:rw \
    -v /path/to/data/on/host/:/home/trainer/rvcdet/RVCDet/data/Waymo/:rw \
    x64/rvcdet:latest 

docker exec -it rvcdet \
    /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/home/trainer/rvcdet/RVCDet\";
    export PYTHONPATH=\"${PYTHONPATH}:/home/trainer/rvcdet/nuscenes-devkit/python-sdk\";
    cd /home/trainer/rvcdet/RVCDet;
    bash setup.sh;"

