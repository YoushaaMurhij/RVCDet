#!/bin/bash
docker exec -it rvcdet \
    /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/home/trainer/rvcdet/RVCDet\";
    export PYTHONPATH=\"${PYTHONPATH}:/home/trainer/rvcdet/nuscenes-devkit/python-sdk\";
    cd /home/trainer/rvcdet/RVCDet;
    /bin/bash"

