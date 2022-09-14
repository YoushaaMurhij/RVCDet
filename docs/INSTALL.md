
## Installation :hammer_and_wrench::
### After cloning the repository:
```bash
cd path/to/RVCDet # specify the correct path here
cd RVCDet
```
### Go to docker directory:
```bash
cd RVCDet/docker
```
### build RVCDet docker image:
```bash
./build.sh
```
### Start RVCDet docker container:
Open start.sh and specify the correct path to nuScenes/Waymo Dataset and run this command in terminal:
```bash
./start.sh
```
### Enter the container:
```bash
./into.sh
```
### Now INSIDE the running container:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/trainer/rvcdet/RVCDet"
export PYTHONPATH="${PYTHONPATH}:/home/trainer/rvcdet/nuscenes-devkit/python-sdk"
Bash setup.bash
```
