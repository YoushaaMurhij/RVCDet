
## Dataset
### Download [WAYMO dataset](https://waymo.com/open/) and organize it as follows:
```bash
└── WAYMO_DATASET_PATH 
       ├── tfrecord_training       
       ├── tfrecord_validation   
       ├── tfrecord_testing 
```
Remember to change the path in [start.sh] to the WAYMO_DATASET_PATH path above.

```bash
export PYTHONPATH="${PYTHONPATH}:/home/trainer/rvcdet/RVCDet"
export PYTHONPATH="${PYTHONPATH}:/home/trainer/rvcdet/nuscenes-devkit/python-sdk"
```

## Getting Started
### Prepare train set 
```bash
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py \
        --record_path '/home/trainer/rvcdet/RVCDet/data/Waymo/training/*.tfrecord' \  
        --root_path '/home/trainer/rvcdet/RVCDet/data/Waymo/train/'
```
### Prepare validation set 
```bash
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py \
        --record_path '/home/trainer/rvcdet/RVCDet/data/Waymo/tfrecord_validation/*.tfrecord' \  
        --root_path '/home/trainer/rvcdet/RVCDet/data/Waymo/val/'
```
### Prepare testing set 
```bash
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py \
        --record_path '/home/trainer/rvcdet/RVCDet/data/Waymo/tfrecord_testing/*.tfrecord' \ 
        --root_path '/home/trainer/rvcdet/RVCDet/data/Waymo/test/'
```
## Create info files
### Three Sweep Infos 
```bash
# for train split
python tools/create_data.py waymo_data_prep \
        --root_path=data/Waymo \
        --split train \
        --nsweeps=3  
```
```bash
# for val split
python tools/create_data.py waymo_data_prep \
        --root_path=data/Waymo \
        --split val \
        --nsweeps=3 
```
```bash
# for test split
python tools/create_data.py waymo_data_prep \
        --root_path=data/Waymo \
        --split test \
        --nsweeps=3 
```

## Training on WAYMO dataset:
For distributed training use:
```bash
python -m torch.distributed.launch --nproc_per_node=2 ./tools/train.py \
/home/josh/workspace/RVCDet/configs/waymo/pp/waymo_rvdet_pp_two_pfn_stride1_3x.py \
        --work_dir waymo_exp/rvdet-pp-dyn
```
For single device training use:
```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
/home/josh/workspace/RVCDet/configs/waymo/pp/waymo_rvdet_pp_two_pfn_stride1_3x.py \
        --work_dir waymo_exp/rvdet-pp-dyn
```
  
## Validation on WAYMO dataset:
```bash
python tools/dist_test.py \
/home/josh/workspace/RVCDet/configs/waymo/pp/waymo_rvdet_pp_two_pfn_stride1_3x.py \
        --work_dir waymo_exp/rvdet-pp-dyn \
        --checkpoint waymo_exp/rvdet-pp-dyn/latset.pth  \
        --speed_test \
        --gpus 1
```

## Testing on WAYMO dataset:
```bash
python tools/dist_test.py \
/home/josh/workspace/RVCDet/configs/waymo/pp/waymo_rvdet_pp_two_pfn_stride1_3x.py \
        --work_dir waymo_exp/rvdet-pp-dyn \
        --checkpoint waymo_exp/rvdet-pp-dyn/latset.pth  \
        --speed_test \
        --testset --gpus 1
```

## Turn on classification module on WAYMO dataset :
To turn on classification module add classification flag.

Example for validation on WAYMO:
```bash
python tools/dist_test.py \
/home/josh/workspace/RVCDet/configs/waymo/pp/waymo_rvdet_pp_two_pfn_stride1_3x.py \
        --work_dir waymo_exp/rvdet-pp-dyn \
        --checkpoint waymo_exp/rvdet-pp-dyn/latset.pth  \
        --speed_test \
        --gpus 1 --classification
```
Example for testing on WAYMO: 
```bash
python tools/dist_test.py \
/home/josh/workspace/RVCDet/configs/waymo/pp/waymo_rvdet_pp_two_pfn_stride1_3x.py \
        --work_dir waymo_exp/rvdet-pp-dyn \
        --checkpoint waymo_exp/rvdet-pp-dyn/latset.pth  \
        --speed_test \
        --testset --gpus 1 --classification
```
