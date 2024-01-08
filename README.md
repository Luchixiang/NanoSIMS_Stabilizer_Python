# NanoSIMS Stabilizer Python version
This repository contains the python source code for our NanoSIMS stabilizer plugin which allows the batch processing and GPU acceleration

If you prefer ImageJ plugin which is much easier for use, please go to https://github.com/Luchixiang/Nanosims_stabilize for more details. 

## Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1.
```Shell
conda create --name stabilizer
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```

## Installation
```Shell
git clone https://github.com/Luchixiang/NanoSIMS_Stabilizer_Python
```

## Stabilize Single NanoSIMS file

```Shell
cd core
python stabilize.py --file file_path --save_file _save_file_path --channel -
```
Please replace the ``--file`` with the NanoSIMS file you want to stabilize (now only support .nrrd file) and `--save_file ` with the path you want to store the stabilized file.

Also please indicate the signal channel which used to calculate the transformation map and apply it to other channels. Strong signal channel are recommended such as - (SE) or 32S.

## Batch Stabilize NanoSIMS files
```Shell
cd core
python stabilize_batch.py --path file_path --save_path _save_file_path --channel 32S
```
Please replace the ``--path`` with the folder stored NanoSIMS files you want to stabilize and `--save_path ` with the folder you want to store the stabilized files.