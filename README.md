# NanoSIMS Stabilizer Python version
This repository contains the python source code for our NanoSIMS stabilizer which allows batch processing and GPU acceleration (more than 10 time faster compared with ImageJ Plugin)

If you prefer ImageJ plugin which is much easier for use, please go to https://github.com/Luchixiang/Nanosims_stabilize for more details. 


## Installation
```Shell
git clone https://github.com/Luchixiang/NanoSIMS_Stabilizer_Python
cd NanoSIMS_Stabilizer_Python
```
## Requirements
The code has been tested with PyTorch 2.0. 
```Shell
conda create --name stabilizer python=3.9
conda activate stabilizer
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 -c pytorch 
pip install -r requirments.txt
```

## Stabilize Single NanoSIMS file

```Shell
cd core
python stabilize.py --file file_path --save_file _save_file_path --channel -
```
Please replace the ``--file`` with the NanoSIMS file you want to stabilize (now only support .nrrd file) and `--save_file ` with the path you want to store the stabilized file.

Also please indicate the signal channel which used to calculate the transformation map and apply it to other channels. Strong signal channel are recommended such as - (SE) or 32S.

We provide a demo file. To register the demo file:
```shell
cd core
python stabilize.py --file ./demo/ --save_file _save_file_path --channel -
```
## Batch Stabilize NanoSIMS files
```Shell
cd core
python stabilize_batch.py --path file_path --save_path _save_file_path --channel 32S
```
Please replace the ``--path`` with the folder stored NanoSIMS files you want to stabilize and `--save_path ` with the folder you want to store the stabilized files.


This repository use [RAFT](https://github.com/princeton-vl/RAFT) for reference. 