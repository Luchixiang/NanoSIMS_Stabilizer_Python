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
We currently only support .nrrd file. If you only have .im file, you can batch convert the .im to .nrrd with OpenMIMS plugin. 
```Shell
cd core
python stabilizev2.py --file nanosims_file.nrrd --save_file nanosims_file_registered.nrrd --channel -
```
Please replace the ``--file`` with the NanoSIMS file you want to stabilize and `--save_file ` with the path you want to store the stabilized file.

Also please indicate the signal channel which used to calculate the transformation map and apply it to other channels. Strong signal channel are recommended such as - (denotes SE) or 32S.

We provide a demo file. To register the demo file:
```shell
cd core
python stabilizev2.py --file ./demo/demo_data.nrrd --save_file ./demo/demo_data_registered.nrrd --channel -
```
## Batch Stabilize NanoSIMS files
```Shell
cd core
python stabilizev2_batch.py --path file_path --save_path _save_file_path --channel 32S
```
Please replace the ``--path`` with the folder stored NanoSIMS files you want to stabilize and `--save_path ` with the folder you want to store the stabilized files.

We also release the model weight and pytorch jit script, which allows the deployment of the model without the Python environment. 

This repository use [RAFT](https://github.com/princeton-vl/RAFT) for reference. 