## Install 
1. Create virtual environment (conda): `conda create -n grraspn python=3.7 anaconda`
1. Activate virtual environment (conda): `source activate grraspn`
1. Install Pytorch & CUDA toolkit: `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
1. Intall pycocotools: `pip install pycocotools`
1. Install opencv: `pip install opencv-python`
1. Export CUDA path: `export CUDA_HOME="/usr/local/cuda-10.1"`
1. Clone source: `git clone <link to this repo>`
1. Compile and install project: `cd grasp_rrpn/ && pip install -e .`

## Find out usable GPU's
CUDA uses different numbering gpu's than those of `nvidia-smi`.
To find out the correct gpu numbers to set in **CUDA_VISIBLE_DEVICES** compile:
`nvcc tools/cuda_gpus.cc -o tools/cuda_gpus` and run with `./tools/cuda_gpus`.

Then, correlate Bus/DeviceID's to those of `nvidia-smi`.

## Train
Run training on TRAIN dataset specified in given config:

`CUDA_VISIBLE_DEVICES="2,3" python tools/train_net.py --config-file configs/grasp_rcnn_R_50_FPN_3x.yaml --num-gpus 2`

To resume training from last model checkpoint, add `--resume` flag.0

## Validate
Run validation on TEST dataset specified in given config and use last model checkpoint:

`CUDA_VISIBLE_DEVICES="2,3" python tools/train_net.py --config-file configs/grasp_rcnn_R_50_FPN_3x.yaml --num-gpus 2 --eval --resume`


## Test
Save image with predicted grasps to given output directory:

`CUDA_VISIBLE_DEVICES="2,3" python demo/demo.py --config-file configs/grasp_rcnn_R_50_FPN_3x.yaml --input <INPUT_IMAGE> --output <OUTDIR> --opts MODEL.WEIGHTS <MODELPATH>`

## Inference
Save grasp predictions to text file Jacquard data format:

`CUDA_VISIBLE_DEVICES="2,3" python demo/inference.py <INPUT_IMAGE_DIR> <OUTPUT_FILE>`

## Simulation

For real-time simulation environment for automated grasp testing check out: [gRRasPN simulator](https://github.com/dnandha/grraspn_simulator)  
