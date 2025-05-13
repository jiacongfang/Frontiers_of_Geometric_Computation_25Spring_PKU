# OCNN
> **Claim:** The training and test code are modified from [the official pytorch implementation of OCNN](https://github.com/octree-nn/ocnn-pytorch)

## Dependency

This code is tested with `python 3.10`, `pytorch 2.6.0+cu124`. And the expriments are run on a single NVIDIA RTX A6000. 
Run the following command to install the dependencies:

```bash
conda create -n ocnn python=3.10
conda activate ocnn
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt     # install other dependencies
```

## Train and Test

My training logs and models are in `./logs/m40/d5_05121714`, and the test logs are in `./logs/m40/d5_05130301`. You can run the following command to reproduce the results:

```bash
# Train
python tools/cls_modelnet.py    # prepare the dataset
python classification.py --config configs/cls_m40.yaml SOLVER.alias time
```

Set the checkpoint path in `configs/cls_m40_test.yaml`, the run the following command to test the model:

```bash
# Test
python classification.py --config configs/cls_m40_test.yaml SOLVER.alias time
```

