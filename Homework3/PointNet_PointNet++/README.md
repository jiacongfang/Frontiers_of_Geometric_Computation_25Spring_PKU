# PointNet & PointNet++
> **Claim:** The training and test code are modified from [the pytorch implementation of PointNet and PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Dependency

The code is tested with `python 3.10` and `pytorch 2.7.0+cu126`. The training and testing is run on a single NVIDIA GeForce RTX 4090 D. Run the following command to install the dependencies:

```bash
conda create -n pointnet python=3.10
conda activate pointnet
pip install torch torchvision torchaudio
pip install -r requirements.txt     # install other dependencies
```

**Note: The `requirements.txt` may not cover all the dependencies. If you encounter any errors, please install the missing packages manually.**

## Train and Test

**Dataset Preparation:** Download the alignment **ModelNet** and save in `data/modelnet40_normal_resampled/`

**Training and Testing:**

My checkpoints and logs are in `./log/classification/`. You can run the following command to reproduce the results:

```bash
# Train and Test PonintNet
python train_classification.py --model pointnet_cls --log_dir pointnet_cls
python test_classification.py --log_dir pointnet_cls --model_name pointnet_cls
```

```bash
# Train and Test PointNet++
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg --model_name pointnet2_cls_ssg
```