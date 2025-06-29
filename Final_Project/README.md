# Final Project: 3D Latent Diffusion Model for 3D Shape Generation

Following the paper [SDFusion: Multimodal 3D Shape Completion, Reconstruction, and Generation](https://yccyenchicheng.github.io/SDFusion/), the project implements a 3D Latent Diffusion Model for 3D shape generation using a UNet architecture including unconditional and text-conditional generation. 

## Dependency

The code is tested with `python 3.12.9`, `torch 2.6.0+cu124`. And all the experiments are run on a single NVIDIA RTX 4090D.
Run the following command to install the dependencies:

```bash
conda create -n 3d_diffusion python=3.12
conda activate 3d_diffusion
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install diffusers==0.29.2 transformers==4.48.3
pip install tensorboardX, tqdm, scikit-image, trimesh, numpy, matplotlib, h5py, scipy, opencv-python, einops, imageio, joblib, termcolor
```

Maybe some missing dependencies, please install them as needed if some errors occur (just like `pip install <missing_package>`).

## Data Preparation

> The dataset preparation is based on the [SDFusion repository](https://github.com/yccyenchicheng/SDFusion), and the preprocessing scripts are copied from there.

* ShapeNet
    1. Download the ShapeNetV1 dataset from the [official website](https://www.shapenet.org/). Then, extract the downloaded file and put the extracted folder in the `./data` folder. Here we assume the extracted folder is at `./data/ShapeNet/ShapeNetCore.v1`.
    2. Run the following command for preprocessing the SDF from mesh.

    ```bash
    mkdir -p data/ShapeNet && cd data/ShapeNet
    wget [url for downloading ShapeNetV1]
    unzip ShapeNetCore.v1.zip
    cd ../../
    ./unzip.sh      # unzip required category files
    cd preprocess
    ./launchers/launch_create_sdf_shapenet.sh
    ```

* text2shape
    1. Run the following command for setting up the text2shape dataset.

    ```bash
    mkdir -p data/ShapeNet/text2shape
    wget http://text2shape.stanford.edu/dataset/captions.tablechair.csv -P data/ShapeNet/text2shape
    cd preprocess
    ./launchers/create_snet-text_splits.sh
    ```

After the above steps, you can run the following command to check if the dataset is prepared correctly, **Modify the path in `dataset.py` first.**

```bash
# check the dataset and data loader
python datasets.py
```


## Training and Testing

All model weights and tensorboard files are saved in the `logs_weight` folder. You can also download from the [PKU disk link](https://disk.pku.edu.cn/link/AA3B115F519B354D259577EF3575C34B2A) (about 3.65G, it's recommended to download with PKU VPN), then put them in the `logs_weight` folder and **modify the path in the test/train scripts respectively.**

### Testing

Run the following command to test the model:

```bash
# VQ-VAE reconstruction test
python test_vqvae.py --checkpoint_path /path_to_vqvae_checkpoint --output_dir ./results_vqvae/ 

# unconditional generation
python test_unet.py --unet_checkpoint_path /path_to_checkpoints \
    --vqvae_path /path_to_vqvae_checkpoint --output_dir ./results/ \
    --num_samples 20 --max_inference_steps 5000 \  
    --sdf_threshold 0.005 # threshold for marcihng cubes.

# max_inference_steps is the number of denoising steps.

# text-conditional generation
HF_ENDPOINT=https://hf-mirror.com python test_unet_attention.py  \
    --unet_checkpoint_path /path_to_checkpoints \
    --vqvae_path /path_to_vqvae_checkpoint --output_dir ./results/ \
    --text "text prompt" --guidance_scale 7.5  \
    --max_inference_steps 3000  --num_samples 20 \
    --sdf_threshold 0.005 # threshold for marcihng cubes.
```

All the generated meshes will be saved in the `./results/` folder. You can visualize them with any 3D viewer, such as [MeshLab](https://www.meshlab.net/) or [Blender](https://www.blender.org/).

### Training

Run the following command to train the model:

```bash
# VQ-VAE training
python train_vqvae.py --batch_size 8 --num_epochs 50

# unconditional UNet training
python train_unet.py --vqvae_path /path_to_vqvae  --num_epochs 200 \
    --category cat # Category of shapes to train on (options: chair, speaker, rifle, sofa, table)

# text-conditional UNet training
HF_ENDPOINT=https://hf-mirror.com python train_unet_attention.py \
    --vqvae_path /path_to_vqvae_checkpoint --log_dir ./logs_unet_attention/ \
    --train_timesteps 3000 --num_epochs 30 --model_size medium \
    --batch_size 40
```

## Results and Reports

All results are placed in the `./results/` folder, stored in the form of `.obj` files. And you can find the report pdf and tex source code in the `./doc/` folder.

## Acknowledgements

The project refers to several wonderful open-source projects, including [SDFusion](https://github.com/yccyenchicheng/SDFusion), [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet), [diffusers](https://github.com/huggingface/diffusers). Special thanks to the authors of these projects for their contributions to the community.
