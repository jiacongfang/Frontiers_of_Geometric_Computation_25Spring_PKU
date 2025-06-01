# Homework 4 of "Frontiers of Geometric Computing"

Refer to the implementation of [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739).
You can find the official implementation at [tancik/fourier-feature-networks](https://github.com/tancik/fourier-feature-networks?tab=readme-ov-file) and 
the PyTorch version at [ndahlquist/pytorch-fourier-feature-networks](https://github.com/ndahlquist/pytorch-fourier-feature-networks).

**For for details of the project, please refer to the report pdf in `./doc`**

## Dependency

The code is tested with `python 3.12.9`, `torch 2.6.0+cu124`. And all the experiments are run on a single NVIDIA RTX 4090D.
Run the following command to install the dependencies:

```bash
conda create -n hw4 python=3.12
conda activate hw4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install tensorboardX, tqdm, scikit-image, trimesh, numpy
```

Maybe some missing dependencies, please install them as needed if some errors occur.

## Training and Testing

I provide the training and testing scripts in `./train.sh` and `./inference.sh`, respectively.
And you can find the meaning of each argument in the corresponding script, `./train.py/args_parser()` and `./inference.py/args_parser()`.

Here is an example of training and testing script:

```bash
# Training with naive MLP
python train.py --data_root ./data --save_path ./checkpoint --lambda_gradient 0.5 \
    --lambda_eikonal 0  --lambda_sdf 2.0 --mix_dataset  --learning_rate 1e-4 \
    --fourier_mapping_size 64 --fourier_scale 5 --num_iters 60000 --sample_size 20000

# Training with MLP and Fourier Features, take about 3 hours
python train.py --data_root ./data --save_path ./checkpoint_fourier --lambda_gradient 0.5 \
    --lambda_eikonal 0  --lambda_sdf 2.0 --mix_dataset --use_fourier --learning_rate 2e-4 \
    --fourier_mapping_size 64 --fourier_scale 5
```

When some arguments are not specified, the default values in the python script will be used.

```bash
# Inference code, take about 40 seconds to generate a mesh, and it will be faster if you set a smaller `grid_size`.
python inference.py --output_path result_fourier \
    --checkpoint_path path_to_checkpoint  \
    --grid_size 512 \
    --level level_of_marching_cube   \ 
    --clean_mesh    # ONLY FOR FOURIER METHOD: clean the artifacts in the space
```

You can specify the `--checkpoint_path` to the path of the checkpoint you want to test, and the `--level` is the level of the marching cube algorithm, which is set to 0.001 by default. And suggest to adjust `--level` between 0.0 and 0.01 to get a better mesh.

## Results

All my results are saved in the `./result` folder, the file tree is as follows:

```bash
./result
├── GT
│   └──  uid.obj
├── result_naive
│   └── uid.obj
└── result_fourier
    └── uid.obj
```

Where `uid` is the unique identifier of the dataset, and `checkpoint_name` is the name of the checkpoint you used to generate the mesh.
