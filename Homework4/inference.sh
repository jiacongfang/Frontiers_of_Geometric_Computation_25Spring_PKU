#!/bin/bash
python inference.py --output_path result_fourier \
    --checkpoint_path path_to_checkpoint  \
    --grid_size 512 \
    --level level_of_marching_cube   \
    --clean_mesh    # postprocessing option to clean the artifacts in the space