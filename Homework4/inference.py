import torch
from model import RecNet, GaussianFourierFeatureTransform
from skimage.measure import marching_cubes
import trimesh
import os
import numpy as np
from tqdm import tqdm
import time

def args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the results output directory')
    parser.add_argument('--grid_size', type=int, default=512, help='Size of the grid for SDF computation')
    parser.add_argument('--level', type=float, default=0.005, help='Iso-surface level for marching cubes')
    
    parser.add_argument('--use_fourier', action='store_true', help='Use Fourier features in the model')
    parser.add_argument('--fourier_mapping_size', type=int, default=128, help='Mapping size for Fourier features')
    parser.add_argument('--fourier_scale', type=float, default=10.0, help='Scale for Fourier features')

    parser.add_argument('--grid_range', type=float, default=0.5, help='Range of the grid for SDF computation')
    parser.add_argument('--clean_mesh', action='store_true', help='Whether to clean the mesh by removing disconnected components')

    return parser.parse_args()

def main():
    args = args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the model from checkpoint
    model = torch.load(args.checkpoint_path, weights_only=False)
    model.to(device)
    model.eval()

    print(model.fourier_transform == None)

    # compute the SDF for a grid of points 
    grid_size = args.grid_size
    grid_points = torch.linspace(-1.0 * args.grid_range, args.grid_range, grid_size)
    grid_x, grid_y, grid_z = torch.meshgrid(grid_points, grid_points, grid_points, indexing='ij')
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)
    grid_points = grid_points.to(device)

    batch_size =  args.grid_size ** 2
    sdf_values = []

    start = time.time()
    print(f"Computing SDF for {grid_points.size(0)} points...")
    with torch.no_grad():
        for i in tqdm(range(0, grid_points.size(0), batch_size)):
            batch_points = grid_points[i:i + batch_size]
            sdf_batch = model(batch_points)
            sdf_values.append(sdf_batch)
    sdf_values = torch.cat(sdf_values, dim=0)
    sdf_values = sdf_values.view(grid_size, grid_size, grid_size).cpu().numpy()
    end = time.time()
    print(f"SDF computation completed in {end - start:.2f} seconds.")


    print(f"SDF statistics: min={np.min(sdf_values)}, max={np.max(sdf_values)}, mean={np.mean(sdf_values)}")
    verts, faces, _, _ = marching_cubes(sdf_values, level=args.level)
    

    print(f"vertices: {verts.shape[0]}, faces: {faces.shape[0]}")
    delta_grid = 2 * args.grid_range / (grid_size - 1)  
    verts = verts * delta_grid + np.array([-1.0 * args.grid_range, -1.0 * args.grid_range, -1.0 * args.grid_range])  

    saved_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    if args.clean_mesh:     # Remove disconnected components
        print("Removing disconnected components...")
        components = saved_mesh.split(only_watertight=False)
        if components:
            # Find the largest component by number of faces
            largest_component = max(components, key=lambda component: len(component.faces))
            saved_mesh = largest_component
            print(f"Mesh after removing components: vertices={saved_mesh.vertices.shape[0]}, faces={saved_mesh.faces.shape[0]}")
        else:
            print("No components found or mesh is empty after split.")

    filename = args.checkpoint_path.split('/')[1] 
    tag = args.checkpoint_path.split('/')[-1].split('.')[0]
    output_dir = os.path.join(args.output_path, filename, tag)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'reconstructed_mesh_{args.level}.obj')
    saved_mesh.export(output_file)
    print(f"Mesh saved to {output_file}")

if __name__ == "__main__":
    main()



