import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from tqdm import tqdm
import trimesh
from skimage import measure

from vqvae.network import VQVAE
from diffusion_unet.model import DiffusionUNet3D, DiffusionModel
from diffusers import DDPMScheduler


def load_models(vqvae_path, unet_checkpoint_path, device):
    vqvae_model_config = {
        "embed_dim": 3,
        "n_embed": 8192,
        "ddconfig": {
            "double_z": False,
            "z_channels": 3,
            "resolution": 64,
            "in_channels": 1,
            "out_ch": 1,
            "ch": 64,
            "ch_mult": [1, 2, 4],  
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "dropout": 0.0
        }
    }
    
    vqvae = VQVAE(
        ddconfig=vqvae_model_config["ddconfig"],
        n_embed=vqvae_model_config["n_embed"],
        embed_dim=vqvae_model_config["embed_dim"],
        remap=None,
        sane_index_shape=False
    )
    
    vqvae_state_dict = torch.load(vqvae_path, map_location=device)
    vqvae.load_state_dict(vqvae_state_dict)
    vqvae = vqvae.to(device)
    vqvae.eval()
    print(f"[*] VQVAE loaded from {vqvae_path}")
    
    unet_checkpoint = torch.load(unet_checkpoint_path, map_location=device)
    
    if 'model_config' in unet_checkpoint:
        model_config = unet_checkpoint['model_config']
    else:       # medium model config as default
        model_config = {
            'in_channels': 3,
            'out_channels': 3,
            'time_emb_dim': 256,
            'f_maps': 128,
            'num_levels': 3
        }
    
    unet = DiffusionUNet3D(**model_config)
    unet.load_state_dict(unet_checkpoint['unet_state_dict'])
    unet = unet.to(device)
    unet.eval()
    print(f"[*] UNet loaded from {unet_checkpoint_path}")
    print(f"[*] Model config: {model_config}")
    
    return vqvae, unet


def sample_with_ddpm_scheduler(unet, vqvae, latents, num_inference_steps=1000, num_train_timesteps = 5000, device='cuda'):    
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon"
    )
    
    scheduler.set_timesteps(num_inference_steps)
    
    print(f"[*] Starting sampling with {num_inference_steps} steps...")
    print(f"[*] Noise shape: {latents.shape}")
    
    with torch.no_grad():
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
            timestep = t.expand(latents.shape[0]).to(device)
            noise_pred = unet(latents, timestep)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
    print("[*] Sampling completed!")
    print(f"[*] Final latents shape: {latents.shape}, range: [{latents.min():.3f}, {latents.max():.3f}]")
    
    print("[*] Decoding latents to SDF...")
    decoded_sdf = vqvae.decode_no_quant(latents)
    print(f"[*] Decoded SDF shape: {decoded_sdf.shape}, range: [{decoded_sdf.min():.3f}, {decoded_sdf.max():.3f}]")
    
    return decoded_sdf.cpu(), latents.cpu()


def sdf_to_mesh(sdf_volume, threshold=0.0, spacing=(0.1, 0.1, 0.1)):
    print(f"[*] SDF volume shape: {sdf_volume.shape}")
    print(f"[*] SDF range: [{sdf_volume.min():.3f}, {sdf_volume.max():.3f}]")
    
    try:
        verts, faces, normals, values = measure.marching_cubes(
            sdf_volume, level=threshold, spacing=spacing
        )
        
        print(f"[*] Extracted mesh: {len(verts)} vertices, {len(faces)} faces")
        return verts, faces, normals
        
    except Exception as e:
        print(f"[#] Error in marching cubes: {e}")
        return None, None, None


def save_mesh_as_obj(vertices, faces, output_path):
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(output_path)
        print(f"[*] Mesh saved to {output_path}")
        return True
    except Exception as e:
        print(f"[#] Error saving mesh: {e}")
        return False


@torch.no_grad()
def test_diffusion_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    vqvae, unet = load_models(args.vqvae_path, args.unet_checkpoint_path, device)
    
    print(f"\n[*] Generating {args.num_samples} samples...")

    latent_shape = (args.num_samples, 3, 16, 16, 16)
    initial_latents = torch.randn(latent_shape, device=device)  

    if args.num_inference_step > 0:
        infer_steps = [ i * args.num_inference_step for i in range(1, args.max_inference_steps // args.num_inference_step + 1) ]
    else:
        infer_steps = [args.max_inference_steps]
    print(f"[*] Using inference steps: {infer_steps}")

    for infer_step in infer_steps:
        latents = initial_latents.clone()  
        
        generated_sdf, generated_latents = sample_with_ddpm_scheduler(
            unet, vqvae, 
            latents,
            num_inference_steps=infer_step,
            num_train_timesteps=args.num_train_timesteps,
            device=device
        )
        
        for i in range(args.num_samples):
            print(f"\n[*] Processing sample {i+1}/{args.num_samples} with {infer_step} inference steps...")
            
            sdf_sample = generated_sdf[i, 0].numpy()  # Shape: (64, 64, 64)
                    
            print(f"[*] Generating mesh for sample {i}")
            vertices, faces, normals = sdf_to_mesh(sdf_sample, threshold=args.sdf_threshold)
            
            if vertices is not None and len(vertices) > 0:
                obj_path = os.path.join(args.output_dir, f'generated_shape_{i}_step_{infer_step}.obj')
                success = save_mesh_as_obj(vertices, faces, obj_path)
                
                if success:
                    print(f"[*] Mesh {i} saved successfully: {obj_path}")
                else:
                    print(f"[#] Failed to save mesh {i}")
            else:
                print(f"[#] Failed to generate mesh for sample {i} - no valid surface found")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Diffusion UNet Pipeline')
    parser.add_argument('--vqvae_path', type=str, 
                        default='logs_vqvae/20250622-154347/vqvae_final.pth',
                        help='Path to the pretrained VQVAE model')
    parser.add_argument('--unet_checkpoint_path', type=str, 
                        default='logs_unet/20250625-223823/diffusion_unet_epoch_100.pth',
                        help='Path to the trained UNet checkpoint')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save generated results')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--max_inference_steps', type=int, default=5000,
                        help='Number of denoising steps')
    parser.add_argument('--sdf_threshold', type=float, default=0.001,
                        help='SDF threshold for marching cubes')
    parser.add_argument('--num_train_timesteps', type=int, default=5000,
                        help='Number of training timesteps for the scheduler')
    parser.add_argument('--num_inference_step', type=int, default=0)
    
    args = parser.parse_args()
    
    print("=== Diffusion Model Testing ===")
    print(f"[*] VQVAE path: {args.vqvae_path}")
    print(f"[*] UNet checkpoint: {args.unet_checkpoint_path}")
    print(f"[*] Output directory: {args.output_dir}")
    print(f"[*] Number of samples: {args.num_samples}")
    
    test_diffusion_pipeline(args)
