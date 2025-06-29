import torch
import os
import argparse
from torch.utils.data import DataLoader
import trimesh
from skimage import measure

from vqvae.network import VQVAE
from datasets import ShapeNetDataset


def load_vqvae_model(checkpoint_path, device):
    model_config = {
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
            "dropout": 0.0,
        },
    }

    vqvae = VQVAE(
        ddconfig=model_config["ddconfig"],
        n_embed=model_config["n_embed"],
        embed_dim=model_config["embed_dim"],
        remap=None,
        sane_index_shape=False,
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    vqvae.load_state_dict(state_dict)
    vqvae = vqvae.to(device)
    vqvae.eval()

    print(f"VQVAE model loaded from {checkpoint_path}")
    return vqvae


def sdf_to_mesh(sdf_volume, threshold=0.0, spacing=(1.0, 1.0, 1.0)):
    print(f"SDF volume shape: {sdf_volume.shape}")
    print(f"SDF range: [{sdf_volume.min():.3f}, {sdf_volume.max():.3f}]")

    try:
        verts, faces, normals, values = measure.marching_cubes(
            sdf_volume, level=threshold, spacing=spacing
        )

        print(f"Extracted mesh: {len(verts)} vertices, {len(faces)} faces")
        return verts, faces, normals

    except Exception as e:
        print(f"Error in marching cubes: {e}")
        return None, None, None


def save_mesh_as_obj(vertices, faces, output_path):
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(output_path)
        print(f"Mesh saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving mesh: {e}")
        return False


def test_vqvae_reconstruction(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    vqvae = load_vqvae_model(args.checkpoint_path, device)

    dataset_config = {
        "info_file": "./dataset_info_files/info-shapenet.json",
        "dataroot": "./data",
        "phase": "test",
        "cat": args.category,
        "res": 64,
        "trunc_thres": 0.2,
    }

    dataset = ShapeNetDataset(
        info_file=dataset_config["info_file"],
        dataroot=dataset_config["dataroot"],
        phase=dataset_config["phase"],
        cat=dataset_config["cat"],
        res=dataset_config["res"],
        trunc_thres=dataset_config["trunc_thres"],
    )

    print(f"Test dataset loaded with {len(dataset)} samples")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    processed_samples = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= args.num_samples:
                break

            print(f"\nProcessing sample {i + 1}/{args.num_samples}")

            original_sdf = data["sdf"].to(device)  # Shape: (1, 1, 64, 64, 64)
            print(f"Original SDF shape: {original_sdf.shape}")
            print(
                f"Original SDF range: [{original_sdf.min():.3f}, {original_sdf.max():.3f}]"
            )

            reconstructed_sdf, vq_loss = vqvae(original_sdf, verbose=False)
            print(f"Reconstructed SDF shape: {reconstructed_sdf.shape}")
            print(
                f"Reconstructed SDF range: [{reconstructed_sdf.min():.3f}, {reconstructed_sdf.max():.3f}]"
            )
            print(f"VQ Loss: {vq_loss.item():.6f}")

            original_np = original_sdf.cpu().numpy()
            reconstructed_np = reconstructed_sdf.cpu().numpy()

            print("Generating mesh from original SDF...")
            orig_sdf_vol = original_np[0, 0]  # Shape: (64, 64, 64)
            orig_verts, orig_faces, orig_normals = sdf_to_mesh(
                orig_sdf_vol, threshold=args.sdf_threshold
            )

            if orig_verts is not None:
                orig_obj_path = os.path.join(
                    args.output_dir, f"original_sample_{i}.obj"
                )
                save_mesh_as_obj(orig_verts, orig_faces, orig_obj_path)

            print("Generating mesh from reconstructed SDF...")
            recon_sdf_vol = reconstructed_np[0, 0]  # Shape: (64, 64, 64)
            recon_verts, recon_faces, recon_normals = sdf_to_mesh(
                recon_sdf_vol, threshold=args.sdf_threshold
            )

            if recon_verts is not None:
                recon_obj_path = os.path.join(
                    args.output_dir, f"reconstructed_sample_{i}.obj"
                )
                save_mesh_as_obj(recon_verts, recon_faces, recon_obj_path)

            processed_samples += 1

    print(f"\nTesting completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VQVAE SDF Reconstruction")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained VQVAE checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_vqvae_results",
        help="Directory to save test results",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Object category to test (chair, table, etc.)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=20, help="Number of samples to test"
    )
    parser.add_argument(
        "--sdf_threshold",
        type=float,
        default=0.005,
        help="SDF threshold for marching cubes",
    )

    args = parser.parse_args()

    print("=== VQVAE Reconstruction Test ===")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Category: {args.category}")
    print(f"Number of samples: {args.num_samples}")
    print(f"SDF threshold: {args.sdf_threshold}")

    test_vqvae_reconstruction(args)
