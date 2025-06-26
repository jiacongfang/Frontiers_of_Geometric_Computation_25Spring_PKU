import torch
import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
import trimesh
from skimage import measure

from vqvae.network import VQVAE
from datasets import ShapeNetDataSet


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
            "dropout": 0.0
        }
    }

    vqvae = VQVAE(
        ddconfig=model_config["ddconfig"],
        n_embed=model_config["n_embed"],
        embed_dim=model_config["embed_dim"],
        remap=None,
        sane_index_shape=False
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
    """保存mesh为OBJ文件"""
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(output_path)
        print(f"Mesh saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving mesh: {e}")
        return False


def visualize_sdf_comparison(original_sdf, reconstructed_sdf, output_dir, sample_idx):
    """可视化原始SDF和重建SDF的对比"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'SDF Comparison - Sample {sample_idx}', fontsize=16)
    
    # 原始SDF的切片
    orig = original_sdf[0, 0]  # Shape: (64, 64, 64)
    recon = reconstructed_sdf[0, 0]  # Shape: (64, 64, 64)
    
    mid_x, mid_y, mid_z = orig.shape[0]//2, orig.shape[1]//2, orig.shape[2]//2
    
    # X方向切片
    axes[0, 0].imshow(orig[mid_x, :, :], cmap='RdBu', vmin=-0.2, vmax=0.2)
    axes[0, 0].set_title('Original - X slice')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(recon[mid_x, :, :], cmap='RdBu', vmin=-0.2, vmax=0.2)
    axes[0, 1].set_title('Reconstructed - X slice')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.abs(orig[mid_x, :, :] - recon[mid_x, :, :]), cmap='Reds')
    axes[0, 2].set_title('Absolute Difference - X slice')
    axes[0, 2].axis('off')
    
    # Y方向切片
    axes[1, 0].imshow(orig[:, mid_y, :], cmap='RdBu', vmin=-0.2, vmax=0.2)
    axes[1, 0].set_title('Original - Y slice')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(recon[:, mid_y, :], cmap='RdBu', vmin=-0.2, vmax=0.2)
    axes[1, 1].set_title('Reconstructed - Y slice')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.abs(orig[:, mid_y, :] - recon[:, mid_y, :]), cmap='Reds')
    axes[1, 2].set_title('Absolute Difference - Y slice')
    axes[1, 2].axis('off')
    
    # Z方向切片
    axes[2, 0].imshow(orig[:, :, mid_z], cmap='RdBu', vmin=-0.2, vmax=0.2)
    axes[2, 0].set_title('Original - Z slice')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(recon[:, :, mid_z], cmap='RdBu', vmin=-0.2, vmax=0.2)
    axes[2, 1].set_title('Reconstructed - Z slice')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(np.abs(orig[:, :, mid_z] - recon[:, :, mid_z]), cmap='Reds')
    axes[2, 2].set_title('Absolute Difference - Z slice')
    axes[2, 2].axis('off')
    
    # 统计信息对比
    mse = np.mean((orig - recon) ** 2)
    mae = np.mean(np.abs(orig - recon))
    
    axes[0, 3].text(0.1, 0.9, 'Original SDF Stats:', fontsize=12, weight='bold')
    axes[0, 3].text(0.1, 0.8, f'Min: {orig.min():.3f}', fontsize=10)
    axes[0, 3].text(0.1, 0.7, f'Max: {orig.max():.3f}', fontsize=10)
    axes[0, 3].text(0.1, 0.6, f'Mean: {orig.mean():.3f}', fontsize=10)
    axes[0, 3].text(0.1, 0.5, f'Std: {orig.std():.3f}', fontsize=10)
    axes[0, 3].axis('off')
    
    axes[1, 3].text(0.1, 0.9, 'Reconstructed SDF Stats:', fontsize=12, weight='bold')
    axes[1, 3].text(0.1, 0.8, f'Min: {recon.min():.3f}', fontsize=10)
    axes[1, 3].text(0.1, 0.7, f'Max: {recon.max():.3f}', fontsize=10)
    axes[1, 3].text(0.1, 0.6, f'Mean: {recon.mean():.3f}', fontsize=10)
    axes[1, 3].text(0.1, 0.5, f'Std: {recon.std():.3f}', fontsize=10)
    axes[1, 3].axis('off')
    
    axes[2, 3].text(0.1, 0.9, 'Reconstruction Quality:', fontsize=12, weight='bold')
    axes[2, 3].text(0.1, 0.8, f'MSE: {mse:.6f}', fontsize=10)
    axes[2, 3].text(0.1, 0.7, f'MAE: {mae:.6f}', fontsize=10)
    axes[2, 3].text(0.1, 0.6, f'Negative voxels (orig): {(orig < 0).sum()}', fontsize=10)
    axes[2, 3].text(0.1, 0.5, f'Negative voxels (recon): {(recon < 0).sum()}', fontsize=10)
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sdf_comparison_sample_{sample_idx}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return mse, mae


def test_vqvae_reconstruction(args):
    """测试VQVAE重建质量"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载VQVAE模型
    vqvae = load_vqvae_model(args.checkpoint_path, device)
    
    # 加载测试数据集
    dataset_config = {
        "info_file": './dataset_info_files/info-shapenet.json',
        "dataroot": './data',
        "phase": 'test',  # 使用测试集
        "cat": args.category,
        "res": 64,
        "trunc_thres": 0.2
    }
    
    dataset = ShapeNetDataSet(
        info_file=dataset_config["info_file"],
        dataroot=dataset_config["dataroot"],
        phase=dataset_config["phase"],
        cat=dataset_config["cat"],
        res=dataset_config["res"],
        trunc_thres=dataset_config["trunc_thres"]
    )
    
    print(f"Test dataset loaded with {len(dataset)} samples")
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # 测试指定数量的样本
    total_mse = 0
    total_mae = 0
    processed_samples = 0
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= args.num_samples:
                break
                
            print(f"\nProcessing sample {i+1}/{args.num_samples}")
            
            # 获取原始SDF
            original_sdf = data['sdf'].to(device)  # Shape: (1, 1, 64, 64, 64)
            print(f"Original SDF shape: {original_sdf.shape}")
            print(f"Original SDF range: [{original_sdf.min():.3f}, {original_sdf.max():.3f}]")
            
            # VQVAE重建
            reconstructed_sdf, vq_loss = vqvae(original_sdf, verbose=False)
            print(f"Reconstructed SDF shape: {reconstructed_sdf.shape}")
            print(f"Reconstructed SDF range: [{reconstructed_sdf.min():.3f}, {reconstructed_sdf.max():.3f}]")
            print(f"VQ Loss: {vq_loss.item():.6f}")
            
            # 转换为numpy用于处理
            original_np = original_sdf.cpu().numpy()
            reconstructed_np = reconstructed_sdf.cpu().numpy()
            
            # 可视化对比
            mse, mae = visualize_sdf_comparison(original_np, reconstructed_np, args.output_dir, i)
            total_mse += mse
            total_mae += mae
            
            # 从原始SDF生成mesh
            print("Generating mesh from original SDF...")
            orig_sdf_vol = original_np[0, 0]  # Shape: (64, 64, 64)
            orig_verts, orig_faces, orig_normals = sdf_to_mesh(orig_sdf_vol, threshold=args.sdf_threshold)
            
            if orig_verts is not None:
                orig_obj_path = os.path.join(args.output_dir, f'original_sample_{i}.obj')
                save_mesh_as_obj(orig_verts, orig_faces, orig_obj_path)
            
            # 从重建SDF生成mesh
            print("Generating mesh from reconstructed SDF...")
            recon_sdf_vol = reconstructed_np[0, 0]  # Shape: (64, 64, 64)
            recon_verts, recon_faces, recon_normals = sdf_to_mesh(recon_sdf_vol, threshold=args.sdf_threshold)
            
            if recon_verts is not None:
                recon_obj_path = os.path.join(args.output_dir, f'reconstructed_sample_{i}.obj')
                save_mesh_as_obj(recon_verts, recon_faces, recon_obj_path)
            
            processed_samples += 1
            
            # 打印当前样本的重建质量
            print(f"Sample {i} - MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    # 计算平均重建质量
    if processed_samples > 0:
        avg_mse = total_mse / processed_samples
        avg_mae = total_mae / processed_samples
        
        print(f"\n=== Overall Reconstruction Quality ===")
        print(f"Processed samples: {processed_samples}")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average MAE: {avg_mae:.6f}")
        
        # 保存统计结果
        with open(os.path.join(args.output_dir, 'reconstruction_stats.txt'), 'w') as f:
            f.write(f"VQVAE Reconstruction Test Results\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Category: {args.category}\n")
            f.write(f"Processed samples: {processed_samples}\n")
            f.write(f"Average MSE: {avg_mse:.6f}\n")
            f.write(f"Average MAE: {avg_mae:.6f}\n")
            f.write(f"SDF threshold: {args.sdf_threshold}\n")
    
    print(f"\nTesting completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test VQVAE SDF Reconstruction')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the trained VQVAE checkpoint')
    parser.add_argument('--output_dir', type=str, default='test_vqvae_results',
                        help='Directory to save test results')
    parser.add_argument('--category', type=str, default='chair',
                        help='Object category to test (chair, table, etc.)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to test')
    parser.add_argument('--sdf_threshold', type=float, default=0.0,
                        help='SDF threshold for marching cubes')
    
    args = parser.parse_args()
    
    print("=== VQVAE Reconstruction Test ===")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Category: {args.category}")
    print(f"Number of samples: {args.num_samples}")
    print(f"SDF threshold: {args.sdf_threshold}")
    
    test_vqvae_reconstruction(args)