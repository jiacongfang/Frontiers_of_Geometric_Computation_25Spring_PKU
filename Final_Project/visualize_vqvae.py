# TODO. the script need to verify and modify. 

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from torch.utils.data import DataLoader

from vqvae.network import VQVAE

def visualize_sdf_slice(sdf, title="SDF", slice_dim=32, figsize=(10, 8)):
    """可视化SDF的2D切片"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 去除批次和通道维度
    if len(sdf.shape) == 5:  # [B, C, D, H, W]
        sdf = sdf[0, 0]  # 取第一个样本的第一个通道
    elif len(sdf.shape) == 4:  # [C, D, H, W]
        sdf = sdf[0]  # 取第一个通道
    
    # XY平面 (固定Z)
    axes[0, 0].imshow(sdf[slice_dim, :, :].cpu().numpy(), cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 0].set_title(f'XY plane (Z={slice_dim})')
    axes[0, 0].axis('off')
    
    # XZ平面 (固定Y)
    axes[0, 1].imshow(sdf[:, slice_dim, :].cpu().numpy(), cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 1].set_title(f'XZ plane (Y={slice_dim})')
    axes[0, 1].axis('off')
    
    # YZ平面 (固定X)
    axes[0, 2].imshow(sdf[:, :, slice_dim].cpu().numpy(), cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 2].set_title(f'YZ plane (X={slice_dim})')
    axes[0, 2].axis('off')
    
    # 直方图
    axes[1, 0].hist(sdf.cpu().numpy().flatten(), bins=50, alpha=0.7)
    axes[1, 0].set_title('SDF Value Distribution')
    axes[1, 0].set_xlabel('SDF Value')
    axes[1, 0].set_ylabel('Frequency')
    
    # 统计信息
    sdf_np = sdf.cpu().numpy()
    stats_text = f"""Statistics:
Min: {sdf_np.min():.3f}
Max: {sdf_np.max():.3f}
Mean: {sdf_np.mean():.3f}
Std: {sdf_np.std():.3f}
Zero-level: {(np.abs(sdf_np) < 0.1).sum()}/{sdf_np.size}"""
    
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='center')
    axes[1, 1].axis('off')
    
    # 空白
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig

def extract_mesh_from_sdf(sdf, threshold=0.0, spacing=(1.0, 1.0, 1.0)):
    """使用scikit-image的marching cubes从SDF中提取网格"""
    if len(sdf.shape) == 5:  # [B, C, D, H, W]
        sdf = sdf[0, 0]  # 取第一个样本的第一个通道
    elif len(sdf.shape) == 4:  # [C, D, H, W]
        sdf = sdf[0]  # 取第一个通道
    
    sdf_np = sdf.cpu().numpy()
    
    try:
        # 使用scikit-image的marching cubes提取等值面
        # measure.marching_cubes返回 (vertices, faces, normals, values)
        vertices, faces, normals, values = marching_cubes(
            sdf_np, 
            level=threshold,
            spacing=spacing
        )
        return vertices, faces, normals
    except Exception as e:
        print(f"Failed to extract mesh: {e}")
        return None, None, None

def visualize_mesh_3d(vertices, faces, title="3D Mesh", figsize=(10, 8)):
    """3D可视化网格"""
    if vertices is None or faces is None:
        print("No mesh to visualize")
        return None
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制三角形网格
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                    triangles=faces, alpha=0.8, shade=True, cmap='viridis')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置相等的坐标轴比例
    max_range = np.array([vertices[:,0].max()-vertices[:,0].min(), 
                         vertices[:,1].max()-vertices[:,1].min(),
                         vertices[:,2].max()-vertices[:,2].min()]).max() / 2.0
    
    mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
    mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
    mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return fig

def compare_reconstructions(original_sdf, reconstructed_sdf, save_dir="comparison_results"):
    """比较原始SDF和重构SDF"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 2D切片比较
    fig1 = visualize_sdf_slice(original_sdf, "Original SDF")
    fig1.savefig(f"{save_dir}/original_sdf_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    fig2 = visualize_sdf_slice(reconstructed_sdf, "Reconstructed SDF")
    fig2.savefig(f"{save_dir}/reconstructed_sdf_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 差异可视化
    diff_sdf = torch.abs(original_sdf - reconstructed_sdf)
    fig3 = visualize_sdf_slice(diff_sdf, "Absolute Difference")
    fig3.savefig(f"{save_dir}/difference_sdf_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 3D网格比较
    orig_vertices, orig_faces, orig_normals = extract_mesh_from_sdf(original_sdf)
    recon_vertices, recon_faces, recon_normals = extract_mesh_from_sdf(reconstructed_sdf)
    
    if orig_vertices is not None:
        fig4 = visualize_mesh_3d(orig_vertices, orig_faces, "Original Mesh")
        fig4.savefig(f"{save_dir}/original_mesh.png", dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"  Original mesh: {len(orig_vertices)} vertices, {len(orig_faces)} faces")
    
    if recon_vertices is not None:
        fig5 = visualize_mesh_3d(recon_vertices, recon_faces, "Reconstructed Mesh")
        fig5.savefig(f"{save_dir}/reconstructed_mesh.png", dpi=150, bbox_inches='tight')
        plt.close(fig5)
        print(f"  Reconstructed mesh: {len(recon_vertices)} vertices, {len(recon_faces)} faces")
    
    # 计算定量指标
    mse = torch.mean((original_sdf - reconstructed_sdf) ** 2).item()
    mae = torch.mean(torch.abs(original_sdf - reconstructed_sdf)).item()
    
    print(f"Reconstruction Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    return mse, mae

def evaluate_vqvae_qualitatively(model_path, data_loader=None, num_samples=5, save_dir="vqvae_evaluation"):
    """定性评估VQVAE"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型配置
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
    
    # 创建和加载模型
    vqvae = VQVAE(
        ddconfig=model_config["ddconfig"],
        n_embed=model_config["n_embed"],
        embed_dim=model_config["embed_dim"],
        remap=None,
        sane_index_shape=False
    )
    
    vqvae.load_state_dict(torch.load(model_path, map_location=device))
    vqvae = vqvae.to(device)
    vqvae.eval()
    
    print(f"Model loaded from: {model_path}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 如果没有提供数据加载器，使用随机数据
    if data_loader is None:
        print("No data loader provided, using random test data")
        test_data = torch.randn(num_samples, 1, 64, 64, 64).to(device)
        data_list = [{'sdf': test_data[i:i+1]} for i in range(num_samples)]
    else:
        print("Using provided data loader")
        data_list = []
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
            data_list.append(batch)
    
    all_mse = []
    all_mae = []
    
    with torch.no_grad():
        for i, data in enumerate(data_list):
            print(f"\nProcessing sample {i+1}/{len(data_list)}")
            
            if isinstance(data, dict):
                x = data['sdf'].to(device)
            else:
                x = data.to(device)
            
            # 前向传播
            x_recon, qloss = vqvae(x, verbose=False)
            
            # 创建样本特定的保存目录
            sample_dir = f"{save_dir}/sample_{i+1}"
            
            # 比较重构结果
            mse, mae = compare_reconstructions(x, x_recon, sample_dir)
            all_mse.append(mse)
            all_mae.append(mae)
            
            print(f"  Sample {i+1} - MSE: {mse:.6f}, MAE: {mae:.6f}")
            
            # 编码分析
            z = vqvae(x, forward_no_quant=True, encode_only=True)
            print(f"  Encoded shape: {z.shape}")
            print(f"  Encoded range: [{z.min().item():.3f}, {z.max().item():.3f}]")
    
    # 总体统计
    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)
    std_mse = np.std(all_mse)
    std_mae = np.std(all_mae)
    
    print(f"\n{'='*50}")
    print(f"Overall Reconstruction Quality:")
    print(f"  Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"  Average MAE: {avg_mae:.6f} ± {std_mae:.6f}")
    print(f"{'='*50}")
    
    # 保存统计结果
    with open(f"{save_dir}/evaluation_results.txt", 'w') as f:
        f.write(f"VQVAE Evaluation Results\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Number of samples: {len(data_list)}\n")
        f.write(f"Average MSE: {avg_mse:.6f} ± {std_mse:.6f}\n")
        f.write(f"Average MAE: {avg_mae:.6f} ± {std_mae:.6f}\n")
        f.write(f"\nPer-sample results:\n")
        for i, (mse, mae) in enumerate(zip(all_mse, all_mae)):
            f.write(f"Sample {i+1}: MSE={mse:.6f}, MAE={mae:.6f}\n")

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VQVAE Qualitative Evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained VQVAE model')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to evaluate')
    parser.add_argument('--save_dir', type=str, default='vqvae_evaluation',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # 运行评估
    evaluate_vqvae_qualitatively(
        model_path=args.model_path,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )
    
    print(f"\nEvaluation completed! Results saved in: {args.save_dir}")
    print("Check the generated images and evaluation_results.txt for detailed analysis.")