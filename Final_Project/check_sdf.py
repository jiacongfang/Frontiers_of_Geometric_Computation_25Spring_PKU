import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse

def inspect_h5_file(file_path, verbose=True):
    """检查HDF5文件的结构和内容"""
    print(f"检查文件: {file_path}")
    print(f"文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    with h5py.File(file_path, 'r') as f:
        print(f"\n=== HDF5 文件结构 ===")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"数据集: {name}")
                print(f"  形状: {obj.shape}")
                print(f"  数据类型: {obj.dtype}")
                print(f"  大小: {obj.size}")
                if obj.size > 0:
                    print(f"  值范围: [{obj[...].min():.4f}, {obj[...].max():.4f}]")
                    print(f"  均值: {obj[...].mean():.4f}")
                    print(f"  标准差: {obj[...].std():.4f}")
                print()
            elif isinstance(obj, h5py.Group):
                print(f"组: {name}")
        
        f.visititems(print_structure)
        
        # 返回主要数据
        data_keys = list(f.keys())
        print(f"顶层键: {data_keys}")
        
        # 尝试读取SDF数据
        sdf_data = None
        possible_keys = ['sdf', 'data', 'voxels', 'volume', 'grid']
        
        for key in possible_keys:
            if key in f:
                sdf_data = f[key][...]
                print(f"找到SDF数据，键名: {key}")
                break
        
        if sdf_data is None and len(data_keys) == 1:
            # 如果只有一个键，假设它就是SDF数据
            sdf_data = f[data_keys[0]][...]
            print(f"使用默认键: {data_keys[0]}")
        
        return sdf_data

def visualize_sdf(sdf_data, title="SDF可视化", save_path=None):
    """可视化SDF数据"""
    if sdf_data is None:
        print("没有找到SDF数据进行可视化")
        return
    
    print(f"\n=== SDF数据分析 ===")
    print(f"形状: {sdf_data.shape}")
    print(f"数据类型: {sdf_data.dtype}")
    print(f"值范围: [{sdf_data.min():.4f}, {sdf_data.max():.4f}]")
    print(f"均值: {sdf_data.mean():.4f}")
    print(f"标准差: {sdf_data.std():.4f}")
    print(f"零值附近的点数 (|sdf| < 0.1): {np.sum(np.abs(sdf_data) < 0.1)}")
    print(f"负值点数 (内部): {np.sum(sdf_data < 0)}")
    print(f"正值点数 (外部): {np.sum(sdf_data > 0)}")
    
    # 确保数据是3D的
    if len(sdf_data.shape) == 3:
        pass  # 已经是3D
    elif len(sdf_data.shape) == 4 and sdf_data.shape[0] == 1:
        sdf_data = sdf_data[0]  # 去除批次维度
    elif len(sdf_data.shape) == 4 and sdf_data.shape[-1] == 1:
        sdf_data = sdf_data[..., 0]  # 去除通道维度
    else:
        print(f"警告: 不支持的SDF数据形状: {sdf_data.shape}")
        return
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    D, H, W = sdf_data.shape
    mid_d, mid_h, mid_w = D//2, H//2, W//2
    
    # XY平面 (固定Z)
    im1 = axes[0, 0].imshow(sdf_data[mid_d, :, :], cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 0].set_title(f'XY plane (Z={mid_d})')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # XZ平面 (固定Y)
    im2 = axes[0, 1].imshow(sdf_data[:, mid_h, :], cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 1].set_title(f'XZ plane (Y={mid_h})')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # YZ平面 (固定X)
    im3 = axes[0, 2].imshow(sdf_data[:, :, mid_w], cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 2].set_title(f'YZ plane (X={mid_w})')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 直方图
    axes[1, 0].hist(sdf_data.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('SDF值分布')
    axes[1, 0].set_xlabel('SDF值')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', label='零等值面')
    axes[1, 0].legend()
    
    # 统计信息
    stats_text = f"""统计信息:
形状: {sdf_data.shape}
最小值: {sdf_data.min():.4f}
最大值: {sdf_data.max():.4f}
均值: {sdf_data.mean():.4f}
标准差: {sdf_data.std():.4f}
零值附近: {np.sum(np.abs(sdf_data) < 0.1)}
内部点: {np.sum(sdf_data < 0)}
外部点: {np.sum(sdf_data > 0)}"""
    
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    # 等值面信息
    zero_level = np.sum(np.abs(sdf_data) < 0.01)  # 更严格的零等值面
    axes[1, 2].text(0.1, 0.5, f"零等值面点数:\n{zero_level}", 
                    transform=axes[1, 2].transAxes, fontsize=12, 
                    verticalalignment='center', ha='center')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化保存到: {save_path}")
    
    plt.show()

def check_sdf_validity(sdf_data):
    """检查SDF数据的有效性"""
    print(f"\n=== SDF有效性检查 ===")
    
    issues = []
    
    # 检查形状
    if len(sdf_data.shape) != 3:
        issues.append(f"形状不是3D: {sdf_data.shape}")
    
    # 检查数据类型
    if not np.issubdtype(sdf_data.dtype, np.floating):
        issues.append(f"数据类型不是浮点数: {sdf_data.dtype}")
    
    # 检查是否有无穷大或NaN值
    if np.any(np.isinf(sdf_data)):
        issues.append("包含无穷大值")
    
    if np.any(np.isnan(sdf_data)):
        issues.append("包含NaN值")
    
    # 检查值域是否合理
    if sdf_data.max() > 10 or sdf_data.min() < -10:
        issues.append(f"SDF值域可能不合理: [{sdf_data.min():.2f}, {sdf_data.max():.2f}]")
    
    # 检查是否有表面（零等值面附近的点）
    near_zero = np.sum(np.abs(sdf_data) < 0.1)
    if near_zero == 0:
        issues.append("没有找到零等值面附近的点，可能没有表面")
    
    # 检查内外部分布
    inside_points = np.sum(sdf_data < 0)
    outside_points = np.sum(sdf_data > 0)
    if inside_points == 0:
        issues.append("没有内部点（负值）")
    if outside_points == 0:
        issues.append("没有外部点（正值）")
    
    if issues:
        print("发现问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ SDF数据看起来正常")
    
    return len(issues) == 0

def main():
    parser = argparse.ArgumentParser(description='检查HDF5格式的SDF文件')
    parser.add_argument('--file_path', type=str, required=True, help='HDF5文件路径')
    parser.add_argument('--save_viz', type=str, help='保存可视化图片的路径')
    parser.add_argument('--no_viz', action='store_true', help='不显示可视化')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"文件不存在: {args.file_path}")
        return
    
    try:
        # 检查文件结构和读取数据
        sdf_data = inspect_h5_file(args.file_path)
        
        if sdf_data is not None:
            # 检查数据有效性
            is_valid = check_sdf_validity(sdf_data)
            
            # 可视化（如果需要）
            if not args.no_viz:
                save_path = args.save_viz if args.save_viz else None
                visualize_sdf(sdf_data, f"SDF: {os.path.basename(args.file_path)}", save_path)
        else:
            print("无法找到SDF数据")
            
    except Exception as e:
        print(f"读取文件时出错: {e}")
        import traceback
        

        aceback.print_exc()

if __name__ == "__main__":
    main()
