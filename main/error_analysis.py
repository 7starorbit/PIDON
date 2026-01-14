"""
误差分析脚本 - 诊断极值点问题
"""
import torch
import numpy as np
from deeponet import DeepONet3D
from data import DCO_dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型和数据
print("加载模型和数据...")
model = DeepONet3D(in_ch=3, out_ch=3, base_ch=32, num_layers=4).to(device)
checkpoint = torch.load('checkpoints/best_model_lbfgs.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_dataset = DCO_dataset(mode='test', normalize=False)

# 选择一个样本进行详细分析
sample_idx = 1  # Sample 2 从图中看误差最大
E, r, curl_target = test_dataset[sample_idx]

with torch.no_grad():
    E_input = E.unsqueeze(0).to(device)
    r_input = r.unsqueeze(0).to(device)
    curl_pred = model(E_input, r_input).cpu().squeeze(0)

print(f"\n{'='*70}")
print(f"样本 {sample_idx+1} 的详细误差分析")
print(f"{'='*70}")

# 逐分量分析
comp_names = ['x', 'y', 'z']
for comp in range(3):
    target = curl_target[comp].numpy()
    pred = curl_pred[comp].numpy()
    abs_error = np.abs(pred - target)
    
    # 相对误差 (逐点)
    rel_error = abs_error / (np.abs(target) + 1e-8)
    
    print(f"\n分量 {comp_names[comp]}:")
    print(f"  真实值范围: [{target.min():.6f}, {target.max():.6f}]")
    print(f"  预测值范围: [{pred.min():.6f}, {pred.max():.6f}]")
    print(f"  绝对误差: 均值={abs_error.mean():.6e}, 最大={abs_error.max():.6e}")
    print(f"  相对误差: 均值={rel_error.mean():.6f}, 最大={rel_error.max():.6f}, 中位数={np.median(rel_error):.6f}")
    
    # 找到相对误差最大的点
    max_rel_idx = np.unravel_index(np.argmax(rel_error), rel_error.shape)
    print(f"  最大相对误差点位置: {max_rel_idx}")
    print(f"    - 真实值: {target[max_rel_idx]:.6e}")
    print(f"    - 预测值: {pred[max_rel_idx]:.6e}")
    print(f"    - 绝对误差: {abs_error[max_rel_idx]:.6e}")
    print(f"    - 相对误差: {rel_error[max_rel_idx]:.6f}")
    
    # 统计接近零的点
    threshold = np.abs(target).max() * 0.01  # 1% 阈值
    small_values = np.abs(target) < threshold
    print(f"  接近零的点数 (< {threshold:.6e}): {small_values.sum()} / {target.size} ({100*small_values.sum()/target.size:.2f}%)")
    
    # 找出相对误差>10的点
    outliers = rel_error > 10
    print(f"  相对误差>10的点数: {outliers.sum()} / {target.size} ({100*outliers.sum()/target.size:.2f}%)")
    if outliers.sum() > 0:
        print(f"    这些点的真实值范围: [{target[outliers].min():.6e}, {target[outliers].max():.6e}]")
        print(f"    这些点的预测值范围: [{pred[outliers].min():.6e}, {pred[outliers].max():.6e}]")

# 可视化真实值的分布
print(f"\n{'='*70}")
print("生成真实值分布直方图...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for comp in range(3):
    target = curl_target[comp].numpy().flatten()
    axes[comp].hist(np.abs(target), bins=100, edgecolor='black', alpha=0.7)
    axes[comp].set_xlabel('|真实值|')
    axes[comp].set_ylabel('频数')
    axes[comp].set_title(f'分量 {comp_names[comp]} 的真实值分布')
    axes[comp].set_yscale('log')
    axes[comp].axvline(np.abs(target).max() * 0.01, color='r', linestyle='--', label='1% 阈值')
    axes[comp].legend()
plt.tight_layout()
plt.savefig('results/true_value_distribution.png', dpi=150, bbox_inches='tight')
print("已保存到 results/true_value_distribution.png")

# 3D可视化误差极值点
print("\n生成3D误差极值点可视化...")
fig = plt.figure(figsize=(15, 5))
for comp in range(3):
    target = curl_target[comp].numpy()
    pred = curl_pred[comp].numpy()
    abs_error = np.abs(pred - target)
    rel_error = abs_error / (np.abs(target) + 1e-8)
    
    # 找出相对误差>10的点的坐标
    outliers = rel_error > 10
    if outliers.sum() > 0:
        coords = np.argwhere(outliers)
        
        ax = fig.add_subplot(1, 3, comp+1, projection='3d')
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                           c=rel_error[outliers], cmap='hot', s=20)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'分量 {comp_names[comp]}: 相对误差>10的点\n(共{outliers.sum()}个)')
        plt.colorbar(scatter, ax=ax, label='相对误差')
plt.tight_layout()
plt.savefig('results/error_outliers_3d.png', dpi=150, bbox_inches='tight')
print("已保存到 results/error_outliers_3d.png")

plt.show()

# 建议
print(f"\n{'='*70}")
print("诊断结论和建议:")
print(f"{'='*70}")
print("1. 如果极值点的真实值接近零,这是正常的数值现象")
print("   → 建议使用归一化相对误差: error/(max_value + eps)")
print("\n2. 如果极值点集中在边界,可能是边界条件处理不当")
print("   → 建议检查数据生成过程的边界条件")
print("\n3. 如果极值点分布在物理上无旋的区域")
print("   → 建议对不同区域使用不同的误差度量")
print("\n4. 如果整体误差分布合理,只是少数几个点异常")
print("   → 可以使用中位数相对误差(MRE)或忽略极小值点")
print(f"{'='*70}")
