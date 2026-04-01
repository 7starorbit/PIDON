import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# ==================== 可视化函数 ====================
def visualize_results(model, dataset, device, num_samples=4, indices=None, save_dir='results'):
    """可视化预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if indices is None:
        num_samples = min(num_samples, len(dataset))
        indices = torch.arange(num_samples)
    
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            E, r, curl_target = dataset[indices[i]]
            E_mean = torch.mean(E)
            E_std = torch.std(E)
            r_mean = torch.mean(r)
            r_std = torch.std(r)
            E = (E - E_mean) / (E_std + 1e-7)
            r = (r - r_mean) / (r_std + 1e-7)
            E = E.unsqueeze(0).to(device)
            r = r.unsqueeze(0).to(device)
            curl_pred = model(E, r).cpu().squeeze(0)
            curl_target = curl_target.cpu()

            curl_pred = curl_pred * (E_std + 1e-7) / (r_std + 1e-7)
            
            # 取中间切片
            z_mid = curl_target.shape[-1] // 2
            
            # 绘制三个分量的真实值和预测值
            comp_names = ['x', 'y', 'z']
            for comp in range(3):
                true_slice = curl_target[comp, :, :, z_mid].numpy()
                pred_slice = curl_pred[comp, :, :, z_mid].numpy()

                vmin = min(true_slice.min(), pred_slice.min())
                vmax = max(true_slice.max(), pred_slice.max())
                # 真实值
                ax = axes[i, comp]
                im = ax.imshow(true_slice, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax.set_title(f'Sample {i+1}: True ∇×E_{comp_names[comp]}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
                
                # 预测值
                ax = axes[i, comp+3]
                im = ax.imshow(pred_slice, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax.set_title(f'Sample {i+1}: Pred ∇×E_{comp_names[comp]}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('DeepONet Curl Operator: True vs Predicted (Mid-plane Slice)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"可视化结果已保存到 {save_dir}/prediction_comparison.png")


def visualize_error_distribution(model, dataset, device, num_samples=4, indices=None, save_dir='results'):
    """可视化误差分布"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if indices is None:
        num_samples = min(num_samples, len(dataset))
        indices = torch.arange(num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            E, r, curl_target = dataset[indices[i]]
            E_mean = torch.mean(E)
            E_std = torch.std(E)
            r_mean = torch.mean(r)
            r_std = torch.std(r)
            E = (E - E_mean) / (E_std + 1e-7)
            r = (r - r_mean) / (r_std + 1e-7)

            E = E.unsqueeze(0).to(device)
            r = r.unsqueeze(0).to(device)
            curl_pred = model(E, r).cpu().squeeze(0)
            curl_target = curl_target.cpu()

            curl_pred = curl_pred * (E_std + 1e-7) / (r_std + 1e-7)

            # 计算误差
            abs_error = torch.abs(curl_pred)
            rel_error = torch.abs(curl_pred - curl_target) / torch.abs(curl_target)

            mask = torch.abs(curl_target) > 1e-1
            error = torch.where(mask, rel_error, abs_error)
            
            z_mid = error.shape[-1] // 2
            
            comp_names = ['x', 'y', 'z']
            for comp in range(3):
                ax = axes[i, comp]
                error_slice = error[comp, :, :, z_mid].numpy()
                im = ax.imshow(error_slice, cmap='hot')
                ax.set_title(f'Sample {i+1}: Error |∇×E_{comp_names[comp]}|')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Absolute Error Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"误差分布已保存到 {save_dir}/error_distribution.png")


def plot_true_vs_rel_error(model, dataset, device, num_samples=4, indices=None, mode='all', z_slice=None, save_dir='results'):
    """
    mode: 'all' -> 统计每个样本体积内所有点
          'slice' -> 仅统计中间切片或指定 z_slice
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if indices is None:
        num_samples = min(num_samples, len(dataset))
        indices = torch.arange(num_samples)

    true_vals = [[], [], []]
    rel_vals = [[], [], []]

    with torch.no_grad():
        for i in range(len(indices)):
            E, r, curl_target = dataset[indices[i]]
            E_mean = torch.mean(E); E_std = torch.std(E)
            r_mean = torch.mean(r); r_std = torch.std(r)

            E_n = (E - E_mean) / (E_std + 1e-7)
            r_n = (r - r_mean) / (r_std + 1e-7)
            E_n = E_n.unsqueeze(0).to(device)
            r_n = r_n.unsqueeze(0).to(device)

            curl_pred = model(E_n, r_n).cpu().squeeze(0)
            curl_target = curl_target.cpu()
            curl_pred = curl_pred * (E_std + 1e-7) / (r_std + 1e-7)

            if mode == 'all':
                for comp in range(3):
                    t = curl_target[comp].reshape(-1).numpy()
                    p = curl_pred[comp].reshape(-1).numpy()
                    rel = np.where(np.abs(t) > 1e-1, np.abs(p - t) / np.abs(t), np.abs(p))
                    true_vals[comp].append(t)
                    rel_vals[comp].append(rel)
            else:  # 'slice'
                z = z_slice if z_slice is not None else curl_target.shape[-1] // 2
                for comp in range(3):
                    t = curl_target[comp, :, :, z].numpy().ravel()
                    p = curl_pred[comp, :, :, z].numpy().ravel()
                    rel = np.where(np.abs(t) > 1e-1, np.abs(p - t) / np.abs(t), np.abs(p))
                    true_vals[comp].append(t)
                    rel_vals[comp].append(rel)

    for comp in range(3):
        if len(true_vals[comp]) == 0:
            true_vals[comp] = np.array([])
            rel_vals[comp] = np.array([])
        else:
            true_vals[comp] = np.concatenate(true_vals[comp])
            rel_vals[comp] = np.concatenate(rel_vals[comp])

    comp_names = ['x', 'y', 'z']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for comp in range(3):
        ax = axes[comp]
        if true_vals[comp].size > 0:
            ax.scatter(true_vals[comp], rel_vals[comp], s=1, alpha=0.5)
        ax.set_xlabel(f'True ∇×E_{comp_names[comp]}')
        ax.set_ylabel('Relative error')
        ax.set_title(f'True vs RelError ({mode}) ∇×E_{comp_names[comp]}')
        ax.set_yscale('log')
    plt.tight_layout()
    out = os.path.join(save_dir, f'true_vs_rel_error_{mode}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'已保存 {out}')