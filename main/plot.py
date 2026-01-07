import os
import torch
import matplotlib.pyplot as plt

# ==================== 可视化函数 ====================
def visualize_results(model, dataset, device, num_samples=4, save_dir='results'):
    """可视化预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 获取归一化统计量
    curl_mean = dataset.curl_mean
    curl_std = dataset.curl_std

    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            E, r, curl_target = dataset[i]
            E = E.unsqueeze(0).to(device)
            r = r.unsqueeze(0).to(device)
            curl_pred = model(E, r).cpu().squeeze(0)
            curl_target = curl_target.cpu()
            # 反归一化
            curl_pred = curl_pred * (curl_std.squeeze(0) + 1e-8) + curl_mean.squeeze(0)
            curl_target = curl_target * (curl_std.squeeze(0) + 1e-8) + curl_mean.squeeze(0)
            
            # 取中间切片
            z_mid = curl_target.shape[-1] // 2
            
            # 绘制三个分量的真实值和预测值
            comp_names = ['x', 'y', 'z']
            for comp in range(3):
                # 真实值
                ax = axes[i, comp]
                true_slice = curl_target[comp, :, :, z_mid].numpy()
                im = ax.imshow(true_slice, cmap='RdBu_r')
                ax.set_title(f'Sample {i+1}: True ∇×E_{comp_names[comp]}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
                
                # 预测值
                ax = axes[i, comp+3]
                pred_slice = curl_pred[comp, :, :, z_mid].numpy()
                im = ax.imshow(pred_slice, cmap='RdBu_r')
                ax.set_title(f'Sample {i+1}: Pred ∇×E_{comp_names[comp]}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('DeepONet Curl Operator: True vs Predicted (Mid-plane Slice)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"可视化结果已保存到 {save_dir}/prediction_comparison.png")


def visualize_error_distribution(model, dataset, device, num_samples=4, save_dir='results'):
    """可视化误差分布"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    curl_mean = dataset.curl_mean
    curl_std = dataset.curl_std
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            E, r, curl_target = dataset[i]
            E = E.unsqueeze(0).to(device)
            r = r.unsqueeze(0).to(device)
            curl_pred = model(E, r).cpu().squeeze(0)
            curl_target = curl_target.cpu()

            # 反归一化
            curl_pred = curl_pred * (curl_std.squeeze(0) + 1e-8) + curl_mean.squeeze(0)
            curl_target = curl_target * (curl_std.squeeze(0) + 1e-8) + curl_mean.squeeze(0)
            
            # 计算误差
            error = torch.abs(curl_pred - curl_target)
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