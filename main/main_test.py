import torch
import numpy as np
from deeponet import DeepONet3D
from data import DCO_dataset
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
from plot import visualize_results, visualize_error_distribution

def main():
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    base_features = 32

    print("加载数据集...")
    train_dataset = DCO_dataset(mode='train', normalize=True)
    test_dataset = DCO_dataset(mode='test', normalize=True)

    print("初始化模型...")
    model = DeepONet3D(in_ch=3, out_ch=3, base_ch=base_features, num_layers=4).to(device)

    # 加载最佳模型并可视化
    print("加载最佳模型进行可视化...")
    checkpoint = torch.load('checkpoints/best_model_lbfgs.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"最佳模型来自 Epoch {checkpoint['epoch']+1}")
    print(f"  Test Loss: {checkpoint['test_loss']:.6f}")
    print(f"  Test MRE: {checkpoint['test_mre']:.6f}\n")

    num_plot_samples = 4
    torch.manual_seed(int(time.time() * 1000) % (2**32))
    indices = torch.randperm(len(test_dataset))[:num_plot_samples]
    print(f"随机选择的可视化样本索引: {indices.tolist()}")
    # 可视化预测结果
    visualize_results(model, test_dataset, device, num_samples=num_plot_samples, indices=indices)
    
    # 可视化误差分布
    visualize_error_distribution(model, test_dataset, device, num_samples=num_plot_samples, indices=indices)

    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()