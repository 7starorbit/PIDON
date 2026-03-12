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

    batch_size = 30
    num_epochs_adam = 100
    lr_adam = 0.001
    base_features = 32

    print("加载数据集...")
    train_dataset = DCO_dataset(mode='train', normalize=False)
    test_dataset = DCO_dataset(mode='test', normalize=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    print("初始化模型...")
    model = DeepONet3D(in_ch=3, out_ch=3, base_ch=base_features, num_layers=4).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    branch_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'branch' in name)
    trunk_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'trunk' in name)
    decoder_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'decoder' in name)

    print(f"\n{'='*70}")
    print(f"{'DeepONet 3D-UNet 旋度算子学习器':^70}")
    print(f"{'='*70}")
    print(f"  - Branch Network (电场编码器): {branch_params/1e6:.2f}M 参数")
    print(f"  - Trunk Network (坐标编码器):  {trunk_params/1e6:.2f}M 参数")
    print(f"  - Decoder (解码器):            {decoder_params/1e6:.2f}M 参数")
    print(f"  总参数量: {total_params/1e6:.2f}M")
    print(f"  激活函数: GELU")
    print(f"  基础特征数: {base_features}")
    print(f"  Batch Size: {batch_size}")
    print(f"  误差指标: MRE (Mean Relative Error)")
    print(f"{'='*70}\n")

    # criterion = torch.nn.MSELoss()

    # history = {
    #     'train_loss': [],
    #     'test_loss': [],
    #     'test_mre': [],
    #     'phase': []
    # }
    # best_loss = float('inf')
    # os.makedirs('checkpoints', exist_ok=True)

    # print("开始训练Adam阶段...\n")
    # optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)

    # for epoch in range(num_epochs_adam):
    #     # ===== 训练阶段 =====
    #     model.train()
    #     train_loss = 0.0
    #     for E, r, curl_target in train_loader:
    #         E, r, curl_target = E.to(device), r.to(device), curl_target.to(device)

    #         optimizer_adam.zero_grad()
    #         curl_pred = model(E, r)
    #         loss = criterion(curl_pred, curl_target)
    #         loss.backward()
    #         optimizer_adam.step()

    #         train_loss += loss.item()
    #     train_loss /= len(train_loader)

    #     # ===== 测试阶段 =====
    #     model.eval()
    #     test_loss = 0.0
    #     with torch.no_grad():
    #         for E, r, curl_target in test_loader:
    #             E, r, curl_target = E.to(device), r.to(device), curl_target.to(device)

    #             curl_pred = model(E, r)
    #             loss = criterion(curl_pred, curl_target)
    #             test_loss += loss.item()
    #         test_loss /= len(test_loader)

    #     # ===== 计算误差 =====
    #     total_mre = 0.0
    #     num_samples = 0
    #     with torch.no_grad():
    #         for E, r, curl_target in test_loader:
    #             E, r, curl_target = E.to(device), r.to(device), curl_target.to(device)

    #             curl_pred = model(E, r)

    #             if test_dataset.normalize:
    #                 curl_pred = curl_pred * test_dataset.curl_std.to(device) + test_dataset.curl_mean.to(device)
    #                 curl_target = curl_target * test_dataset.curl_std.to(device) + test_dataset.curl_mean.to(device)
    #             for i in range(curl_pred.shape[0]):
    #                 pred = curl_pred[i,2,:,:,:]
    #                 target = curl_target[i,2,:,:,:]
    #                 pred_flat = pred.flatten()
    #                 target_flat = target.flatten()
    #                 non_zero_mask = torch.abs(target_flat) > 1e-6
    #                 n_non_zero = torch.sum(non_zero_mask).item()
    #                 n_zero = torch.sum(~non_zero_mask).item()
    #                 if n_non_zero > 0 and n_zero > 0:
    #                     relative_mre = torch.mean(torch.abs(pred_flat[non_zero_mask] - target_flat[non_zero_mask]) / torch.abs(target_flat[non_zero_mask]))
    #                     absolute_mre = torch.mean(torch.abs(pred_flat[~non_zero_mask]))
    #                 elif n_non_zero > 0:
    #                     relative_mre = torch.mean(torch.abs(pred_flat[non_zero_mask] - target_flat[non_zero_mask]) / torch.abs(target_flat[non_zero_mask]))
    #                     absolute_mre = 0.0
    #                 else:
    #                     relative_mre = 0.0
    #                     absolute_mre = torch.mean(torch.abs(pred_flat[~non_zero_mask]))
    #                 total_mre += (relative_mre + absolute_mre).item()
    #                 num_samples += 1
    #     test_mre = total_mre / num_samples

    #     history['train_loss'].append(train_loss)
    #     history['test_loss'].append(test_loss)
    #     history['test_mre'].append(test_mre)
    #     history['phase'].append('Adam')

    #     print(f"[Adam] Epoch [{epoch+1:4d}/{num_epochs_adam}],"
    #           f" Train Loss: {train_loss:.6f},"
    #           f" Test Loss: {test_loss:.6f},"
    #           f" Test MRE: {test_mre:.6f}")
        
    #     if test_loss < best_loss:
    #         best_loss = test_loss
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer_adam.state_dict(),
    #             'train_loss': train_loss,
    #             'test_loss': test_loss,
    #             'test_mre': test_mre,
    #         }, 'checkpoints/best_model_adam.pth')
    
    # # 绘制训练曲线
    # print("\n训练完成，绘制训练曲线...")
    # # plt.figure(figsize=(18,5))

    # 加载最佳模型并可视化
    print("加载最佳模型进行可视化...")
    checkpoint = torch.load('checkpoints/best_model_adam.pth')
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