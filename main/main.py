import torch
import numpy as np
from deeponet import DeepONet3D
from data import DCO_dataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from plot import visualize_results, visualize_error_distribution

# def initialize_weights(model):
#     for name, module in model.named_modules():
#         if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
#             # GELU 的最佳初始化：Kaiming Normal with fan_in
#             # 使用 fan_in 模式可以更好地保持前向传播的方差
#             torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
#             if module.bias is not None:
#                 torch.nn.init.constant_(module.bias, 0)
        
#         elif isinstance(module, torch.nn.BatchNorm3d):
#             # BatchNorm 标准初始化
#             torch.nn.init.constant_(module.weight, 1)
#             torch.nn.init.constant_(module.bias, 0)
    
#     # 输出层使用小的初始化，让模型从接近0的输出开始学习
#     if hasattr(model, 'decoder') and hasattr(model.decoder, 'out_conv'):
#         torch.nn.init.xavier_normal_(model.decoder.out_conv.weight, gain=0.02)
#         if model.decoder.out_conv.bias is not None:
#             torch.nn.init.constant_(model.decoder.out_conv.bias, 0)
    
#     print("✓ 模型参数初始化完成 (针对 GELU 优化)")
#     print("  - 卷积层: Kaiming Normal (mode='fan_in', 适合GELU)")
#     print("  - BatchNorm: weight=1, bias=0")
#     print("  - 输出层: Xavier Normal (gain=0.02, 小初始化)\n")


def main():
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 32
    num_epochs_adam = 2000
    num_epochs_lbfgs = 50
    lr_adam = 0.0001
    base_features = 32

    print("加载数据集...")
    train_dataset = DCO_dataset(mode='train', normalize=True)
    test_dataset = DCO_dataset(mode='test', normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    print("初始化模型...")
    model = DeepONet3D(in_ch=3, out_ch=3, base_ch=base_features, num_layers=4).to(device)
    # initialize_weights(model)

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

    criterion = torch.nn.MSELoss()

    history = {
        'train_loss': [],
        'test_loss': [],
        'test_mre': [],
        'phase': []
    }
    best_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    print("开始训练Adam阶段...\n")
    optimizer_adam = torch.optim.AdamW(model.parameters(), lr=lr_adam)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=50, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_adam,
    mode='min',
    factor=0.5,
    patience=100,
    min_lr=1e-6
)

    for epoch in range(num_epochs_adam):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0
        for E, r, curl_target in train_loader:
            E, r, curl_target = E.to(device), r.to(device), curl_target.to(device)

            optimizer_adam.zero_grad()
            curl_pred = model(E, r)
            loss = criterion(curl_pred, curl_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer_adam.step()
            scheduler.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ===== 测试阶段 =====
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for E, r, curl_target in test_loader:
                E_mean = torch.mean(E, dim=(1,2,3,4), keepdim=True)
                E_std = torch.std(E, dim=(1,2,3,4), keepdim=True)
                r_mean = torch.mean(r, dim=(1,2,3,4), keepdim=True)
                r_std = torch.std(r, dim=(1,2,3,4), keepdim=True)
                E = (E - E_mean) / (E_std + 1e-7)
                r = (r - r_mean) / (r_std + 1e-7)
                curl_target = curl_target * (r_std + 1e-7) / (E_std + 1e-7)
                E, r, curl_target = E.to(device), r.to(device), curl_target.to(device)

                curl_pred = model(E, r)
                loss = criterion(curl_pred, curl_target)
                test_loss += loss.item()
            test_loss /= len(test_loader)

        # ===== 计算误差 =====
        total_mre = 0.0
        num_samples = 0
        with torch.no_grad():
            for E, r, curl_target in test_loader:
                E_mean = torch.mean(E, dim=(1,2,3,4), keepdim=True)
                E_std = torch.std(E, dim=(1,2,3,4), keepdim=True)
                r_mean = torch.mean(r, dim=(1,2,3,4), keepdim=True)
                r_std = torch.std(r, dim=(1,2,3,4), keepdim=True)
                E = (E - E_mean) / (E_std + 1e-7)
                r = (r - r_mean) / (r_std + 1e-7)

                E, r, curl_target = E.to(device), r.to(device), curl_target.to(device)

                curl_pred = model(E, r)

                if test_dataset.normalize:
                    curl_pred = curl_pred * (E_std.to(device) + 1e-7) / (r_std.to(device) + 1e-7)
                for i in range(curl_pred.shape[0]):
                    for channel in range(3):
                        pred = curl_pred[i,channel,:,:,:]
                        target = curl_target[i,channel,:,:,:]
                        pred_flat = pred.flatten()
                        target_flat = target.flatten()
                        non_zero_mask = torch.abs(target_flat) > 1e-2
                        n_non_zero = torch.sum(non_zero_mask).item()
                        n_zero = torch.sum(~non_zero_mask).item()
                        if n_non_zero > 0 and n_zero > 0:
                            relative_sum = torch.sum(torch.abs(pred_flat[non_zero_mask] - target_flat[non_zero_mask]) / torch.abs(target_flat[non_zero_mask]))
                            absolute_sum = torch.sum(torch.abs(pred_flat[~non_zero_mask]))
                            mre = (relative_sum + absolute_sum) / (n_non_zero + n_zero)
                        elif n_non_zero > 0:
                            mre = torch.mean(torch.abs(pred_flat[non_zero_mask] - target_flat[non_zero_mask]) / torch.abs(target_flat[non_zero_mask]))
                        else:
                            mre = torch.mean(torch.abs(pred_flat))
                        total_mre += mre.item()
                        num_samples += 1
        test_mre = total_mre / num_samples

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_mre'].append(test_mre)
        history['phase'].append('Adam')

        print(f"[Adam] Epoch [{epoch+1:4d}/{num_epochs_adam}],"
              f" Train Loss: {train_loss:.6f},"
              f" Test Loss: {test_loss:.6f},"
              f" Test MRE: {test_mre:.6f}")
        
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_adam.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_mre': test_mre,
            }, 'checkpoints/best_model_adam.pth')
    
    print("\n清理显存准备 L-BFGS 训练...")
    del optimizer_adam  # 删除 Adam 优化器
    torch.cuda.empty_cache()
    import gc
    gc.collect()


    print("开始训练L-BFGS阶段...\n")
    batch_size_lbfgs = 16
    train_loader_lbfgs = DataLoader(train_dataset, batch_size=batch_size_lbfgs, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    lr=1.0,               # L-BFGS 的学习率通常设为 1.0
    max_iter=20,          # 每次调用的最大迭代次数
    history_size=10,      # 存储的历史梯度数量
    line_search_fn='strong_wolfe'  # 使用强 Wolfe 条件线搜索
    )
    for epoch in range(num_epochs_lbfgs):
        model.train()
        train_loss = 0.0
        for E, r, curl_target in train_loader_lbfgs:
            E, r, curl_target = E.to(device), r.to(device), curl_target.to(device)
            def closure():
                optimizer_lbfgs.zero_grad()
                curl_pred = model(E, r)
                loss = criterion(curl_pred, curl_target)
                loss.backward()
                return loss
            loss = optimizer_lbfgs.step(closure)
            train_loss += loss.item()
        train_loss /= len(train_loader_lbfgs)

        # 评估
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for E, r, curl_target in test_loader:
                E_mean = torch.mean(E, dim=(1,2,3,4), keepdim=True)
                E_std = torch.std(E, dim=(1,2,3,4), keepdim=True)
                r_mean = torch.mean(r, dim=(1,2,3,4), keepdim=True)
                r_std = torch.std(r, dim=(1,2,3,4), keepdim=True)
                E = (E - E_mean) / (E_std + 1e-7)
                r = (r - r_mean) / (r_std + 1e-7)
                curl_target = curl_target * (r_std + 1e-7) / (E_std + 1e-7)
                E, r, curl_target = E.to(device), r.to(device), curl_target.to(device)
                curl_pred = model(E, r)
                test_loss += criterion(curl_pred, curl_target).item()
        test_loss /= len(test_loader)

        # ===== 计算误差 =====
        total_mre = 0.0
        num_samples = 0
        with torch.no_grad():
            for E, r, curl_target in test_loader:
                E_mean = torch.mean(E, dim=(1,2,3,4), keepdim=True)
                E_std = torch.std(E, dim=(1,2,3,4), keepdim=True)
                r_mean = torch.mean(r, dim=(1,2,3,4), keepdim=True)
                r_std = torch.std(r, dim=(1,2,3,4), keepdim=True)
                E = (E - E_mean) / (E_std + 1e-7)
                r = (r - r_mean) / (r_std + 1e-7)

                E, r, curl_target = E.to(device), r.to(device), curl_target.to(device)

                curl_pred = model(E, r)

                if test_dataset.normalize:
                    curl_pred = curl_pred * (E_std.to(device) + 1e-7) / (r_std.to(device) + 1e-7)
                for i in range(curl_pred.shape[0]):
                    for channel in range(3):
                        pred = curl_pred[i,channel,:,:,:]
                        target = curl_target[i,channel,:,:,:]
                        pred_flat = pred.flatten()
                        target_flat = target.flatten()
                        non_zero_mask = torch.abs(target_flat) > 1e-2
                        n_non_zero = torch.sum(non_zero_mask).item()
                        n_zero = torch.sum(~non_zero_mask).item()
                        if n_non_zero > 0 and n_zero > 0:
                            relative_sum = torch.sum(torch.abs(pred_flat[non_zero_mask] - target_flat[non_zero_mask]) / torch.abs(target_flat[non_zero_mask]))
                            absolute_sum = torch.sum(torch.abs(pred_flat[~non_zero_mask]))
                            mre = (relative_sum + absolute_sum) / (n_non_zero + n_zero)
                        elif n_non_zero > 0:
                            mre = torch.mean(torch.abs(pred_flat[non_zero_mask] - target_flat[non_zero_mask]) / torch.abs(target_flat[non_zero_mask]))
                        else:
                            mre = torch.mean(torch.abs(pred_flat))
                        total_mre += mre.item()
                        num_samples += 1
        test_mre = total_mre / num_samples
        
        print(f"[L-BFGS] Epoch [{epoch+1}/{num_epochs_lbfgs}], "
            f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Test MRE: {test_mre:.6f}")
        
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_lbfgs.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_mre': test_mre,
            }, 'checkpoints/best_model_lbfgs.pth')


    # 绘制训练曲线
    print("\n训练完成，绘制训练曲线...")
    # plt.figure(figsize=(18,5))

    # 加载最佳模型并可视化
    print("加载最佳模型进行可视化...")
    checkpoint = torch.load('checkpoints/best_model_lbfgs.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"最佳模型来自 Epoch {checkpoint['epoch']+1}")
    print(f"  Test Loss: {checkpoint['test_loss']:.6f}")
    print(f"  Test MRE: {checkpoint['test_mre']:.6f}\n")

    num_plot_samples = 4
    indices = torch.randperm(len(test_dataset))[:num_plot_samples]
    # 可视化预测结果
    visualize_results(model, test_dataset, device, num_samples=num_plot_samples, indices=indices)
    
    # 可视化误差分布
    visualize_error_distribution(model, test_dataset, device, num_samples=num_plot_samples, indices=indices)

    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()