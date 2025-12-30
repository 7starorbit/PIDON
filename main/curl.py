"""
@Author: Huang Qing
@Date: 2025-12-124 22:17:00
"""
import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from torch.utils.data import DataLoader, Dataset

np.random.seed(1234)
torch.manual_seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# =================== 数据集定义并归一化 ===================
class DCO_dataset(Dataset):
    def __init__(self, mode='train', data_dir='../DCO_data/', normalize=True):
        super(DCO_dataset, self).__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.normalize = normalize
        if self.mode == 'train':
            mat_data = sci.io.loadmat(os.path.join(self.data_dir, 'train_data.mat'))
            self.E = mat_data['E_train']    #(Nx, Ny, Nz, 3, n_samples)
            self.curl = mat_data['curl_train']
            self.r = mat_data['r_train']
        else:
            mat_data = sci.io.loadmat(os.path.join(self.data_dir, 'test_data.mat'))
            self.E = mat_data['E_test']
            self.curl = mat_data['curl_test']
            self.r = mat_data['r_test']
        self.E = torch.from_numpy(self.E).permute(4, 3, 0, 1, 2).float()  #(n_samples, channels(3), Nx, Ny, Nz)
        self.curl = torch.from_numpy(self.curl).permute(4, 3, 0, 1, 2).float()
        self.r = torch.from_numpy(self.r).permute(4, 3, 0, 1, 2).float()
        self.n_samples = self.E.shape[0]
        print(f"{mode} 数据集: {self.n_samples} 个样本")
        print(f"  E shape: {self.E.shape}")
        print(f"  curl shape: {self.curl.shape}")
        print(f"  r shape: {self.r.shape}")
        if self.normalize is True :
            if self.mode == 'train':
                self.E_mean = torch.mean(self.E, dim=(0,2,3,4), keepdim=True)
                self.E_std = torch.std(self.E, dim=(0,2,3,4), keepdim=True)
                self.curl_mean = torch.mean(self.curl, dim=(0,2,3,4), keepdim=True)
                self.curl_std = torch.std(self.curl, dim=(0,2,3,4), keepdim=True)
                self.r_mean = torch.mean(self.r, dim=(0,2,3,4), keepdim=True)
                self.r_std = torch.std(self.r, dim=(0,2,3,4), keepdim=True)
                self.E = (self.E - self.E_mean) / (self.E_std + 1e-8)
                self.curl = (self.curl - self.curl_mean) / (self.curl_std + 1e-8)
                self.r = (self.r - self.r_mean) / (self.r_std + 1e-8)
                torch.save({
                    'E_mean': self.E_mean,
                    'E_std': self.E_std,
                    'curl_mean': self.curl_mean,
                    'curl_std': self.curl_std,
                    'r_mean': self.r_mean,
                    'r_std': self.r_std
                }, os.path.join('./', 'norm_stats.pth'))
            else:
                norm_stats = torch.load(os.path.join('./', 'norm_stats.pth'))
                self.E_mean = norm_stats['E_mean']
                self.E_std = norm_stats['E_std']
                self.curl_mean = norm_stats['curl_mean']
                self.curl_std = norm_stats['curl_std']
                self.r_mean = norm_stats['r_mean']
                self.r_std = norm_stats['r_std']
                self.E = (self.E - self.E_mean) / (self.E_std + 1e-8)
                self.curl = (self.curl - self.curl_mean) / (self.curl_std + 1e-8)
                self.r = (self.r - self.r_mean) / (self.r_std + 1e-8)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 返回: 电场E(3通道), 旋度curl(3通道), 坐标r(3通道)
        return self.E[idx], self.r[idx], self.curl[idx]

# =================== 3D卷积基础模块 ===================
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

# =================== 3D双卷积模块（带残差连接）===================
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv3D, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = ConvBlock3D(in_channels, mid_channels)
        self.conv2 = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # 最终激活函数（在残差连接之后）
        self.gelu = nn.GELU()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out = out + identity
        out = self.gelu(out)
        
        return out

# =================== 3D下采样模块 ===================
class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down3D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# =================== Branch Network(电场分支) ===================
class BranchNetwork(nn.Module):
    """
    输入: E (batch, 3, Nx, Ny, Nz)
    输出: 多层级特征 [f1, f2, f3, f4, f5]
    """    
    def __init__(self, in_channels=3, base_features=32):
        super(BranchNetwork, self).__init__()
        
        self.inc = DoubleConv3D(in_channels, base_features) # 3 -> 32
        self.down1 = Down3D(base_features, base_features * 2) # 32 -> 64
        self.down2 = Down3D(base_features * 2, base_features * 4) # 64 -> 128
        self.down3 = Down3D(base_features * 4, base_features * 8) # 128 -> 256
        self.down4 = Down3D(base_features * 8, base_features * 16) # 256 -> 512
        
    def forward(self, x):
        x1 = self.inc(x)      # 3 -> 32 32*32*32
        x2 = self.down1(x1)   # 16*16*16 32 -> 64
        x3 = self.down2(x2)   # 8*8*8 64 -> 128
        x4 = self.down3(x3)   # 4*4*4 128 -> 256
        x5 = self.down4(x4)   # 2*2*2 256 -> 512 (瓶颈层)
        
        return [x1, x2, x3, x4, x5]

# ==================== Trunk Network (坐标分支) ====================
class TrunkNetwork(nn.Module):
    """
    输入: r (batch, 3, Nx, Ny, Nz)
    输出: 多层级特征 [f1, f2, f3, f4, f5]
    """
    def __init__(self, in_channels=3, base_features=32):
        super(TrunkNetwork, self).__init__()
        
        self.inc = DoubleConv3D(in_channels, base_features)
        self.down1 = Down3D(base_features, base_features * 2)
        self.down2 = Down3D(base_features * 2, base_features * 4)
        self.down3 = Down3D(base_features * 4, base_features * 8)
        self.down4 = Down3D(base_features * 8, base_features * 16)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        return [x1, x2, x3, x4, x5]

# ==================== Decoder with Skip Connections ====================
class DecoderBlock(nn.Module):
    """
    解码器块: 上采样 -> 跳跃连接拼接 -> 独立解码卷积
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.decode_conv = DoubleConv3D(skip_channels + in_channels // 2, out_channels)

    def forward(self, x:torch.Tensor, skip_feature):
        # 上采样
        x = self.up(x)
        # 跳跃连接拼接
        x = torch.cat([skip_feature, x], dim=1)
        # 解码卷积 (stride=1)
        x = self.decode_conv(x)
        return x

class DeepONet_UNet3D(nn.Module):
    """
    架构流程:
    1. Branch Network (E) 和 Trunk Network (r) 分别进行下采样编码
    2. 每个层级的特征通过 Hadamard 乘积融合
    3. 融合后的特征作为跳跃连接传递到解码器
    4. 解码器包含独立的上采样和解码卷积层
    5. 最终输出旋度场
    
    卷积参数设置:
    - kernel_size=3, stride=1, padding=1
    - 激活函数: GELU
    - 下采样: MaxPool stride=2
    - 上采样: Upsample scale=2 或 ConvTranspose stride=2
    """
    def __init__(self, in_channels_branch=3, in_channels_trunk=3, out_channels=3, base_features=32):
        super(DeepONet_UNet3D, self).__init__()
        
        # Branch Network: 处理电场 E
        self.branch_net = BranchNetwork(in_channels_branch, base_features)
        
        # Trunk Network: 处理坐标 r
        self.trunk_net = TrunkNetwork(in_channels_trunk, base_features)
        
        # 解码器路径 (上采样 + 独立解码卷积)
        self.decoder1 = DecoderBlock(base_features * 16, base_features * 8, base_features * 8)  #512/2 + 256 -> 256
        self.decoder2 = DecoderBlock(base_features * 8, base_features * 4, base_features * 4)   #256/2 + 128 -> 128
        self.decoder3 = DecoderBlock(base_features * 4, base_features * 2, base_features * 2)   #128/2 + 64 -> 64
        self.decoder4 = DecoderBlock(base_features * 2, base_features, base_features)   #64/2 + 32 -> 32
        
        # 输出卷积层 (1x1卷积，stride=1)
        self.outc = nn.Conv3d(base_features, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, E, r):
        """
        Args:
            E: 电场 (batch, 3, Nx, Ny, Nz)
            r: 坐标 (batch, 3, Nx, Ny, Nz)
        Returns:
            curl: 旋度 (batch, 3, Nx, Ny, Nz)
        """
        branch_features = self.branch_net(E)  # [f1_b, f2_b, f3_b, f4_b, f5_b]
        trunk_features = self.trunk_net(r)    # [f1_t, f2_t, f3_t, f4_t, f5_t]
        
        # Hadamard 乘积融合 (元素相乘)
        fused_features = []
        for bf, tf in zip(branch_features, trunk_features):
            fused = bf * tf  # Hadamard product: 逐元素相乘
            fused_features.append(fused)
        
        # 解码器路径 (上采样 + 跳跃连接 + 解码卷积)
        x = self.decoder1(fused_features[4], fused_features[3])
        x = self.decoder2(x, fused_features[2])
        x = self.decoder3(x, fused_features[1])
        x = self.decoder4(x, fused_features[0])
        
        # 输出层
        output = self.outc(x)  # (batch, 3, Nx, Ny, Nz)
        return output

# class ChannelNormalizedMSELoss(nn.Module):
#     """对每个通道的MSE损失进行归一化"""
#     def __init__(self, epsilon=1e-8):
#         super(ChannelNormalizedMSELoss, self).__init__()
#         self.epsilon = epsilon
    
#     def forward(self, pred, target):
#         # pred, target: [batch, 3, Nx, Ny, Nz]
#         num_channels = pred.shape[1]
#         total_loss = 0.0
        
#         for c in range(num_channels):
#             # 提取当前通道
#             pred_c = pred[:, c]
#             target_c = target[:, c]
            
#             # 计算该通道的MSE
#             mse_c = torch.mean((pred_c - target_c) ** 2)
            
#             # 用该通道目标值的方差归一化
#             target_var = torch.var(target_c) + self.epsilon
#             normalized_mse = mse_c / target_var
            
#             total_loss += normalized_mse

#         return total_loss / num_channels

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (E, r, curl_target) in enumerate(dataloader):
        E = E.to(device)
        r = r.to(device)
        curl_target = curl_target.to(device)
        
        optimizer.zero_grad()
        curl_pred = model(E, r)
        loss = criterion(curl_pred, curl_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (E, r, curl_target) in enumerate(dataloader):
            E = E.to(device)
            r = r.to(device)
            curl_target = curl_target.to(device)
            
            # 前向传播
            curl_pred = model(E, r)
            
            # 计算损失
            loss = criterion(curl_pred, curl_target)
            
            # 累加损失
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def compute_MRE(model, dataloader, device, epsilon=1e-10):
    model.eval()
    total_mre = 0.0
    total_mre0 = 0.0
    total_mre1 = 0.0
    total_mre2 = 0.0
    num_samples = 0
    # curl_mean = dataloader.dataset.curl_mean.to(device)
    # curl_std = dataloader.dataset.curl_std.to(device)
    
    with torch.no_grad():
        for E, r, curl_target in dataloader:
            E = E.to(device)
            r = r.to(device)
            curl_target = curl_target.to(device)
            
            curl_pred = model(E, r)

            # # 反归一化
            # curl_pred = curl_pred * (curl_std + 1e-8) + curl_mean
            # curl_target = curl_target * (curl_std + 1e-8) + curl_mean
            
            # 计算每个样本的MRE
            batch_size = curl_pred.shape[0]
            for i in range(batch_size):
                pred = curl_pred[i]
                target = curl_target[i]
                
                pred_flat0 =pred[0].flatten()
                target_flat0 = target[0].flatten()
                none_zero_mask0 = torch.abs(target_flat0) > epsilon
                relative_error_non_zero0 = torch.abs(pred_flat0[none_zero_mask0] - target_flat0[none_zero_mask0]) / torch.abs(target_flat0[none_zero_mask0])
                absolute_error_zero0 = torch.abs(pred_flat0[~none_zero_mask0])
                all_errors0 = torch.cat([relative_error_non_zero0, absolute_error_zero0])
                mre_sample0 = torch.mean(all_errors0)
                total_mre0 += mre_sample0.item()
                pred_flat1 =pred[1].flatten()
                target_flat1 = target[1].flatten()
                none_zero_mask1 = torch.abs(target_flat1) > epsilon
                relative_error_non_zero1 = torch.abs(pred_flat1[none_zero_mask1] - target_flat1[none_zero_mask1]) / torch.abs(target_flat1[none_zero_mask1])
                absolute_error_zero1 = torch.abs(pred_flat1[~none_zero_mask1])
                all_errors1 = torch.cat([relative_error_non_zero1, absolute_error_zero1])
                mre_sample1 = torch.mean(all_errors1)
                total_mre1 += mre_sample1.item()
                pred_flat2 =pred[2].flatten()
                target_flat2 = target[2].flatten()
                none_zero_mask2 = torch.abs(target_flat2) > epsilon
                relative_error_non_zero2 = torch.abs(pred_flat2[none_zero_mask2] - target_flat2[none_zero_mask2]) / torch.abs(target_flat2[none_zero_mask2])
                absolute_error_zero2 = torch.abs(pred_flat2[~none_zero_mask2])
                all_errors2 = torch.cat([relative_error_non_zero2, absolute_error_zero2])
                mre_sample2 = torch.mean(all_errors2)
                total_mre2 += mre_sample2.item()
                num_samples += 1
                # pred_flat = pred.flatten()
                # target_flat = target.flatten()
                
                # non_zero_mask = torch.abs(target_flat) > epsilon
                
                # relative_error_non_zero = torch.abs(pred_flat[non_zero_mask] - target_flat[non_zero_mask]) / torch.abs(target_flat[non_zero_mask])
                
                # absolute_error_zero = torch.abs(pred_flat[~non_zero_mask])
                
                # all_errors = torch.cat([relative_error_non_zero, absolute_error_zero])
                # mre_sample = torch.mean(all_errors)
                
                # total_mre += mre_sample.item()
                # num_samples += 1
    return total_mre0 / num_samples, total_mre1 / num_samples, total_mre2 / num_samples
    # return total_mre / num_samples

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

def main():
    batch_size = 30
    num_epochs_adam = 200
    num_epochs_lbfgs = 100
    learning_rate_adam = 1e-3
    learning_rate_lbfgs = 0.1
    base_features = 32

    # 创建数据集和数据加载器
    print("加载数据集...")
    train_dataset = DCO_dataset(mode='train', normalize=True)
    test_dataset = DCO_dataset(mode='test', normalize=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                             num_workers=0, pin_memory=True)
    # 创建模型
    print("\n创建模型...")
    model = DeepONet_UNet3D(in_channels_branch=3, in_channels_trunk=3, out_channels=3, base_features=base_features).to(device)
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    branch_params = sum(p.numel() for p in model.branch_net.parameters())
    trunk_params = sum(p.numel() for p in model.trunk_net.parameters())
    decoder_params = total_params - branch_params - trunk_params
    
    print(f"\n{'='*70}")
    print(f"{'DeepONet 3D-UNet 旋度算子学习器':^70}")
    print(f"{'='*70}")
    print(f"  - Branch Network (电场编码器): {branch_params/1e6:.2f}M 参数")
    print(f"  - Trunk Network (坐标编码器):  {trunk_params/1e6:.2f}M 参数")
    print(f"  - Decoder (解码器):             {decoder_params/1e6:.2f}M 参数")
    print(f"  总参数量: {total_params/1e6:.2f}M")
    print(f"\n  训练策略: 二阶段优化 (Adam预热 + L-BFGS精调)")
    print(f"  阶段1: Adam优化器, {num_epochs_adam} epochs, lr={learning_rate_adam}")
    print(f"  阶段2: L-BFGS优化器, {num_epochs_lbfgs} epochs, lr={learning_rate_lbfgs}")
    print(f"\n  卷积设置: kernel_size=3, stride=1, padding=1")
    print(f"  特征融合: Hadamard Product (逐元素相乘)")
    print(f"  跳跃连接: 融合特征传递到解码器各层")
    print(f"  激活函数: GELU")
    print(f"  基础特征数: {base_features}")
    print(f"  Batch Size: {batch_size}")
    print(f"  误差指标: MRE (Mean Relative Error)")
    print(f"{'='*70}\n")

    # 损失函数和优化器
    criterion = nn.MSELoss()

    # 训练历史
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_mre': [],
        'phase': []
    }
    best_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    # 训练循环
    print("开始训练Adam阶段...\n")
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=learning_rate_adam)
    for epoch in range(2):
        
        train_loss = train_epoch(model, train_loader, optimizer_adam, criterion, device)
        test_loss = validate(model, test_loader, criterion, device)
        # test_mre = compute_MRE(model, test_loader, device)
        test_mre0, test_mre1, test_mre2 = compute_MRE(model, test_loader, device)
        test_mre = (test_mre0 + test_mre1 + test_mre2) / 3.0
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_mre'].append(test_mre)
        history['phase'].append('Adam')

        print(f"[Adam] Epoch [{epoch+1:4d}/{num_epochs_adam}],"
              f" Train Loss: {train_loss:.6f},"
              f" Test Loss: {test_loss:.6f},"
              f" Test MRE0: {test_mre0:.6f},"
              f" Test MRE1: {test_mre1:.6f},"
              f" Test MRE2: {test_mre2:.6f},"
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
            # print(f"  ✓ 保存最佳模型 (Test Loss: {best_loss:.6f}, MRE: {test_mre:.6f})")
    
    print("\n开始训练L-BFGS阶段...\n")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=learning_rate_lbfgs,
        max_iter=20,           # 每次调用最多20次线搜索
        max_eval=25,           # 最多25次函数评估
        tolerance_grad=1e-7,   # 梯度容差
        tolerance_change=1e-9, # 参数变化容差
        history_size=50,      # 存储50个历史信息用于Hessian近似
        line_search_fn='strong_wolfe'  # 使用强Wolfe条件线搜索
    )
    def closure():
        optimizer_lbfgs.zero_grad()
        total_loss = 0.0
        for E, r, curl_target in train_loader:
            E = E.to(device)
            r = r.to(device)
            curl_target = curl_target.to(device)
            curl_pred = model(E, r)
            loss = criterion(curl_pred, curl_target)
            loss.backward()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss
    for epoch in range(num_epochs_lbfgs):
        model.train()
        train_loss = optimizer_lbfgs.step(closure)
        test_loss = validate(model, test_loader, criterion, device)
        test_mre0, test_mre1, test_mre2 = compute_MRE(model, test_loader, device)
        test_mre = (test_mre0 + test_mre1 + test_mre2) / 3.0
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_mre'].append(test_mre)
        history['phase'].append('LBFGS')
        print(f"[L-BFGS] Epoch [{epoch+1:4d}/{num_epochs_lbfgs}],"
              f" Train Loss: {train_loss:.6f},"
              f" Test Loss: {test_loss:.6f},"
              f" Test MRE0: {test_mre0:.6f},"
              f" Test MRE1: {test_mre1:.6f},"
              f" Test MRE2: {test_mre2:.6f},"
              f" Test MRE: {test_mre:.6f}")
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch + num_epochs_adam,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_lbfgs.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_mre': test_mre,
            }, 'checkpoints/best_model.pth')
            # print(f"  ✓ 保存最佳模型 (Test Loss: {best_loss:.6f}, MRE: {test_mre:.6f})")

    # 绘制训练曲线
        print("绘制训练曲线...")
    plt.figure(figsize=(18, 5))
    
    adam_epochs = num_epochs_adam
    total_epochs = num_epochs_adam + num_epochs_lbfgs
    
    # 子图1: Loss曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['test_loss'], label='Test Loss', linewidth=2)
    plt.axvline(x=adam_epochs, color='red', linestyle='--', label='Adam→L-BFGS', linewidth=1.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.title('Training Progress (Two-Stage)', fontsize=12)
    
    # 子图2: Log-scale Loss
    plt.subplot(1, 3, 2)
    plt.semilogy(history['train_loss'], label='Train Loss', linewidth=2)
    plt.semilogy(history['test_loss'], label='Test Loss', linewidth=2)
    plt.axvline(x=adam_epochs, color='red', linestyle='--', label='Adam→L-BFGS', linewidth=1.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss (log scale)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.title('Loss (Log Scale)', fontsize=12)
    
    # 子图3: MRE曲线
    plt.subplot(1, 3, 3)
    plt.plot(history['test_mre'], linewidth=2, color='green')
    plt.axvline(x=adam_epochs, color='red', linestyle='--', label='Adam→L-BFGS', linewidth=1.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Relative Error', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.title('Test MRE', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('checkpoints/training_curves_two_stage.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 加载最佳模型并可视化
    print("\n加载最佳模型进行可视化...")
    if os.path.exists('checkpoints/best_model_lbfgs.pth'):
        checkpoint = torch.load('checkpoints/best_model_lbfgs.pth')
        print(f"加载L-BFGS阶段最佳模型 (Epoch {checkpoint['epoch']+1})")
    else:
        checkpoint = torch.load('checkpoints/best_model_adam.pth')
        print(f"加载Adam阶段最佳模型 (Epoch {checkpoint['epoch']+1})")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"最佳模型来自 Epoch {checkpoint['epoch']+1}")
    print(f"  Test Loss: {checkpoint['test_loss']:.6f}")
    print(f"  Test MRE: {checkpoint['test_mre']:.6f}\n")

    # 可视化预测结果
    visualize_results(model, test_dataset, device, num_samples=4)
    
    # 可视化误差分布
    visualize_error_distribution(model, test_dataset, device, num_samples=4)

    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
