clear; 
clc; 
close all;

% 数据集参数
n_samples = 1000;
train_ratio = 0.8;
n_train = floor(n_samples * train_ratio);
n_test = n_samples - n_train;

% 计算域参数
Nx = 32; Ny = 32; Nz = 32;
dx_range = [0.3e-3, 0.8e-3];
dy_range = dx_range;
dz_range = dx_range;

% 平面波参数
n_waves = 5;
k_range = [0, 1048];
E_amplitude_range = [0, 5];

%% 生成数据集
fprintf('开始生成数据集...\n');
fprintf('样本总数: %d\n', n_samples);
fprintf('训练集: %d, 测试集: %d\n', n_train, n_test);
fprintf('网格尺寸: %dx%dx%d\n', Nx, Ny, Nz);

% 预分配存储
E_data = zeros(Nx, Ny, Nz, 3, n_samples, 'single');
curl_data = zeros(Nx, Ny, Nz, 3, n_samples, 'single');
r_data = zeros(Nx, Ny, Nz, 3, n_samples, 'single');
dx_data = zeros(n_samples, 1, 'single');
dy_data = zeros(n_samples, 1, 'single');
dz_data = zeros(n_samples, 1, 'single');

% 进度显示
progress_interval = max(1, floor(n_samples/20));
start_time = tic;

%生成数据集
for sample_idx = 1:n_samples
    % 显示进度
    if mod(sample_idx, progress_interval) == 0
        elapsed = toc(start_time);
        fprintf('  进度: %d/%d (%.1f%%) 已用时间: %.1f秒\n', ...
                sample_idx, n_samples, 100*sample_idx/n_samples, elapsed);
    end
    
    % 随机计算域网格尺寸
    dx = dx_range(1) + (dx_range(2)-dx_range(1)) * rand();
    dy = dy_range(1) + (dy_range(2)-dy_range(1)) * rand();
    dz = dz_range(1) + (dz_range(2)-dz_range(1)) * rand();
    
    % 存储网格尺寸
    dx_data(sample_idx) = dx;
    dy_data(sample_idx) = dy;
    dz_data(sample_idx) = dz;
    
    % 生成空间坐标网格
    x = ((0:Nx-1) - (Nx-1)/2) * dx;
    y = ((0:Ny-1) - (Ny-1)/2) * dy;
    z = ((0:Nz-1) - (Nz-1)/2) * dz;
    [X, Y, Z] = meshgrid(x, y, z);
    X = permute(X, [2, 1, 3]);
    Y = permute(Y, [2, 1, 3]);
    Z = permute(Z, [2, 1, 3]);
    
    % 初始化电场为零
    Ex = zeros(Nx, Ny, Nz);
    Ey = zeros(Nx, Ny, Nz);
    Ez = zeros(Nx, Ny, Nz);
    
    % 初始化旋度为零
    curl_x = zeros(Nx, Ny, Nz);
    curl_y = zeros(Nx, Ny, Nz);
    curl_z = zeros(Nx, Ny, Nz);
    
    % 生成多个平面波并叠加
    for wave_idx = 1:n_waves
        % 随机生成角度
        phi = -pi + 2*pi*rand();
        theta = pi*rand();
        if cos(theta) < 10e-5
            theta = pi*rand();
        end
        
        % 随机生成波数
        k = k_range(1) + (k_range(2)-k_range(1)) * rand();
        
        % 随机生成Ex, Ey的振幅
        Ex0 = E_amplitude_range(1) + (E_amplitude_range(2)-E_amplitude_range(1)) * rand();
        Ey0 = E_amplitude_range(1) + (E_amplitude_range(2)-E_amplitude_range(1)) * rand();
        
        % 根据横波条件计算Ez的振幅
        k_x = sin(theta)*cos(phi);
        k_y = sin(theta)*sin(phi);
        k_z = cos(theta);
        Ez0 = -(k_x*Ex0 + k_y*Ey0) / k_z;
        
        % 创建波矢量E0
        E0 = [Ex0, Ey0, Ez0];
        
        % 生成相位因子网格
        phase = -1j * k * (X*sin(theta)*cos(phi) + ...
                          Y*sin(theta)*sin(phi) + ...
                          Z*cos(theta));
        
        % 生成该平面波的电场分量
        Ex_wave = E0(1) * exp(phase);
        Ey_wave = E0(2) * exp(phase);
        Ez_wave = E0(3) * exp(phase);
        
        % 计算该平面波的旋度 (解析解)
        curl_x_wave = -1j * k * (k_y * Ez_wave - k_z * Ey_wave);
        curl_y_wave = -1j * k * (k_z * Ex_wave - k_x * Ez_wave);
        curl_z_wave = -1j * k * (k_x * Ey_wave - k_y * Ex_wave);
        
        % 叠加到总电场和总旋度
        Ex = Ex + real(Ex_wave);
        Ey = Ey + real(Ey_wave);
        Ez = Ez + real(Ez_wave);
        
        curl_x = curl_x + real(curl_x_wave);
        curl_y = curl_y + real(curl_y_wave);
        curl_z = curl_z + real(curl_z_wave);
    end
    
    % 存储数据
    E_data(:, :, :, 1, sample_idx) = Ex;
    E_data(:, :, :, 2, sample_idx) = Ey;
    E_data(:, :, :, 3, sample_idx) = Ez;
    
    curl_data(:, :, :, 1, sample_idx) = curl_x;
    curl_data(:, :, :, 2, sample_idx) = curl_y;
    curl_data(:, :, :, 3, sample_idx) = curl_z;

    r_data(:,:,:,1,sample_idx) = X;
    r_data(:,:,:,2,sample_idx) = Y;
    r_data(:,:,:,3,sample_idx) = Z;
end

elapsed_time = toc(start_time);
fprintf('数据集生成完成! 总用时: %.1f秒\n', elapsed_time);

%% 划分训练集和测试集
fprintf('\n划分训练集和测试集...\n');

% 生成随机索引
rand_idx = randperm(n_samples);
train_idx = rand_idx(1:n_train);
test_idx = rand_idx(n_train+1:end);

% 分割数据
E_train = E_data(:, :, :, :, train_idx);
curl_train = curl_data(:, :, :, :, train_idx);
r_train = r_data(:, :, :, :, train_idx);

E_test = E_data(:, :, :, :, test_idx);
curl_test = curl_data(:, :, :, :, test_idx);
r_test = r_data(:, :, :, :, test_idx);

fprintf('训练集: %d个样本\n', n_train);
fprintf('测试集: %d个样本\n', n_test);

%% 可视化部分样本
fprintf('\n可视化部分样本...\n');

% 选择几个样本进行可视化
vis_samples = min(8, n_samples);
figure('Position', [100, 100, 1200, 800]);

for i = 1:vis_samples
    sample_id = i;
    
    % 电场Ex分量
    subplot(vis_samples, 4, (i-1)*4 + 1);
    Ex_slice = squeeze(E_data(:,:,Nz/2,1,sample_id));
    imagesc(Ex_slice);
    title(sprintf('样本%d: Ex', sample_id));
    colorbar;
    axis equal tight;
    
    % 电场Ey分量
    subplot(vis_samples, 4, (i-1)*4 + 2);
    Ey_slice = squeeze(E_data(:,:,Nz/2,2,sample_id));
    imagesc(Ey_slice);
    title(sprintf('样本%d: Ey', sample_id));
    colorbar;
    axis equal tight;
    
    % 旋度x分量
    subplot(vis_samples, 4, (i-1)*4 + 3);
    curlx_slice = squeeze(curl_data(:,:,Nz/2,1,sample_id));
    imagesc(curlx_slice);
    title(sprintf('样本%d: ∇×E_x', sample_id));
    colorbar;
    axis equal tight;
    
    % 旋度y分量
    subplot(vis_samples, 4, (i-1)*4 + 4);
    curly_slice = squeeze(curl_data(:,:,Nz/2,2,sample_id));
    imagesc(curly_slice);
    title(sprintf('样本%d: ∇×E_y', sample_id));
    colorbar;
    axis equal tight;
end

colormap jet;
sgtitle('电场和旋度数据示例 (z=中间切片)');

%% 数据统计信息
fprintf('\n数据统计信息:\n');
fprintf('电场数据尺寸: %d x %d x %d x %d x %d\n', size(E_data));
fprintf('旋度数据尺寸: %d x %d x %d x %d x %d\n', size(curl_data));
fprintf('坐标数据尺寸: %d x %d x %d x %d x %d\n', size(r_data))

% 计算统计量
E_all = E_data(:);
curl_all = curl_data(:);
r_all = r_data(:);

fprintf('电场统计:\n');
fprintf('  最小值: %.4f\n', min(E_all));
fprintf('  最大值: %.4f\n', max(E_all));
fprintf('  平均值: %.4f\n', mean(E_all));
fprintf('  标准差: %.4f\n', std(E_all));

fprintf('旋度统计:\n');
fprintf('  最小值: %.4f\n', min(curl_all));
fprintf('  最大值: %.4f\n', max(curl_all));
fprintf('  平均值: %.4f\n', mean(curl_all));
fprintf('  标准差: %.4f\n', std(curl_all));

fprintf('网格尺寸统计:\n');
fprintf('  dx: %.3f-%.3f mm\n', min(dx_data)*1000, max(dx_data)*1000);
fprintf('  dy: %.3f-%.3f mm\n', min(dy_data)*1000, max(dy_data)*1000);
fprintf('  dz: %.3f-%.3f mm\n', min(dz_data)*1000, max(dz_data)*1000);

%% 保存数据
fprintf('\n保存数据到文件...\n');

save_dir = 'DCO_data';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% 保存训练集
save(fullfile(save_dir, 'train_data.mat'), ...
     'E_train', 'curl_train', 'r_train', ...
     'Nx', 'Ny', 'Nz', 'n_train', '-mat');

% 保存测试集
save(fullfile(save_dir, 'test_data.mat'), ...
     'E_test', 'curl_test', 'r_test', ...
     'Nx', 'Ny', 'Nz', 'n_test', '-mat');

fprintf('数据已保存到文件夹: %s\n', save_dir);
fprintf('  训练集: train_data.mat\n');
fprintf('  测试集: test_data.mat\n');
fprintf('\n所有数据生成完成!\n');