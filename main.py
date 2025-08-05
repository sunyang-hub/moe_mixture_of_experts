import torch
from torch import nn
from mixture_of_experts import MoE

# 创建混合专家系统(Mixture of Experts, MoE)模型
# MoE是一种高效的神经网络架构，可以在不增加计算量的情况下增加模型参数
moe = MoE(
    dim = 512,                      # 输入和输出特征的维度
    num_experts = 16,               # 专家数量：增加专家数量可以增加模型参数而不增加计算量
    hidden_dim = 512 * 4,           # 每个专家的隐藏层维度，默认为输入维度的4倍(2048)
    activation = nn.LeakyReLU,      # 激活函数：使用LeakyReLU替代默认的GELU
    second_policy_train = 'random', # 训练时第二专家的选择策略：random表示随机选择
    second_policy_eval = 'random',  # 评估时第二专家的选择策略
    # 策略选项：'all'(总是使用) | 'none'(从不使用) | 'threshold'(如果门控值>阈值) | 'random'(随机选择)
    second_threshold_train = 0.2,   # 训练时第二专家的门控阈值
    second_threshold_eval = 0.2,    # 评估时第二专家的门控阈值
    capacity_factor_train = 1.25,   # 训练时的容量因子：确保有足够的容量处理不平衡的门控
    capacity_factor_eval = 2.,      # 评估时的容量因子：应该设置为>=1的值
    loss_coef = 1e-2                # 辅助损失系数：用于平衡专家使用的损失权重
)

# 创建随机输入张量用于测试
# 形状：(batch_size, sequence_length, feature_dimension)
inputs = torch.randn(4, 1024, 512)  # 批次大小=4, 序列长度=1024, 特征维度=512

# 前向传播：通过MoE模型处理输入
# 返回：输出张量和辅助损失
out, aux_loss = moe(inputs) # 输出形状：(4, 1024, 512)，辅助损失形状：(1,)

# 打印结果信息
print(f"输入形状: {inputs.shape}")
print(f"输出形状: {out.shape}")
print(f"辅助损失: {aux_loss.item():.6f}")
print(f"辅助损失用于平衡专家使用，防止某些专家被过度使用或忽略")