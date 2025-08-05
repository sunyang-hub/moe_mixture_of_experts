import torch
from torch import nn
import torch.nn.functional as F

import math
from inspect import isfunction

# constants

MIN_EXPERT_CAPACITY = 4

# helper functions

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# tensor related helper functions

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

# activations

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# expert class

class Experts(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = GELU):
        super().__init__()

        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)  #x.shape [16, 320, 512] self.w1.shape [16, 512, 2048]
        hidden = self.act(hidden)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out

# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network

class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x, importance = None):
        """
        Top2Gating的前向传播函数
        实现Top-2门控机制，为每个输入位置选择前两个最相关的专家
        
        Args:
            x: 输入张量，形状为 [..., batch, group_size, dim]
            importance: 可选的重要性权重，用于控制哪些输入应该被优先处理
            
        Returns:
            dispatch_tensor: 布尔张量，指示输入如何分配给专家
            combine_tensor: 组合张量，包含专家分配和权重信息
            loss: 负载平衡损失
        """
        # 解包输入张量的形状：batch_size, group_size, dimension
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        # 根据训练/评估模式选择不同的策略参数
        if self.training:
            policy = self.second_policy_train #第二份 专家是否被选用
            threshold = self.second_threshold_train  # 是否选择的阈值
            capacity_factor = self.capacity_factor_train # 容量因子，每个专家最多能接受多少token
        else:
            policy = self.second_policy_eval # 第二份 专家是否被选用
            threshold = self.second_threshold_eval # 是否选择的阈值
            capacity_factor = self.capacity_factor_eval

        # 计算原始门控分数：使用爱因斯坦求和计算 x 与门控权重的乘积
        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating) # 门控分数
        # 通过softmax归一化得到概率分布
        raw_gates = raw_gates.softmax(dim=-1)  #用于选择 哪个门

        # ========== 为每个位置选择前两个专家 ==========
        # 找到每个位置的第一选择专家
        gate_1, index_1 = top1(raw_gates)  # gate_1: 第一专家的权重, index_1: 第一专家的索引
        mask_1 = F.one_hot(index_1, num_gates).float()  # 创建第一专家的one-hot掩码
        density_1_proxy = raw_gates  # 用于计算负载平衡的代理密度

        # 为 Hierarchical MoE  提供帮助
        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]  # 只保留重要性为1的位置
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
            del equals_one_mask

        # 排除第一专家后，选择第二专家
        gates_without_top_1 = raw_gates * (1. - mask_1)  # 将第一专家的分数置零
        gate_2, index_2 = top1(gates_without_top_1)  # 找到第二专家
        mask_2 = F.one_hot(index_2, num_gates).float()  # 创建第二专家的one-hot掩码

        # 如果提供了重要性权重，只保留重要性大于0的位置
        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]
            del greater_zero_mask

        # 归一化Top-2门控分数，确保两个专家的权重和为1
        denom = gate_1 + gate_2 + self.eps  # 添加eps避免除零  
        gate_1 /= denom
        gate_2 /= denom

        # ========== 计算负载平衡损失 ==========
        # 计算每个专家被分配到的输入比例
        density_1 = mask_1.mean(dim=-2)  # [batch, experts] - 每个专家被选中的平均概率
        # 计算代理密度，用于负载平衡损失
        density_1_proxy = density_1_proxy.mean(dim=-2)
        # 负载平衡损失：鼓励专家负载均匀分布
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        # ========== 应用第二专家策略 ==========
        # 根据策略决定是否使用第二专家
        if policy == "all":
            pass  # 使用所有第二专家
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)  # 不使用第二专家
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()  # 只使用权重超过阈值的第二专家
        elif policy == "random":
            # 随机策略：根据第二专家权重概率性地使用
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        # ========== 计算专家容量 ==========
        # 计算每个专家能处理的最大输入数量
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)  # 确保最小容量
        expert_capacity_f = float(expert_capacity)

        # ========== 计算输入到专家的分配 ==========
        # 计算每个输入在专家中的位置（累积和）
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1  #判断 是否超过容量
        # 移除超出容量的输入
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # 计算每个专家被分配到的输入数量
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # 扁平化掩码：mostly ones, but zeros where something didn't fit
        mask_1_flat = mask_1.sum(dim=-1)
        # 计算第一专家中的位置
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # 更新第一专家的权重
        gate_1 *= mask_1_flat

        # 计算第二专家中的位置（需要考虑第一专家的占用）
        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        # 移除超出容量的输入
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        # 更新第二专家的权重
        gate_2 *= mask_2_flat
        
        # ========== 构建组合张量 ==========
        # 构建最终的组合张量，包含专家分配和权重信息
        # 形状: [batch, group, experts, expert_capacity]
        combine_tensor = (
            # 第一专家的贡献
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
            # 第二专家的贡献
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        # 创建布尔调度张量
        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss

# plain mixture of experts

class MoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        self.num_experts = num_experts

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, e = *inputs.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)  #expert_outputs.shape torch.Size([16, 320, 512])

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output, loss * self.loss_coef

# 2-level heirarchical mixture of experts

class HeirarchicalMoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = (4, 4),
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        assert len(num_experts) == 2, 'only 2 levels of heirarchy for experts allowed for now'
        num_experts_outer, num_experts_inner = num_experts
        self.num_experts_outer = num_experts_outer
        self.num_experts_inner = num_experts_inner

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}

        self.gate_outer = Top2Gating(dim, num_gates = num_experts_outer, **gating_kwargs)
        self.gate_inner = Top2Gating(dim, num_gates = num_experts_inner, outer_expert_dims = (num_experts_outer,), **gating_kwargs)

        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, eo, ei = *inputs.shape, self.num_experts_outer, self.num_experts_inner
        dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(inputs)
        expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor_outer)

        # we construct an "importance" Tensor for the inputs to the second-level
        # gating.  The importance of an input is 1.0 if it represents the
        # first-choice expert-group and 0.5 if it represents the second-choice expert
        # group.  This is used by the second-level gating.
        importance = combine_tensor_outer.permute(2, 0, 3, 1).sum(dim=-1)
        importance = 0.5 * ((importance > 0.5).float() + (importance > 0.).float())

        dispatch_tensor_inner, combine_tensor_inner, loss_inner = self.gate_inner(expert_inputs_outer, importance = importance)
        expert_inputs = torch.einsum('ebnd,ebnfc->efbcd', expert_inputs_outer, dispatch_tensor_inner)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(eo, ei, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        # NOW COMBINE EXPERT OUTPUTS (reversing everything we have done)
        # expert_output has shape [y0, x1, h, d, n]

        expert_outputs_outer = torch.einsum('efbcd,ebnfc->ebnd', expert_outputs, combine_tensor_inner)
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs_outer, combine_tensor_outer)
        return output, (loss_outer + loss_inner) * self.loss_coef
