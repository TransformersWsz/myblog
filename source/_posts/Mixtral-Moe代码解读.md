---
title: Mixtral Moe代码解读
mathjax: true
toc: true
date: 2024-01-11 02:42:13
categories:
- NLP
tags:
- LLM
- Sparse MOE
---
一直对稀疏专家网络好奇，有些专家没被选中，那么梯度是否为0，这一轮被选中有梯度，下一轮没被选中无梯度，模型可以训练收敛吗？

<!--more-->

- 由于每个token都会选择topk个专家，所以在每一轮epoch中，所有专家都参与了前向传播，所以梯度都能得到更新
- 即使真有专家一直没被选中，那么其梯度保持不变，没有参与更新而已

```python
self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

# 获取到每个token的mlp层输入特征 
batch_size, sequence_length, hidden_dim = hidden_states.shape
hidden_states = hidden_states.view(-1, hidden_dim)

# 得到每个专家的打分，维度是batch * sequence, num_experts，取topk个专家
router_logits = self.gate(hidden_states)
routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

# 取到topk个专家的打分，需要计算在归一化一下，用于对后面的expert计算出来的结果进行加权
routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
# routing_weights、selected_experts 维度是一致的，取了topk   (bs * sl, topk)
routing_weights = routing_weights.to(hidden_states.dtype)

final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

# 如果不做后面的维度切换，那expert_mask的维度是 (bs*sl, topk, n_experts)，但是后面要遍历n_experts来计算，所以颠倒一下，得到(n_experts, topk, bs * sl); 
expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

for expert_idx in range(self.num_experts):
    expert_layer = self.experts[expert_idx]
    idx, top_x = torch.where(expert_mask[expert_idx])
    
    """
    这样取到expert_mask[expert_idx]，从上面的注释可以知道维度是
    [topk, bs * sl]；torch.where的结果，第一个结果代表选到了哪一行，第二个代表选择了哪一列
    
    对应到实际意义，top_x表示取的列，也就是取哪些token
    而行表示，取到的这些token，根据路由gate计算，当前expert是排行第几；
    所以这里变量名字可能有点混淆，
    """
    
    # 没有token需要当前的expert计算
    if top_x.shape[0] == 0:
        continue
    
    # tensor index使用list比tensor快
    top_x_list = top_x.tolist()
    idx_list = idx.tolist()

    # 前面hidden states已经转成了 [bs * sl, hs]，根据top_x 可以找到需要计算的token，这些token依旧是有序的
    current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
    
    # 找到这个expert对应的权重 乘进去
    # 上面计算的权重是routing_weights，维度是bs * sl, topk
    # 根据top_x_list 对应的token，idx_list表示topk中第几个
    # 可以直接取到相应的权重
    current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

    # 合到最终的特征里边去
    final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    
final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
```

___

## 参考
- [理解Mixtral Moe模型原理与代码实现](https://mp.weixin.qq.com/s/NNyyA7zJb5-Su5H353lovw)