---
title: 自定义CUDA算子融合实现模型推理加速
mathjax: true
toc: true
date: 2025-03-28 03:34:18
updated: 2025-03-28 03:34:18
categories:
- Machine Learning
tags:
- CUDA
---
对模型进行推理加速的最常用方法就是算子融合，这里用个简单demo记录下：

<!--more-->

总共有如下三个步骤：

## 导出模型权重

用pytorch定义一个多层DNN模型，然后导出其各层的网络参数。

```python
# export_model.py

import torch
import torch.nn as nn

# 定义PyTorch模型结构
class DNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(100, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.layer3(x)


# 创建模型并随机初始化
model = DNNModel()
model.eval()

# 导出权重为二进制文件
for name, param in model.named_parameters():
    param.detach().cpu().numpy().tofile(f"{name}.bin")

print("权重导出完成！")

```

运行 `python export_model.py` 。

## 编写CUDA融合算子

神经网络的每一层前向传播，都先从全局内存中读取tensor到寄存器内存，完成计算后再写回到全局内存，IO次数较多。利用算子融合，将多次计算融合成一次计算，减少IO读写，从而实现模型推理加速。

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_forward(
    const float* input,
    float* output,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* W3, const float* b3,
    int batch_size, int in_dim, int hid1, int hid2, int out_dim
) {
    // 每个线程处理一个样本的完整计算
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= batch_size) return;

    // 指向当前样本的输入和输出
    const float* x = input + sample_idx * in_dim;
    float* out = output + sample_idx * out_dim;

    // 第一层：Linear + ReLU
    float hidden1[128];
    for (int i = 0; i < hid1; ++i) {
        float sum = b1[i];
        for (int j = 0; j < in_dim; ++j) {
            sum += x[j] * W1[j * hid1 + i]; // 转置访问权重
        }
        hidden1[i] = fmaxf(sum, 0.0f);
    }

    // 第二层：Linear + ReLU
    float hidden2[64];
    for (int i = 0; i < hid2; ++i) {
        float sum = b2[i];
        for (int j = 0; j < hid1; ++j) {
            sum += hidden1[j] * W2[j * hid2 + i]; // 转置访问权重
        }
        hidden2[i] = fmaxf(sum, 0.0f);
    }

    // 第三层：Linear
    for (int i = 0; i < out_dim; ++i) {
        float sum = b3[i];
        for (int j = 0; j < hid2; ++j) {
            sum += hidden2[j] * W3[j * out_dim + i]; // 转置访问权重
        }
        out[i] = sum;
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3
) {
    int batch_size = input.size(0);
    int in_dim = W1.size(1);
    int hid1 = W1.size(0);
    int hid2 = W2.size(0);
    int out_dim = W3.size(0);

    torch::Tensor output = torch::zeros({batch_size, out_dim}, input.options());

    // 每个block处理多个样本，根据GPU配置调整
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    fused_forward<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        W1.data_ptr<float>(),
        b1.data_ptr<float>(),
        W2.data_ptr<float>(),
        b2.data_ptr<float>(),
        W3.data_ptr<float>(),
        b3.data_ptr<float>(),
        batch_size, in_dim, hid1, hid2, out_dim
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_forward_cuda, "Fused forward pass (CUDA)");
}
```