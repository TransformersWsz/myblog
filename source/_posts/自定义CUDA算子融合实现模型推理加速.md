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

具体的代码包括如下三个步骤：
1. 加载pytorch导出的模型参数
2. 将多次前向传播融合到一个函数中
3. 将优化后的函数绑定到python模块中

```cpp
// fused_op.cu

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

上述代码用到了`libtorch`库，不过我们不需要手动安装，只要本地有pytorch库就可以。在绑定python模块的时候，pytorch会自动将其转换。

```python
# setup.py

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='fused_op',
    ext_modules=[
        CUDAExtension(
            name='fused_op',
            sources=['fused_op.cu']  # 根据GPU架构调整
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```
运行 `python setup.py install`

## 测试加速效果

主要为了验证模型推理耗时和结果一致性。

```python
# test.py

import torch
import numpy as np
import fused_op  # 导入CUDA模块
import time


# 加载原始PyTorch模型
class DNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(100, 128)
        self.relu1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(128, 64)
        self.relu2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.layer3(x)

origin_model = DNNModel()

# 加载原始模型的网络参数
def load_transposed_weights(filename, original_shape):
    # 从文件加载并重塑为转置后的形状
    transposed_weights = np.fromfile(filename, dtype=np.float32)
    transposed_shape = (original_shape[1], original_shape[0])  # 交换维度
    return torch.from_numpy(
        transposed_weights.reshape(transposed_shape).T  # 转置回原始形状
    )

origin_model.layer1.weight.data = load_transposed_weights("layer1.weight.bin", (128, 100))
origin_model.layer2.weight.data = load_transposed_weights("layer2.weight.bin", (64, 128))
origin_model.layer3.weight.data = load_transposed_weights("layer3.weight.bin", (10, 64))
origin_model.layer1.bias.data = torch.from_numpy(np.fromfile("layer1.bias.bin", dtype=np.float32))
origin_model.layer2.bias.data = torch.from_numpy(np.fromfile("layer2.bias.bin", dtype=np.float32))
origin_model.layer3.bias.data = torch.from_numpy(np.fromfile("layer3.bias.bin", dtype=np.float32))

origin_model = origin_model.cuda()

# 测试函数（返回时间和结果）
def measure_time(func, input_data, repeats=30):
    timings = []
    outputs = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.time()
            output = func(input_data)
            end = time.time()
            timings.append(end - start)
            outputs.append(output)
    return np.mean(timings), outputs[0]  # 返回平均时间和单次结果


# 准备输入数据并固定随机种子
# np.random.seed(42)
# torch.manual_seed(42)
input_data = torch.randn(32, 100, dtype=torch.float32).cuda()

# 加载优化后的模型权重，并将其作为参数传入
layer1_weight = origin_model.layer1.weight
layer2_weight = origin_model.layer2.weight
layer3_weight = origin_model.layer3.weight
layer1_bias = origin_model.layer1.bias
layer2_bias = origin_model.layer2.bias
layer3_bias = origin_model.layer3.bias

# 测试原始模型
origin_time, origin_output = measure_time(lambda x: origin_model(x), input_data)
print(f"原始PyTorch推理时间: {origin_time * 1000:.2f} ms")

# 测试CUDA融合模块
optimised_time, optimised_output = measure_time(
    lambda x: fused_op.fused_forward(x, layer1_weight, layer1_bias, layer2_weight, layer2_bias, layer3_weight, layer3_bias), input_data
)
print(f"CUDA融合推理时间: {optimised_time * 1000:.2f} ms")

# 结果对比
native_result = origin_output.cpu().numpy()
cuda_result = optimised_output.cpu().numpy()


abs_diff = np.abs(native_result - cuda_result)
max_abs_diff = np.max(abs_diff)
rel_diff = np.mean(abs_diff / (np.abs(native_result) + 1e-8))

print(f"最大绝对误差: {max_abs_diff:.6e}")
print(f"平均相对误差: {rel_diff:.6e}")

if max_abs_diff < 1e-5:
    print("✅ 结果一致，优化成功！")
else:
    print("❌ 结果不一致，可能存在错误！")

```

运行 `python test.py`：

![result](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.3goknytxty.webp)

___

- [op_fuse](https://github.com/TransformersWsz/cuda_examples/blob/main/op_fuse/README.md)