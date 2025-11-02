# HSTU Training Code Refactoring Summary - A100 Only Version

## 重构目标
本次重构旨在简化HSTU训练代码，**只保留A100 GPU (SM 8.0) 的CUDA加速版本**，删除所有其他GPU和后端的适配代码，使代码更清晰、易于使用。

## 主要改动

### 1. GPU 支持限制
- **仅支持 A100 GPU (SM 8.0)**
- 移除了 H100/H200 (SM 9.0) 的支持
- 添加了启动时的GPU检测，非A100会立即报错

### 2. 后端简化
- **仅保留 CUTLASS 后端**
- 移除了 Triton 后端
- 移除了 PyTorch 后端

### 3. 删除的文件和目录
- `corelib/hstu/hopper/` - H100/H200 专用实现（整个目录）

### 4. 核心文件修改

#### `examples/hstu/modules/hstu_attention.py`
- 删除 `TorchHSTUAttention` 类
- 删除 `TritonHSTUAttention` 类
- 删除 `FusedHSTUAttentionHopper` 类（H100专用）
- 保留 `FusedHSTUAttention` 类（A100 CUTLASS实现）
- 简化 `create_hstu_attention()` 函数，只支持 CUTLASS 后端
- 添加 `check_a100_gpu()` 函数用于GPU验证

#### `examples/hstu/ops/fused_hstu_op.py`
- 移除所有 SM 9.0 (Hopper) 相关代码路径
- 移除 `hstu_hopper_cuda` 导入
- 简化前向和反向传播中的GPU版本检测逻辑
- 固定使用 A100 的 `flash_attn_cuda_ampere` 实现

#### `examples/hstu/configs/hstu_config.py`
- 简化 `KernelBackend` 枚举，只保留 `CUTLASS`
- 移除 `TRITON` 和 `PYTORCH` 选项

#### `examples/hstu/training/gin_config_args.py`
- 修改 `NetworkArgs.__post_init__()` 检查，强制要求 CUTLASS 后端

#### `examples/hstu/training/utils.py`
- 简化 `create_hstu_config()` 中的后端选择逻辑
- 只允许 CUTLASS 后端，其他后端会报错

#### `examples/hstu/pretrain_gr_ranking.py`
- 在 `main()` 函数开始处添加 A100 GPU 检测

#### `examples/hstu/pretrain_gr_retrieval.py`
- 在 `main()` 函数开始处添加 A100 GPU 检测

## 使用方法

### 前置要求
- NVIDIA A100 GPU
- CUDA 11.8+
- PyTorch 2.0+
- 已编译的 `hstu_attn_2_cuda` CUTLASS 扩展

### 运行训练

#### Ranking 任务
```bash
torchrun --nproc_per_node=1 examples/hstu/pretrain_gr_ranking.py \
    --gin-config-file=examples/hstu/kuairand_1k_ranking.gin
```

#### Retrieval 任务
```bash
torchrun --nproc_per_node=1 examples/hstu/pretrain_gr_retrieval.py \
    --gin-config-file=examples/hstu/movielen_retrieval.gin
```

### GPU 检测
代码会在启动时自动检测GPU类型：
- ✅ 如果是 A100 (SM 8.0)，继续执行
- ❌ 如果不是 A100，立即报错并退出

```
[GPU Check] Detected A100 GPU (SM 8.0) - OK
```

或

```
RuntimeError: This code is optimized for A100 GPU (SM 8.0) only. 
Detected SM 9.0. Please use an A100 GPU.
```

## 配置要求

所有 `.gin` 配置文件中必须设置：
```python
NetworkArgs.kernel_backend = "cutlass"  # 必须是 cutlass
```

如果设置为其他值（`triton` 或 `pytorch`），代码会报错。

## 重构效果

### 代码简化
- 删除了约 **300+ 行**的适配代码
- 移除了 3 个后端实现类
- 删除了整个 Hopper 目录

### 性能
- **无性能损失** - 保留的代码就是原本的A100优化版本
- CUTLASS 后端针对 A100 进行了深度优化

### 可维护性
- ✅ 代码逻辑更清晰
- ✅ 无需考虑多GPU适配
- ✅ 配置选项更少，减少错误配置
- ✅ 启动时GPU检测，快速失败

## 注意事项

1. **此代码只能在 A100 GPU 上运行**
   - 不支持其他 GPU（V100、P100、H100、H200 等）
   - 尝试在其他 GPU 上运行会立即报错

2. **必须使用 CUTLASS 后端**
   - 配置文件中不要设置其他后端
   - 代码会自动验证后端设置

3. **需要预编译的 CUDA 扩展**
   - `hstu_attn_2_cuda` (A100 CUTLASS kernels)
   - 确保编译时针对 SM 8.0

## 恢复到原版本

如果需要恢复到支持多GPU的原版本，请从 git 历史恢复：
```bash
git checkout HEAD~1  # 恢复到重构前的版本
```

## 技术细节

### A100 特性使用
- Tensor Cores (3rd generation)
- CUTLASS 2.x 模板库
- 针对 SM 8.0 优化的 kernel

### CUTLASS 后端特点
- 融合的注意力计算
- 高效的内存访问模式
- 针对变长序列的优化

## 相关文件

### 主要训练脚本
- `examples/hstu/pretrain_gr_ranking.py` - Ranking任务训练
- `examples/hstu/pretrain_gr_retrieval.py` - Retrieval任务训练

### 核心模块
- `examples/hstu/modules/hstu_attention.py` - HSTU注意力实现
- `examples/hstu/modules/fused_hstu_layer.py` - 融合HSTU层
- `examples/hstu/modules/native_hstu_layer.py` - 原生HSTU层
- `examples/hstu/ops/fused_hstu_op.py` - 融合操作实现

### 配置
- `examples/hstu/configs/hstu_config.py` - HSTU配置定义
- `examples/hstu/training/gin_config_args.py` - Gin配置参数
- `examples/hstu/*.gin` - 具体任务的配置文件

## 联系方式

如有问题，请参考原始仓库的文档或提交 Issue。

---

**重构日期**: 2025-11-02
**重构目标**: 简化代码，专注 A100 GPU 支持

