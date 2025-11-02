# 代码改动详细清单

## 文件修改

### 1. `examples/hstu/modules/hstu_attention.py`

**删除的类**:
- `TorchHSTUAttention` (约60行) - PyTorch 原生实现
- `TritonHSTUAttention` (约70行) - Triton 后端实现  
- `FusedHSTUAttentionHopper` (约90行) - H100/H200 专用实现

**保留的类**:
- `FusedHSTUAttention` - A100 CUTLASS 实现

**新增函数**:
- `check_a100_gpu()` - GPU检测函数
  - 检查 CUDA 可用性
  - 验证 SM 版本为 8.0
  - 非 A100 立即报错

**简化的函数**:
- `create_hstu_attention()` - 从约40行简化为约20行
  - 移除多后端选择逻辑
  - 只支持 CUTLASS 后端
  - 添加 A100 检测

### 2. `examples/hstu/ops/fused_hstu_op.py`

**导入简化**:
```python
# 之前
try:
    import hstu_attn_2_cuda as flash_attn_cuda_ampere
except ImportError:
    pass

try:
    import hstu_hopper_cuda as flash_attn_cuda_hopper
except ImportError:
    pass

# 之后
import hstu_attn_2_cuda as flash_attn_cuda_ampere
```

**前向传播简化** (`FusedHSTULayerFunction.forward`):

1. 删除 SM 版本判断（约15行）:
```python
# 删除前
sm = torch.cuda.get_device_properties(0).major
if sm == 8:
    addmm_silu_fwd_impl = triton_addmm_silu_fwd
elif sm == 9:
    addmm_silu_fwd_impl = torch_addmm_silu_fwd
else:
    raise ValueError(f"Unsupported SM major version: {sm}")

# 删除后
# A100 uses triton implementation
addmm_silu_fwd_impl = triton_addmm_silu_fwd
```

2. 简化 CUTLASS 前向实现（约30行）:
```python
# 删除前
sm_major_version = torch.cuda.get_device_properties(0).major
extension_args = ()
if sm_major_version == 8:
    cutlass_hstu_varlen_fwd = flash_attn_cuda_ampere.varlen_fwd
    ampere_paged_kv_args = (None, None, None, None, None)
    extension_args = ampere_paged_kv_args
elif sm_major_version == 9:
    cutlass_hstu_varlen_fwd = flash_attn_cuda_hopper.varlen_fwd
    hopper_fp8_args = (-1, None, None, None, None, None, None, None, None)
    extension_args = hopper_fp8_args
else:
    raise ValueError(f"Unsupported SM major version: {sm_major_version}")

# 删除后
# A100 (SM 8.0) CUTLASS implementation
cutlass_hstu_varlen_fwd = flash_attn_cuda_ampere.varlen_fwd
ampere_paged_kv_args = (None, None, None, None, None)
extension_args = ampere_paged_kv_args
```

**反向传播简化** (`FusedHSTULayerFunction.backward`):

删除 SM 9.0 分支（约40行）:
```python
# 删除前
sm_major_version = torch.cuda.get_device_properties(0).major
assert dout.dim() == 3
if sm_major_version == 8:
    dq, dk, dv, _ = flash_attn_cuda_ampere.varlen_bwd(...)
elif sm_major_version == 9:
    fp8_args = (None,) * 11
    dq, dk, dv, _ = flash_attn_cuda_hopper.varlen_bwd(...)
else:
    raise ValueError(f"Unsupported SM major version: {sm_major_version}")

# 删除后
# A100 (SM 8.0) CUTLASS backward implementation
assert dout.dim() == 3
dq, dk, dv, _ = flash_attn_cuda_ampere.varlen_bwd(...)
```

### 3. `examples/hstu/configs/hstu_config.py`

**KernelBackend 枚举简化**:
```python
# 之前
class KernelBackend(Enum):
    TRITON = "TRITON"
    PYTORCH = "PYTORCH"
    CUTLASS = "CUTLASS"

# 之后
class KernelBackend(Enum):
    """For A100 GPU, only CUTLASS backend is supported."""
    CUTLASS = "CUTLASS"
```

### 4. `examples/hstu/training/gin_config_args.py`

**NetworkArgs 后置检查**:
```python
# 之前
assert self.kernel_backend.lower() in ["cutlass", "triton", "pytorch"]

# 之后  
assert self.kernel_backend.lower() == "cutlass", \
    "Only CUTLASS backend is supported for A100 GPU"
```

### 5. `examples/hstu/training/utils.py`

**create_hstu_config() 简化**:
```python
# 之前
kernel_backend = None
if network_args.kernel_backend == "cutlass":
    kernel_backend = KernelBackend.CUTLASS
elif network_args.kernel_backend == "triton":
    kernel_backend = KernelBackend.TRITON
elif network_args.kernel_backend == "pytorch":
    kernel_backend = KernelBackend.PYTORCH
else:
    raise ValueError(f"Kernel backend {network_args.kernel_backend} is not supported.")

# 之后
# Only CUTLASS backend is supported for A100
if network_args.kernel_backend.lower() != "cutlass":
    raise ValueError(
        f"Only CUTLASS backend is supported for A100 GPU. Got: {network_args.kernel_backend}"
    )
kernel_backend = KernelBackend.CUTLASS
```

### 6. `examples/hstu/pretrain_gr_ranking.py`

**main() 函数添加 GPU 检测**:
```python
def main():
    # Check A100 GPU at the very beginning
    from modules.hstu_attention import check_a100_gpu
    check_a100_gpu()
    
    init.initialize_distributed()
    # ... 其余代码
```

### 7. `examples/hstu/pretrain_gr_retrieval.py`

**main() 函数添加 GPU 检测**:
```python
def main():
    # Check A100 GPU at the very beginning
    from modules.hstu_attention import check_a100_gpu
    check_a100_gpu()
    
    init.initialize_distributed()
    # ... 其余代码
```

## 删除的目录

### `corelib/hstu/hopper/` (整个目录)

包含的文件：
- `__init__.py`
- `hstu_api.cpp`
- `hstu_attn_interface.py`
- `hstu_bwd_kernel.h`
- `hstu_bwd_launch_template.h`
- `hstu_bwd_postprocess_kernel.h`
- `hstu_fwd_kernel.h`
- `hstu_fwd_launch_template.h`
- `hstu.h`
- `kernel_traits.h`
- `mainloop_bwd_sm90_tma_gmma_ws.hpp`
- `mainloop_fwd_sm90_tma_gmma_ws.hpp`
- `epilogue_bwd_sm90_tma.hpp`
- `epilogue_fwd_sm90_tma.hpp`
- `tile_scheduler_bwd.hpp`
- `tile_scheduler.hpp`
- `named_barrier.hpp`
- `seq_len.h`
- `static_switch.h`
- `utils.h`
- `setup.py`
- `Makefile`
- `instantiations/` 目录及其内容

## 统计数据

### 代码行数变化
- **删除总行数**: 约 350+ 行
- **新增总行数**: 约 20 行
- **净减少**: 约 330 行

### 文件数量变化
- **删除文件**: 20+ 个（Hopper 目录）
- **修改文件**: 7 个
- **新增文件**: 3 个（文档）

### 复杂度降低
- **后端选项**: 3 个 → 1 个 (-66%)
- **GPU 支持**: 3+ 种 → 1 种 (-66%)
- **注意力实现类**: 4 个 → 1 个 (-75%)

## 兼容性影响

### 不再支持的功能
1. ❌ H100/H200 GPU (SM 9.0)
2. ❌ V100 及更早的 GPU
3. ❌ Triton 后端
4. ❌ PyTorch 原生后端
5. ❌ 非 CUTLASS 的配置选项

### 保留的功能
1. ✅ A100 GPU (SM 8.0) - 完整支持
2. ✅ CUTLASS 后端 - 完整支持
3. ✅ 动态 Embedding - 完整支持
4. ✅ Tensor Parallel - 完整支持
5. ✅ 混合精度训练 - 完整支持
6. ✅ Ranking 和 Retrieval 任务 - 完整支持

## 性能影响

- ✅ **无性能损失**: 保留的代码就是原有的 A100 优化路径
- ✅ **启动更快**: 减少了条件判断
- ✅ **内存占用**: 无变化
- ✅ **计算效率**: 无变化

## 错误处理改进

### 之前
- 不支持的 GPU 可能会回退到低效的 PyTorch 实现
- 不清楚使用的是哪个后端
- 静默失败或性能下降

### 之后
- 非 A100 立即报错，清晰的错误信息
- 非 CUTLASS 后端立即报错
- 快速失败，明确提示

## 代码质量提升

1. **可读性**: 删除大量分支判断，代码更直接
2. **可维护性**: 只需维护一个代码路径
3. **可测试性**: 测试场景大幅减少
4. **文档化**: 添加清晰的注释和文档

---

**修改日期**: 2025-11-02
**修改目的**: 专注 A100 支持，简化代码结构

