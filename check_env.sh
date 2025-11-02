#!/bin/bash
# ============================================
# A100环境验证脚本
# 使用方法: bash check_env.sh
# ============================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}A100环境验证脚本${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${RED}✗ Conda环境未激活${NC}"
    echo "  请先运行: conda activate hstu_a100"
    exit 1
fi

echo -e "${GREEN}✓ Conda环境: $CONDA_DEFAULT_ENV${NC}"
echo ""

# 运行Python验证
python << 'EOF'
import sys

# 颜色代码
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def print_header(text):
    print(f"{BLUE}{'=' * 45}{NC}")
    print(f"{BLUE}{text}{NC}")
    print(f"{BLUE}{'=' * 45}{NC}")

def check(name, func, details=None):
    """检查单个组件"""
    try:
        result = func()
        if details:
            print(f"{GREEN}✓ {name}{NC}")
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {result}")
        else:
            print(f"{GREEN}✓ {name}{NC}")
        return True
    except Exception as e:
        print(f"{RED}✗ {name}{NC}")
        print(f"  错误: {e}")
        return False

print_header("1. 系统环境检查")
print("")

# Python版本
def check_python():
    return f"Python {sys.version.split()[0]}"

check("Python版本", check_python, details=True)

# ============================================
print("")
print_header("2. GPU和CUDA检查")
print("")

# PyTorch
def check_pytorch():
    import torch
    return {
        "版本": torch.__version__,
        "编译CUDA": torch.version.cuda,
    }

passed_torch = check("PyTorch", check_pytorch, details=True)

# CUDA可用性
def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    return "CUDA可用"

passed_cuda = check("CUDA可用性", check_cuda, details=True)

# GPU信息
def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    gpu_count = torch.cuda.device_count()
    
    return {
        "GPU名称": gpu_name,
        "计算能力": f"{compute_cap[0]}.{compute_cap[1]}",
        "GPU数量": gpu_count,
    }

passed_gpu = check("GPU信息", check_gpu, details=True)

# A100检测
def check_a100():
    import torch
    compute_cap = torch.cuda.get_device_capability(0)
    if compute_cap != (8, 0):
        raise RuntimeError(f"不是A100 GPU (计算能力: {compute_cap})")
    return "A100 GPU (SM 8.0)"

passed_a100 = check("A100检测", check_a100, details=True)

# ============================================
print("")
print_header("3. 基础依赖检查")
print("")

# FBGEMM_GPU
def check_fbgemm():
    import fbgemm_gpu
    return f"版本: {fbgemm_gpu.__version__}"

passed_fbgemm = check("FBGEMM_GPU", check_fbgemm, details=True)

# TorchRec
def check_torchrec():
    import torchrec
    return f"版本: {torchrec.__version__}"

passed_torchrec = check("TorchRec", check_torchrec, details=True)

# Megatron-Core
def check_megatron():
    import megatron
    from megatron.core import parallel_state
    return "已安装"

passed_megatron = check("Megatron-Core", check_megatron, details=True)

# 其他依赖
def check_others():
    import gin
    import torchmetrics
    import einops
    return "gin-config, torchmetrics, einops"

check("其他依赖", check_others, details=True)

# ============================================
print("")
print_header("4. CUDA加速模块检查")
print("")

# HSTU Attention
def check_hstu_attn():
    import hstu_attn
    return "CUTLASS内核"

passed_hstu = check("HSTU Attention", check_hstu_attn, details=True)

# DynamicEmb
def check_dynamicemb():
    import dynamicemb
    return f"版本: {dynamicemb.__version__}"

passed_dynamicemb = check("Dynamic Embeddings", check_dynamicemb, details=True)

# HSTU CUDA Ops
def check_hstu_ops():
    import hstu_cuda_ops
    return "Jagged Tensor算子"

passed_ops1 = check("HSTU CUDA Ops", check_hstu_ops, details=True)

# Paged KVCache Ops
def check_kvcache_ops():
    import paged_kvcache_ops
    return "Paged KVCache算子"

passed_ops2 = check("Paged KVCache Ops", check_kvcache_ops, details=True)

# ============================================
print("")
print_header("5. 功能测试")
print("")

# HSTU Attention前向传播测试
def test_hstu_forward():
    import torch
    from hstu_attn import hstu_attn_varlen_func
    
    batch_size = 2
    nheads = 8
    headdim = 64
    seqlen = 100
    
    q = torch.randn(batch_size * seqlen, nheads, headdim, dtype=torch.float16).cuda()
    k = torch.randn(batch_size * seqlen, nheads, headdim, dtype=torch.float16).cuda()
    v = torch.randn(batch_size * seqlen, nheads, headdim, dtype=torch.float16).cuda()
    cu_seqlens = torch.tensor([0, seqlen, 2*seqlen], dtype=torch.int32).cuda()
    
    out = hstu_attn_varlen_func(
        q, k, v,
        cu_seqlens, cu_seqlens,
        seqlen, seqlen
    )
    
    return f"输出shape: {out.shape}, dtype: {out.dtype}"

check("HSTU Attention前向传播", test_hstu_forward, details=True)

# Dynamic Embeddings测试
def test_dynamicemb():
    import torch
    from torchrec import EmbeddingBagConfig
    # 只检查导入，不实际创建 (避免初始化开销)
    return "导入成功"

check("Dynamic Embeddings导入", test_dynamicemb, details=True)

# ============================================
print("")
print_header("6. 统计结果")
print("")

checks = [
    passed_torch,
    passed_cuda,
    passed_gpu,
    passed_a100,
    passed_fbgemm,
    passed_torchrec,
    passed_megatron,
    passed_hstu,
    passed_dynamicemb,
    passed_ops1,
    passed_ops2,
]

total = len(checks)
passed = sum(checks)
failed = total - passed

print(f"总计: {total}")
print(f"{GREEN}通过: {passed}{NC}")
if failed > 0:
    print(f"{RED}失败: {failed}{NC}")

print("")
if passed == total:
    print(f"{GREEN}🎉 所有检查通过！环境配置完成！{NC}")
    print("")
    print("下一步:")
    print("1. 准备数据: python preprocessor.py --dataset_name ml-20m")
    print("2. 开始训练: torchrun pretrain_gr_ranking.py --gin-config-file movielen_ranking.gin")
    sys.exit(0)
else:
    print(f"{RED}❌ 部分检查失败，请根据上述错误信息进行修复{NC}")
    print("")
    print("常见问题排查:")
    print("1. 检查conda环境是否正确激活")
    print("2. 检查CUDA和GPU驱动是否正常")
    print("3. 检查子模块是否正确初始化: git submodule update --init --recursive")
    print("4. 重新编译CUDA模块: cd corelib/hstu && pip install . --force-reinstall")
    print("")
    print("详细教程: A100环境配置完整教程.md")
    sys.exit(1)

EOF

