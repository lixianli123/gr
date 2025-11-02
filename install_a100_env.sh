#!/bin/bash
# ============================================
# A100 HSTU训练环境自动配置脚本
# 作者: AI Assistant
# 使用方法: bash install_a100_env.sh
# ============================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=====================================${NC}"
}

print_step() {
    echo -e "${GREEN}[$1] $2${NC}"
}

print_warning() {
    echo -e "${YELLOW}警告: $1${NC}"
}

print_error() {
    echo -e "${RED}错误: $1${NC}"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================
# 开始安装
# ============================================

print_header "A100 HSTU训练环境自动配置脚本"

# ============================================
# 1. 检查GPU
# ============================================
print_step "1/12" "检查GPU..."

if ! command_exists nvidia-smi; then
    print_error "nvidia-smi未找到，请先安装NVIDIA驱动"
    exit 1
fi

nvidia-smi --query-gpu=gpu_name,compute_cap --format=csv

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')
echo "检测到计算能力: $COMPUTE_CAP"

if [ "$COMPUTE_CAP" != "8.0" ]; then
    print_warning "当前GPU计算能力为 $COMPUTE_CAP，不是A100 (8.0)"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ============================================
# 2. 检查CUDA
# ============================================
print_step "2/12" "检查CUDA..."

if ! command_exists nvcc; then
    print_error "nvcc未找到，请先安装CUDA Toolkit"
    exit 1
fi

nvcc --version

# ============================================
# 3. 检查conda
# ============================================
print_step "3/12" "检查conda..."

if ! command_exists conda; then
    print_error "conda未找到，请先安装Miniconda或Anaconda"
    exit 1
fi

# ============================================
# 4. 创建conda环境
# ============================================
print_step "4/12" "创建conda环境 hstu_a100..."

# 检查环境是否已存在
if conda env list | grep -q "hstu_a100"; then
    print_warning "环境 hstu_a100 已存在"
    read -p "是否删除并重新创建? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n hstu_a100 -y
        conda create -n hstu_a100 python=3.10 -y
    fi
else
    conda create -n hstu_a100 python=3.10 -y
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hstu_a100

# ============================================
# 5. 安装PyTorch
# ============================================
print_step "5/12" "安装PyTorch..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证PyTorch
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
assert torch.cuda.is_available(), 'CUDA不可用'
"

# ============================================
# 6. 安装基础依赖
# ============================================
print_step "6/12" "安装基础依赖..."

pip install --upgrade pip setuptools wheel
pip install ninja psutil packaging einops

# ============================================
# 7. 编译FBGEMM_GPU
# ============================================
print_step "7/12" "编译FBGEMM_GPU (需要10-30分钟，请耐心等待)..."

cd ~
pip install --no-cache setuptools==69.5.1 setuptools-git-versioning scikit-build

if [ -d "fbgemm" ]; then
    print_warning "fbgemm目录已存在，跳过克隆"
else
    git clone --recursive -b main https://github.com/pytorch/FBGEMM.git fbgemm
fi

cd fbgemm/fbgemm_gpu
git checkout 642ccb980d05aa1be00ccd131c5991b0914e2e64

# 编译 (只为A100的SM 8.0编译)
MAX_JOBS=4 python setup.py install --package_variant=cuda -DTORCH_CUDA_ARCH_LIST="8.0"

# 验证
python -c "import fbgemm_gpu; print(f'FBGEMM_GPU: {fbgemm_gpu.__version__}')"

# ============================================
# 8. 安装TorchRec
# ============================================
print_step "8/12" "安装TorchRec..."

cd ~
pip install --no-deps tensordict orjson

if [ -d "torchrec" ]; then
    print_warning "torchrec目录已存在，跳过克隆"
else
    git clone --recursive -b main https://github.com/pytorch/torchrec.git torchrec
fi

cd torchrec
git checkout 6aaf1fa72e884642f39c49ef232162fa3772055e
pip install --no-deps .

# 验证
python -c "import torchrec; print(f'TorchRec: {torchrec.__version__}')"

# ============================================
# 9. 安装Megatron-Core
# ============================================
print_step "9/12" "安装Megatron-Core..."

cd ~

if [ -d "megatron-lm" ]; then
    print_warning "megatron-lm目录已存在，跳过克隆"
else
    git clone -b core_r0.9.0 https://github.com/NVIDIA/Megatron-LM.git megatron-lm
fi

cd megatron-lm
pip install -e .

# 验证
python -c "import megatron; print('Megatron-Core安装成功')"

# ============================================
# 10. 安装其他依赖
# ============================================
print_step "10/12" "安装其他Python依赖..."

pip install torchx gin-config torchmetrics==1.0.3 typing-extensions iopath

# ============================================
# 11. 询问项目路径
# ============================================
print_header "项目路径配置"

echo "请输入recsys-examples-main的完整路径:"
echo "例如: /home/user/recsys-examples-main"
read -p "路径: " PROJECT_PATH

# 验证路径
if [ ! -d "$PROJECT_PATH" ]; then
    print_error "路径不存在: $PROJECT_PATH"
    exit 1
fi

if [ ! -f "$PROJECT_PATH/README.md" ]; then
    print_error "路径下没有找到README.md，请确认是否为正确的项目路径"
    exit 1
fi

cd "$PROJECT_PATH"

# ============================================
# 12. 初始化子模块
# ============================================
print_step "11/12" "初始化Git子模块..."

git submodule update --init third_party/cutlass
git submodule update --init third_party/HierarchicalKV

# 验证子模块
if [ ! -f "third_party/cutlass/include/cutlass/cutlass.h" ]; then
    print_error "CUTLASS子模块初始化失败"
    exit 1
fi

if [ ! -d "third_party/HierarchicalKV/include" ]; then
    print_error "HierarchicalKV子模块初始化失败"
    exit 1
fi

# ============================================
# 13. 编译CUDA加速模块
# ============================================
print_step "12/12" "编译CUDA加速模块 (需要30-60分钟，请耐心等待)..."

# 13.1 编译HSTU Attention
echo "  → 编译HSTU Attention (CUTLASS内核)..."
cd "$PROJECT_PATH/corelib/hstu"

export HSTU_DISABLE_86OR89=TRUE
export HSTU_DISABLE_ARBITRARY=TRUE
export HSTU_DISABLE_LOCAL=TRUE
export HSTU_DISABLE_RAB=TRUE
export HSTU_DISABLE_DRAB=TRUE
export NVCC_THREADS=4
export MAX_JOBS=4

pip install .

# 验证
python -c "import hstu_attn; print('HSTU Attention安装成功')"

# 13.2 编译Dynamic Embeddings
echo "  → 编译Dynamic Embeddings..."
cd "$PROJECT_PATH/corelib/dynamicemb"

python setup.py install

# 验证
python -c "import dynamicemb; print(f'DynamicEmb: {dynamicemb.__version__}')"

# 13.3 编译HSTU训练算子
echo "  → 编译HSTU训练算子..."
cd "$PROJECT_PATH/examples/hstu"

python setup.py install

# 验证
python -c "import hstu_cuda_ops; import paged_kvcache_ops; print('HSTU训练算子安装成功')"

# ============================================
# 14. 完整验证
# ============================================
print_header "验证安装"

python -c '
import torch
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA {torch.version.cuda}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ Compute Capability: {torch.cuda.get_device_capability(0)}")

import fbgemm_gpu
print(f"✓ FBGEMM_GPU {fbgemm_gpu.__version__}")

import torchrec
print(f"✓ TorchRec {torchrec.__version__}")

import megatron
print(f"✓ Megatron-Core")

import hstu_attn
print(f"✓ HSTU Attention (CUTLASS)")

import dynamicemb
print(f"✓ DynamicEmb {dynamicemb.__version__}")

import hstu_cuda_ops
import paged_kvcache_ops
print(f"✓ HSTU CUDA Ops")

print("\n🎉 所有模块安装成功！")
'

# ============================================
# 完成
# ============================================
print_header "安装完成！"

echo ""
echo "环境已成功配置到conda环境: hstu_a100"
echo "项目路径: $PROJECT_PATH"
echo ""
echo -e "${GREEN}下一步操作:${NC}"
echo ""
echo "1. 激活环境:"
echo "   conda activate hstu_a100"
echo ""
echo "2. 进入项目目录:"
echo "   cd $PROJECT_PATH/examples/hstu"
echo ""
echo "3. 准备数据:"
echo "   mkdir -p ./tmp_data"
echo "   python preprocessor.py --dataset_name ml-20m"
echo ""
echo "4. 开始训练 (Ranking任务):"
echo "   PYTHONPATH=\${PYTHONPATH}:\$(realpath ../) \\"
echo "   torchrun --nproc_per_node 1 \\"
echo "            --master_addr localhost \\"
echo "            --master_port 6000 \\"
echo "            pretrain_gr_ranking.py \\"
echo "            --gin-config-file movielen_ranking.gin"
echo ""
echo "5. 开始训练 (Retrieval任务):"
echo "   PYTHONPATH=\${PYTHONPATH}:\$(realpath ../) \\"
echo "   torchrun --nproc_per_node 1 \\"
echo "            --master_addr localhost \\"
echo "            --master_port 6000 \\"
echo "            pretrain_gr_retrieval.py \\"
echo "            --gin-config-file movielen_retrieval.gin"
echo ""
echo -e "${BLUE}更多信息请查看: A100环境配置完整教程.md${NC}"
echo ""

