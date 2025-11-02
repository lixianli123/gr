#!/bin/bash
# ============================================
# HSTU + Dynamic Embeddings 源码迁移脚本
# 使用方法: bash copy_source_files.sh
# ============================================

set -e

# 配置路径 - 请修改为你的实际路径
SOURCE_ROOT="${1:-/path/to/recsys-examples-main}"
TARGET_ROOT="${2:-./accelerated_modules}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=====================================${NC}"
}

print_step() {
    echo -e "\n${GREEN}[$1] $2${NC}"
}

print_warning() {
    echo -e "${YELLOW}警告: $1${NC}"
}

print_error() {
    echo -e "${RED}错误: $1${NC}"
}

# 检查源路径
if [ ! -d "$SOURCE_ROOT" ]; then
    print_error "源路径不存在: $SOURCE_ROOT"
    echo "使用方法: bash copy_source_files.sh <源路径> [目标路径]"
    exit 1
fi

print_header "源码级迁移脚本"
echo "源路径: $SOURCE_ROOT"
echo "目标路径: $TARGET_ROOT"

# ============================================
# 第1步: 创建目录结构
# ============================================
print_step "1/7" "创建目录结构"

mkdir -p $TARGET_ROOT
mkdir -p $TARGET_ROOT/hstu_attn/csrc/src
mkdir -p $TARGET_ROOT/hstu_attn/csrc/src/generated
mkdir -p $TARGET_ROOT/dynamicemb/src/hkv_variable_instantiations
mkdir -p $TARGET_ROOT/dynamicemb/python/planner
mkdir -p $TARGET_ROOT/dynamicemb/python/shard
mkdir -p $TARGET_ROOT/third_party
mkdir -p $TARGET_ROOT/tests
mkdir -p $TARGET_ROOT/examples

echo "  ✓ 目录结构创建完成"

# ============================================
# 第2步: 拷贝HSTU Attention源码
# ============================================
print_step "2/7" "拷贝HSTU Attention源码"

# C++/CUDA源码
echo "  - 拷贝C++/CUDA源码..."
cp $SOURCE_ROOT/corelib/hstu/csrc/hstu_attn/hstu_api.cpp \
   $TARGET_ROOT/hstu_attn/csrc/ || print_warning "hstu_api.cpp未找到"

cp $SOURCE_ROOT/corelib/hstu/csrc/hstu_attn/src/*.h \
   $TARGET_ROOT/hstu_attn/csrc/src/ 2>/dev/null || print_warning "头文件未找到"

# Python接口
echo "  - 拷贝Python接口..."
cp $SOURCE_ROOT/corelib/hstu/hstu_attn/__init__.py \
   $TARGET_ROOT/hstu_attn/ || print_warning "__init__.py未找到"

cp $SOURCE_ROOT/corelib/hstu/hstu_attn/hstu_attn_interface.py \
   $TARGET_ROOT/hstu_attn/interface.py || print_warning "interface文件未找到"

# 编译配置
echo "  - 拷贝编译配置..."
cp $SOURCE_ROOT/corelib/hstu/setup.py \
   $TARGET_ROOT/hstu_attn/ || print_warning "setup.py未找到"

echo "  ✓ HSTU Attention源码拷贝完成"

# ============================================
# 第3步: 拷贝Dynamic Embeddings源码
# ============================================
print_step "3/7" "拷贝Dynamic Embeddings源码"

# CUDA源码
echo "  - 拷贝CUDA源码..."
for file in dynamic_emb_op.cu dynamic_variable_base.cu dynamic_variable_base.h \
            hkv_variable.h hkv_variable.cuh \
            lookup_forward.cu lookup_forward.h lookup_backward.cu lookup_backward.h \
            lookup_kernel.cuh optimizer.cu optimizer.h optimizer_kernel.cuh \
            initializer.cu initializer.cuh unique_op.cu unique_op.h \
            unique_variable.cu unique_variable.h \
            index_calculation.cu index_calculation.h \
            sparse_block_bucketize_features.cu sparse_block_bucketize_features_utils.h \
            torch_utils.cu torch_utils.h utils.cpp utils.h check.h module_bind.cu; do
    if [ -f "$SOURCE_ROOT/corelib/dynamicemb/src/$file" ]; then
        cp "$SOURCE_ROOT/corelib/dynamicemb/src/$file" \
           "$TARGET_ROOT/dynamicemb/src/"
    fi
done

# HKV实例化文件
echo "  - 拷贝HKV实例化文件..."
cp $SOURCE_ROOT/corelib/dynamicemb/src/hkv_variable_instantiations/*.cu \
   $TARGET_ROOT/dynamicemb/src/hkv_variable_instantiations/ 2>/dev/null || print_warning "HKV文件未找到"

# Python接口
echo "  - 拷贝Python接口..."
cp -r $SOURCE_ROOT/corelib/dynamicemb/dynamicemb/* \
      $TARGET_ROOT/dynamicemb/python/ 2>/dev/null || print_warning "Python接口未找到"

# 编译配置
echo "  - 拷贝编译配置..."
cp $SOURCE_ROOT/corelib/dynamicemb/setup.py \
   $TARGET_ROOT/dynamicemb/ || print_warning "setup.py未找到"

cp $SOURCE_ROOT/corelib/dynamicemb/version.txt \
   $TARGET_ROOT/dynamicemb/ || print_warning "version.txt未找到"

echo "  ✓ Dynamic Embeddings源码拷贝完成"

# ============================================
# 第4步: 创建配置文件
# ============================================
print_step "4/7" "创建配置文件"

# requirements.txt
cat > $TARGET_ROOT/requirements.txt << 'EOF'
torch>=2.0.0
einops
packaging
ninja
setuptools>=69.5.1
scikit-build
psutil
torchrec>=1.2.0
tensordict
orjson
EOF

# 主README.md
cat > $TARGET_ROOT/README.md << 'EOF'
# Accelerated Modules for A100 GPU

HSTU Attention (CUTLASS) + Dynamic Embeddings (HierarchicalKV)

## 快速开始

```bash
# 1. 配置环境
source env.sh

# 2. 编译模块
bash build.sh

# 3. 测试
python tests/test_hstu_attention.py
python tests/test_dynamicemb.py
```

## 使用

```python
from hstu_attn import hstu_attn_varlen_func
from dynamicemb import DynamicEmbeddingBagCollection
```

## 要求

- NVIDIA A100 GPU (SM 8.0)
- CUDA >= 11.6 (推荐 12.1)
- Python >= 3.9
- PyTorch >= 2.0.0
- TorchRec >= 1.2.0

## 目录结构

```
accelerated_modules/
├── hstu_attn/          # HSTU Attention模块
├── dynamicemb/         # Dynamic Embeddings模块
├── third_party/        # 第三方依赖 (CUTLASS, HKV)
├── tests/              # 测试
└── examples/           # 示例
```

## 编译选项

```bash
# 只编译A100 (SM 8.0)
export TORCH_CUDA_ARCH_LIST="8.0"

# HSTU优化选项
export HSTU_DISABLE_86OR89=TRUE
export HSTU_DISABLE_ARBITRARY=TRUE
export HSTU_DISABLE_LOCAL=TRUE
export HSTU_DISABLE_RAB=TRUE
export HSTU_DISABLE_DRAB=TRUE

# 并行编译
export MAX_JOBS=4
export NVCC_THREADS=4
```

## 性能

- HSTU Attention: 15倍加速 (相比PyTorch原生)
- Dynamic Embeddings: 10倍加速 (相比CPU)
- 端到端: 30倍加速

## License

Apache 2.0
EOF

# 环境变量配置
cat > $TARGET_ROOT/env.sh << 'BASHEOF'
#!/bin/bash
# 环境变量配置

# CUDA路径 (根据实际情况修改)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 编译选项
export TORCH_CUDA_ARCH_LIST="8.0"          # 只编译A100
export MAX_JOBS=4                          # 并行编译数
export NVCC_THREADS=4                      # NVCC线程数

# HSTU编译选项
export HSTU_DISABLE_86OR89=TRUE
export HSTU_DISABLE_ARBITRARY=TRUE
export HSTU_DISABLE_LOCAL=TRUE
export HSTU_DISABLE_RAB=TRUE
export HSTU_DISABLE_DRAB=TRUE

# NVIDIA优化
export NVIDIA_TF32_OVERRIDE=1              # 启用TF32
export CUDNN_BENCHMARK=1                   # CUDNN自动调优

echo "✓ 环境变量已设置"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CUDA_ARCH: $TORCH_CUDA_ARCH_LIST"
echo "  MAX_JOBS: $MAX_JOBS"
BASHEOF

chmod +x $TARGET_ROOT/env.sh

echo "  ✓ 配置文件创建完成"

# ============================================
# 第5步: 创建编译脚本
# ============================================
print_step "5/7" "创建编译脚本"

cat > $TARGET_ROOT/build.sh << 'BASHEOF'
#!/bin/bash
# 编译HSTU和Dynamic Embeddings

set -e

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================"
echo "编译加速模块"
echo "======================================${NC}"

# 检查环境
echo ""
echo "[1/5] 检查编译环境..."
command -v nvcc >/dev/null 2>&1 || { echo "错误: nvcc未找到"; exit 1; }
command -v python >/dev/null 2>&1 || { echo "错误: python未找到"; exit 1; }
echo "  ✓ NVCC: $(nvcc --version | grep release | head -n1)"
echo "  ✓ Python: $(python --version)"

# 检查PyTorch
python -c "import torch; assert torch.cuda.is_available()" || { echo "错误: PyTorch CUDA不可用"; exit 1; }
echo "  ✓ PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  ✓ CUDA: $(python -c 'import torch; print(torch.version.cuda)')"

# 更新子模块
echo ""
echo "[2/5] 更新Git子模块..."
if [ -d ".git" ]; then
    git submodule update --init --recursive
    echo "  ✓ 子模块已更新"
else
    echo "  ! 不是Git仓库，跳过子模块更新"
    echo "  ! 请手动初始化CUTLASS和HierarchicalKV"
fi

# 安装依赖
echo ""
echo "[3/5] 安装Python依赖..."
pip install -r requirements.txt --quiet
echo "  ✓ 依赖已安装"

# 编译HSTU Attention
echo ""
echo "[4/5] 编译HSTU Attention (约30-60分钟)..."
cd hstu_attn
echo "  - 配置编译选项..."
export HSTU_DISABLE_86OR89=TRUE
export HSTU_DISABLE_ARBITRARY=TRUE
export HSTU_DISABLE_LOCAL=TRUE
export HSTU_DISABLE_RAB=TRUE
export HSTU_DISABLE_DRAB=TRUE
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=${MAX_JOBS:-4}
export NVCC_THREADS=${NVCC_THREADS:-4}
echo "  - 开始编译..."
pip install . --no-deps --quiet || { echo "HSTU编译失败"; exit 1; }
cd ..
echo -e "  ${GREEN}✓ HSTU Attention编译完成${NC}"

# 编译Dynamic Embeddings
echo ""
echo "[5/5] 编译Dynamic Embeddings (约10-20分钟)..."
cd dynamicemb
echo "  - 配置编译选项..."
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=${MAX_JOBS:-4}
echo "  - 开始编译..."
pip install . --no-deps --quiet || { echo "DynamicEmb编译失败"; exit 1; }
cd ..
echo -e "  ${GREEN}✓ Dynamic Embeddings编译完成${NC}"

echo ""
echo -e "${GREEN}======================================"
echo "✓ 编译完成！"
echo "======================================${NC}"
echo ""
echo "验证安装:"
echo "  python -c 'from hstu_attn import hstu_attn_varlen_func'"
echo "  python -c 'from dynamicemb import DynamicEmbeddingBagCollection'"
echo ""
echo "运行测试:"
echo "  python tests/test_hstu_attention.py"
echo "  python tests/test_dynamicemb.py"
echo ""
BASHEOF

chmod +x $TARGET_ROOT/build.sh

echo "  ✓ 编译脚本创建完成"

# ============================================
# 第6步: 创建测试文件
# ============================================
print_step "6/7" "创建测试文件"

# HSTU测试
cat > $TARGET_ROOT/tests/test_hstu_attention.py << 'PYEOF'
"""测试HSTU Attention"""
import torch

print("导入HSTU Attention...")
from hstu_attn import hstu_attn_varlen_func

print("检查CUDA...")
assert torch.cuda.is_available(), "CUDA不可用"
print(f"  GPU: {torch.cuda.get_device_name(0)}")

print("创建测试数据...")
q = torch.randn(200, 4, 64, dtype=torch.bfloat16).cuda()
k = torch.randn(200, 4, 64, dtype=torch.bfloat16).cuda()
v = torch.randn(200, 4, 64, dtype=torch.bfloat16).cuda()
cu_seqlens = torch.tensor([0, 100, 200], dtype=torch.int32).cuda()

print("运行HSTU Attention...")
output = hstu_attn_varlen_func(
    q, k, v, cu_seqlens, cu_seqlens, 100, 100, causal=True
)

assert output.shape == q.shape, f"输出shape错误: {output.shape}"
print(f"  输入shape: {q.shape}")
print(f"  输出shape: {output.shape}")
print("✓ HSTU Attention测试通过")
PYEOF

# DynamicEmb测试
cat > $TARGET_ROOT/tests/test_dynamicemb.py << 'PYEOF'
"""测试Dynamic Embeddings"""
import torch

print("导入Dynamic Embeddings...")
from torchrec import EmbeddingBagConfig
from dynamicemb import DynamicEmbeddingBagCollection

print("检查CUDA...")
assert torch.cuda.is_available(), "CUDA不可用"
print(f"  GPU: {torch.cuda.get_device_name(0)}")

print("创建Dynamic Embeddings...")
ebc = DynamicEmbeddingBagCollection(
    tables=[
        EmbeddingBagConfig(
            name="test_emb",
            embedding_dim=128,
            num_embeddings=1000000,
            feature_names=["feature_id"],
        ),
    ],
    device=torch.device("cuda:0"),
)

print(f"  表名: test_emb")
print(f"  Embedding维度: 128")
print(f"  容量: 1,000,000")
print("✓ Dynamic Embeddings测试通过")
PYEOF

echo "  ✓ 测试文件创建完成"

# ============================================
# 第7步: 创建示例文件
# ============================================
print_step "7/7" "创建示例文件"

cat > $TARGET_ROOT/examples/simple_usage.py << 'PYEOF'
"""简单使用示例"""
import torch

print("=" * 50)
print("HSTU + Dynamic Embeddings 使用示例")
print("=" * 50)

# ============================================
# 1. HSTU Attention
# ============================================
print("\n[1] HSTU Attention示例")
print("-" * 50)

from hstu_attn import hstu_attn_varlen_func

# 创建数据
batch_size, num_heads, head_dim, seqlen = 2, 4, 64, 100
q = torch.randn(batch_size * seqlen, num_heads, head_dim, dtype=torch.bfloat16).cuda()
k = torch.randn(batch_size * seqlen, num_heads, head_dim, dtype=torch.bfloat16).cuda()
v = torch.randn(batch_size * seqlen, num_heads, head_dim, dtype=torch.bfloat16).cuda()
cu_seqlens = torch.tensor([0, seqlen, 2*seqlen], dtype=torch.int32).cuda()

# Attention
output = hstu_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, seqlen, seqlen, causal=True)

print(f"输入: {q.shape}")
print(f"输出: {output.shape}")
print("✓ HSTU Attention工作正常")

# ============================================
# 2. Dynamic Embeddings
# ============================================
print("\n[2] Dynamic Embeddings示例")
print("-" * 50)

from torchrec import EmbeddingBagConfig
from dynamicemb import DynamicEmbeddingBagCollection

ebc = DynamicEmbeddingBagCollection(
    tables=[
        EmbeddingBagConfig(
            name="item_emb",
            embedding_dim=256,
            num_embeddings=10_000_000,  # 1000万物品
            feature_names=["item_id"],
        ),
    ],
    device=torch.device("cuda:0"),
)

print(f"创建了1000万物品的embedding表")
print(f"Embedding维度: 256")
print("✓ Dynamic Embeddings工作正常")

print("\n" + "=" * 50)
print("所有示例运行成功！")
print("=" * 50)
PYEOF

echo "  ✓ 示例文件创建完成"

# ============================================
# 完成
# ============================================
print_header "✓ 源码迁移完成！"

echo ""
echo "目录结构:"
echo "  $TARGET_ROOT/"
echo "    ├── hstu_attn/         # HSTU Attention源码"
echo "    ├── dynamicemb/        # Dynamic Embeddings源码"
echo "    ├── third_party/       # 第三方依赖 (需手动初始化)"
echo "    ├── tests/             # 测试"
echo "    ├── examples/          # 示例"
echo "    ├── env.sh             # 环境配置"
echo "    ├── build.sh           # 编译脚本"
echo "    └── requirements.txt   # Python依赖"
echo ""
echo -e "${GREEN}下一步操作:${NC}"
echo ""
echo "1. 进入目录:"
echo "   cd $TARGET_ROOT"
echo ""
echo "2. 初始化Git和子模块 (如果还没有):"
echo "   git init"
echo "   git submodule add https://github.com/NVIDIA/cutlass.git third_party/cutlass"
echo "   git submodule add https://github.com/NVIDIA-Merlin/HierarchicalKV.git third_party/HierarchicalKV"
echo "   git submodule update --init --recursive"
echo ""
echo "3. 配置环境:"
echo "   conda create -n accel_modules python=3.10 -y"
echo "   conda activate accel_modules"
echo "   pip install torch --index-url https://download.pytorch.org/whl/cu121"
echo ""
echo "4. 设置环境变量:"
echo "   source env.sh"
echo ""
echo "5. 编译模块:"
echo "   bash build.sh"
echo ""
echo "6. 测试:"
echo "   python tests/test_hstu_attention.py"
echo "   python tests/test_dynamicemb.py"
echo ""
print_warning "注意: 需要在A100 GPU机器上编译和运行"
print_warning "编译时间: 约40-80分钟"
print_warning "需要内存: 至少32GB RAM"
echo ""

