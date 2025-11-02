# 🏗️ Ranking模型结构详解

## 📋 目录
1. [模型整体结构](#模型整体结构)
2. [代码层级结构](#代码层级结构)
3. [核心类定义](#核心类定义)
4. [前向传播流程](#前向传播流程)
5. [配置方式](#配置方式)

---

## 模型整体结构

### 架构图

```
输入数据 (RankingBatch)
    ↓
┌─────────────────────────────────────────────────────────┐
│              RankingGR 模型                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  1. Embedding层 (ShardedEmbedding)             │   │
│  │     - Contextual Embedding (用户特征)          │   │
│  │     - Item Embedding (物品特征)                │   │
│  │     - Action Embedding (动作特征)              │   │
│  └─────────────────────────────────────────────────┘   │
│                        ↓                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  2. HSTU Block (HSTUBlock)                     │   │
│  │     - Preprocessing (序列拼接、位置编码)       │   │
│  │     - Multi-layer HSTU Attention               │   │
│  │       * FusedHSTULayer (CUTLASS加速)           │   │
│  │       * LayerNorm + Linear + SiLU              │   │
│  │       * HSTU Attention (自定义attention)       │   │
│  │     - Postprocessing (候选物品筛选)            │   │
│  └─────────────────────────────────────────────────┘   │
│                        ↓                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  3. MLP预测头 (MLP)                            │   │
│  │     - 多层全连接网络                            │   │
│  │     - ReLU/GELU激活                            │   │
│  │     - Dropout                                   │   │
│  └─────────────────────────────────────────────────┘   │
│                        ↓                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  4. Loss计算 (MultiTaskLossModule)             │   │
│  │     - BCE Loss (多任务)                         │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
    ↓
输出: Logits (预测评分) + Loss
```

---

## 代码层级结构

### 文件组织

```
examples/hstu/
├── pretrain_gr_ranking.py          # 训练入口
├── movielen_ranking.gin            # 模型配置文件
├── model/
│   ├── __init__.py                 # get_ranking_model()
│   ├── base_model.py               # BaseModel基类
│   └── ranking_gr.py               # ⭐ RankingGR核心类
├── modules/
│   ├── embedding.py                # ShardedEmbedding
│   ├── hstu_block.py               # HSTUBlock
│   ├── hstu_layer.py               # FusedHSTULayer
│   ├── hstu_attention.py           # HSTU Attention
│   └── mlp.py                      # MLP预测头
└── configs/
    ├── task_config.py              # RankingConfig
    └── hstu_config.py              # HSTUConfig
```

---

## 核心类定义

### 1. RankingGR (model/ranking_gr.py)

这是**模型的核心类**，定义了完整的"Embedding → HSTU → MLP → Loss"结构。

```python
class RankingGR(BaseModel):
    """
    Ranking生成推荐模型
    
    结构:
        self._embedding_collection  # Embedding层
        self._hstu_block            # HSTU Block
        self._mlp                   # MLP预测头
        self._loss_module           # Loss计算
        self._metric_module         # 评估指标
    """
    
    def __init__(
        self,
        hstu_config: HSTUConfig,        # HSTU配置
        task_config: RankingConfig,     # Ranking任务配置
    ):
        super().__init__()
        
        # 第1层: Embedding层
        self._embedding_collection = ShardedEmbedding(
            task_config.embedding_configs
        )
        
        # 第2层: HSTU Block (核心注意力机制)
        self._hstu_block = HSTUBlock(hstu_config)
        
        # 第3层: MLP预测头
        self._mlp = MLP(
            hstu_config.hidden_size,              # 输入维度 (HSTU输出)
            task_config.prediction_head_arch,     # MLP架构 [512, 10]
            task_config.prediction_head_act_type, # 激活函数 'relu'
            task_config.prediction_head_bias,     # 是否使用bias
            device=self._device,
        )
        
        # 第4层: Loss模块
        self._loss_module = MultiTaskLossModule(
            num_classes=task_config.prediction_head_arch[-1],  # 输出类别数
            num_tasks=task_config.num_tasks,                   # 任务数
            reduction="none",
        )
        
        # 评估指标模块
        self._metric_module = get_multi_event_metric_module(...)
```

---

### 2. RankingGR的前向传播 (forward)

```python
def forward(self, batch: RankingBatch):
    """
    完整的前向传播流程
    
    Args:
        batch (RankingBatch): 包含features和labels的批次数据
        
    Returns:
        losses: 损失值
        (losses, logits, labels, seqlen): 用于日志和评估
    """
    # 1. 获取logits和labels
    (
        jagged_item_logit,        # 预测的logits
        seqlen_after_preprocessor,# 序列长度信息
        labels,                    # 真实标签
    ) = self.get_logit_and_labels(batch)
    
    # 2. 计算loss
    losses = self._loss_module(
        jagged_item_logit.float(), 
        labels
    )
    
    # 3. 返回loss和用于评估的信息
    return losses, (
        losses.detach(),
        jagged_item_logit.detach(),
        labels.detach(),
        seqlen_after_preprocessor,
    )
```

---

### 3. get_logit_and_labels (核心逻辑)

这是**模型结构体现最清晰的地方**：

```python
def get_logit_and_labels(self, batch: RankingBatch):
    """
    完整的 Embedding → HSTU → MLP 流程
    """
    
    # ========================================
    # 第1步: Embedding层
    # ========================================
    # 输入: batch.features (user_id, item_ids, action_ids)
    # 输出: embeddings 字典 {"contextual": JT, "item": JT, "action": JT}
    embeddings: Dict[str, JaggedTensor] = self._embedding_collection(
        batch.features
    )
    
    # 梯度缩放 (用于模型并行)
    embeddings = self._embedding_collection._maybe_detach(embeddings)
    embeddings = jt_dict_grad_scaling_and_allgather(
        embeddings,
        grad_scaling_factor=self._tp_size,
        parallel_state.get_tensor_model_parallel_group(),
    )
    
    # 数据格式转换 (用于模型并行)
    batch = dmp_batch_to_tp(batch)
    
    # ========================================
    # 第2步: HSTU Block
    # ========================================
    # 输入: embeddings字典 + batch
    # 输出: hidden_states_jagged (JaggedData格式的隐藏状态)
    hidden_states_jagged, seqlen_after_preprocessor = self._hstu_block(
        embeddings=embeddings,
        batch=batch,
    )
    
    # 提取实际的tensor值
    hidden_states = hidden_states_jagged.values  # [total_tokens, hidden_size]
    
    # ========================================
    # 第3步: MLP预测头
    # ========================================
    # 输入: hidden_states [total_tokens, hidden_size=128]
    # 输出: logits [total_tokens, num_classes=10]
    logits = self._mlp(hidden_states)
    
    return logits, seqlen_after_preprocessor, batch.labels
```

---

## 前向传播流程

### 详细数据流

```python
# ============================================
# 输入: RankingBatch
# ============================================
batch.features = {
    "contextual": KeyedJaggedTensor,  # 用户特征 (user_id)
    "item": KeyedJaggedTensor,        # 物品序列 (movie_ids)
    "action": KeyedJaggedTensor,      # 动作序列 (ratings)
}
batch.labels = torch.Tensor           # 标签 (真实评分)

# ============================================
# 第1层: ShardedEmbedding
# ============================================
embeddings = self._embedding_collection(batch.features)
# embeddings = {
#     "contextual": JaggedTensor,  # [batch_size, 1, emb_dim]
#     "item": JaggedTensor,        # [batch_size, seq_len, emb_dim]
#     "action": JaggedTensor,      # [batch_size, seq_len, emb_dim]
# }

# ============================================
# 第2层: HSTUBlock
# ============================================
hidden_states_jagged, seqlen = self._hstu_block(
    embeddings=embeddings,
    batch=batch,
)
# 内部流程:
#   2.1 Preprocessing: 
#       - 拼接contextual + interleaved(item, action)
#       - 添加位置编码
#       输出: [total_tokens, hidden_size]
#
#   2.2 Multi-layer HSTU Attention:
#       for layer in self._attention_layers:  # num_layers次
#           x = FusedHSTULayer(x)
#               - LayerNorm
#               - Linear + SiLU
#               - HSTU Attention (CUTLASS加速)
#               - 残差连接
#
#   2.3 Postprocessing:
#       - 如果有候选物品，筛选候选物品对应的token
#       - 否则返回所有item token
#       输出: JaggedData [total_item_tokens, hidden_size]

# hidden_states = hidden_states_jagged.values
# shape: [total_item_tokens, hidden_size=128]

# ============================================
# 第3层: MLP预测头
# ============================================
logits = self._mlp(hidden_states)
# 内部流程:
#   Linear(128 → 512) → ReLU → Dropout
#   Linear(512 → 10)  → (无激活)
#
# 输出: [total_item_tokens, num_classes=10]

# ============================================
# 第4层: Loss计算
# ============================================
losses = self._loss_module(logits, labels)
# BCE Loss for multi-class classification
# 输出: [total_item_tokens, num_tasks=1]
```

---

## 配置方式

### 1. Gin配置文件 (movielen_ranking.gin)

这是**模型结构的配置入口**：

```python
# ========================================
# 网络结构配置
# ========================================
NetworkArgs.dtype_str = "bfloat16"       # 数据类型
NetworkArgs.num_layers = 1               # HSTU层数 ← HSTU Block有几层
NetworkArgs.num_attention_heads = 4      # 注意力头数
NetworkArgs.hidden_size = 128            # 隐藏层维度 ← HSTU输出维度
NetworkArgs.kv_channels = 128            # K/V维度
NetworkArgs.target_group_size = 1        # Target分组大小

# ========================================
# Ranking任务配置
# ========================================
RankingArgs.prediction_head_arch = [512, 10]  # MLP结构 ← [中间层, 输出层]
RankingArgs.prediction_head_bias = True       # MLP使用bias
RankingArgs.num_tasks = 1                     # 任务数
RankingArgs.eval_metrics = ("AUC",)           # 评估指标

# ========================================
# 优化器配置
# ========================================
OptimizerArgs.optimizer_str = 'adam'
OptimizerArgs.learning_rate = 1e-3

# ========================================
# 并行配置
# ========================================
TensorModelParallelArgs.tensor_model_parallel_size = 1  # A100只支持TP=1
```

### 2. 模型实例化 (pretrain_gr_ranking.py)

```python
# 第1步: 解析配置文件
parser = argparse.ArgumentParser()
parser.add_argument("--gin-config-file", type=str)
args = parser.parse_args()
gin.parse_config_file(args.gin_config_file)  # 读取movielen_ranking.gin

# 第2步: 创建配置对象
ranking_args = RankingArgs()  # 从gin读取
network_args = NetworkArgs()  # 从gin读取

# 第3步: 创建HSTU配置
hstu_config = create_hstu_config(network_args, tp_args)
# 包含:
#   - num_layers=1
#   - num_attention_heads=4
#   - hidden_size=128
#   - kernel_backend=KernelBackend.CUTLASS
#   - hstu_layer_type=HSTULayerType.FUSED

# 第4步: 创建Ranking配置
ranking_config = RankingConfig(
    embedding_configs=create_embedding_configs(...),
    prediction_head_arch=[512, 10],          # ← MLP结构
    prediction_head_act_type="relu",
    prediction_head_bias=True,
    num_tasks=1,
    eval_metrics=("AUC",),
)

# 第5步: 实例化模型
model = get_ranking_model(
    hstu_config=hstu_config,
    task_config=ranking_config,
)
# → 返回 RankingGR 实例
# → 内部自动创建: Embedding + HSTU Block + MLP + Loss
```

---

## 模型结构总结

### 核心组件

| 组件 | 类名 | 作用 | 输入 | 输出 |
|------|------|------|------|------|
| **Embedding层** | `ShardedEmbedding` | 查表获取embedding向量 | KeyedJaggedTensor | Dict[str, JaggedTensor] |
| **HSTU Block** | `HSTUBlock` | 序列建模 (核心) | embeddings + batch | JaggedData (hidden_states) |
| **MLP预测头** | `MLP` | 分类/回归预测 | hidden_states | logits |
| **Loss模块** | `MultiTaskLossModule` | 计算损失 | logits + labels | losses |

### 数据形状变化

```python
# 输入
batch.features["item"]: [batch_size, seq_len] 的item IDs

# ↓ Embedding
embeddings["item"]: [batch_size, seq_len, emb_dim=128]

# ↓ HSTU Block (拼接、Attention)
hidden_states: [total_item_tokens, hidden_size=128]

# ↓ MLP (两层全连接)
# Layer 1: [total_item_tokens, 128] → [total_item_tokens, 512]
# Layer 2: [total_item_tokens, 512] → [total_item_tokens, 10]
logits: [total_item_tokens, 10]

# ↓ Loss
losses: [total_item_tokens, num_tasks=1]
```

### 关键参数

| 参数 | 配置位置 | 示例值 | 说明 |
|------|----------|--------|------|
| `num_layers` | NetworkArgs | 1 | HSTU层数 (有几个FusedHSTULayer) |
| `hidden_size` | NetworkArgs | 128 | HSTU隐藏层维度 |
| `num_attention_heads` | NetworkArgs | 4 | 注意力头数 |
| `prediction_head_arch` | RankingArgs | [512, 10] | MLP结构 (中间层→输出层) |
| `kernel_backend` | NetworkArgs | "cutlass" | CUDA内核后端 (必须cutlass for A100) |

---

## 代码追踪示例

### 如果想修改MLP结构

```python
# 修改配置文件: movielen_ranking.gin
RankingArgs.prediction_head_arch = [256, 512, 10]  # 改成3层MLP

# 代码会自动更新:
# model/ranking_gr.py → __init__()
self._mlp = MLP(
    hstu_config.hidden_size,              # 128
    task_config.prediction_head_arch,     # [256, 512, 10] ← 新配置
    task_config.prediction_head_act_type, # 'relu'
    task_config.prediction_head_bias,     # True
)

# modules/mlp.py → MLP类会根据prediction_head_arch自动构建
# Linear(128 → 256) → ReLU → Dropout
# Linear(256 → 512) → ReLU → Dropout
# Linear(512 → 10)  → (输出层)
```

### 如果想增加HSTU层数

```python
# 修改配置文件: movielen_ranking.gin
NetworkArgs.num_layers = 3  # 从1改到3

# 代码会自动更新:
# modules/hstu_block.py → __init__()
self._attention_layers = torch.nn.ModuleList(
    [FusedHSTULayer(config) for _ in range(self.config.num_layers)]
    #                                        ↑ num_layers=3
)
# 结果: 会创建3个FusedHSTULayer，前向传播时依次经过3层
```

---

## 实际例子：MovieLens-20M Ranking

### 数据流

```python
# 输入样本
user_id = 1
movie_sequence = [924, 919, 2683, 1584, ...]  # 历史观看的电影
rating_sequence = [6, 6, 6, 7, ...]           # 对应的评分

# 第1步: Embedding
user_emb = embedding_table_user[1]            # [128]
movie_embs = embedding_table_movie[movie_seq] # [seq_len, 128]
rating_embs = embedding_table_action[rating_seq] # [seq_len, 128]

# 第2步: HSTU Block
# Preprocessing: 拼接成 [user_emb, movie_emb[0], rating_emb[0], movie_emb[1], ...]
# HSTU Attention: 自注意力机制，学习序列依赖
# Postprocessing: 提取movie token的hidden states
# 输出: [seq_len, 128]

# 第3步: MLP
# 每个movie token经过MLP
# Linear(128 → 512) → ReLU
# Linear(512 → 10)
# 输出: [seq_len, 10]  # 10个评分类别 (0-9，对应原始评分0-4.5)

# 第4步: Loss
# 与真实评分对比，计算BCE Loss
```

---

## 总结

### 模型结构在哪里定义？

1. **核心定义**: `model/ranking_gr.py` 的 `RankingGR.__init__()`
   ```python
   self._embedding_collection = ShardedEmbedding(...)
   self._hstu_block = HSTUBlock(...)
   self._mlp = MLP(...)
   ```

2. **前向流程**: `model/ranking_gr.py` 的 `get_logit_and_labels()`
   ```python
   embeddings = self._embedding_collection(batch.features)
   hidden_states = self._hstu_block(embeddings, batch)
   logits = self._mlp(hidden_states)
   ```

3. **结构配置**: `movielen_ranking.gin`
   ```python
   NetworkArgs.num_layers = 1           # HSTU层数
   NetworkArgs.hidden_size = 128        # 隐藏层维度
   RankingArgs.prediction_head_arch = [512, 10]  # MLP结构
   ```

### 关键代码位置

| 内容 | 文件路径 | 关键代码 |
|------|----------|----------|
| 模型总体定义 | `model/ranking_gr.py` | `class RankingGR` |
| 前向传播流程 | `model/ranking_gr.py` | `get_logit_and_labels()` |
| HSTU Block | `modules/hstu_block.py` | `class HSTUBlock` |
| HSTU Attention | `modules/hstu_attention.py` | `FusedHSTUAttention` |
| MLP预测头 | `modules/mlp.py` | `class MLP` |
| 配置文件 | `movielen_ranking.gin` | gin参数 |
| 训练入口 | `pretrain_gr_ranking.py` | `main()` |

---

**现在您应该清楚地知道模型结构是如何定义和串联的了！** 🎉

- **定义**: 在 `RankingGR.__init__()` 中实例化三个组件
- **串联**: 在 `get_logit_and_labels()` 中依次调用
- **配置**: 在 `.gin` 文件中设置超参数

需要修改模型结构时，可以直接修改gin配置或相应的模块代码。

