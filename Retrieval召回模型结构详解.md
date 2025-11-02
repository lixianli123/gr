# 🔍 Retrieval召回模型结构详解

## 📋 目录
1. [模型整体结构](#模型整体结构)
2. [与Ranking模型的区别](#与ranking模型的区别)
3. [代码层级结构](#代码层级结构)
4. [核心类定义](#核心类定义)
5. [前向传播流程](#前向传播流程)
6. [配置方式](#配置方式)
7. [双塔结构详解](#双塔结构详解)

---

## 模型整体结构

### 架构图

```
输入数据 (RetrievalBatch)
    ↓
┌─────────────────────────────────────────────────────────┐
│              RetrievalGR 模型 (双塔结构)                 │
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
│  │       * HSTU Attention                          │   │
│  │     - Postprocessing (提取item embeddings)     │   │
│  └─────────────────────────────────────────────────┘   │
│                        ↓                                 │
│              Split为两部分 (双塔)                         │
│                        ↓                                 │
│  ┌─────────────────────┬─────────────────────────┐     │
│  │  Query Tower        │  Item Tower             │     │
│  │  (历史前n-1个)      │  (监督信号：最后1个)    │     │
│  │  [BS, hidden_size]  │  [BS, hidden_size]      │     │
│  └─────────────────────┴─────────────────────────┘     │
│                        ↓                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  3. L2归一化 (L2NormEmbeddingPostprocessor)    │   │
│  │     normalize(query_emb)                        │   │
│  │     normalize(item_emb)                         │   │
│  └─────────────────────────────────────────────────┘   │
│                        ↓                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  4. 相似度计算 (DotProductSimilarity)          │   │
│  │     scores = query_emb @ item_emb.T             │   │
│  └─────────────────────────────────────────────────┘   │
│                        ↓                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  5. Loss计算 (SampledSoftmaxLoss)              │   │
│  │     - InBatchNegativesSampler (负样本采样)     │   │
│  │     - Softmax with temperature                  │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
    ↓
输出: Similarity Scores + Loss
```

---

## 与Ranking模型的区别

### 对比表

| 特性 | Ranking模型 | Retrieval模型 |
|------|------------|--------------|
| **任务目标** | 预测评分/点击率 | 从候选池中召回topK物品 |
| **输出形式** | Logits (评分类别概率) | Embedding (向量表示) |
| **预测头** | MLP (多层全连接) | 无 (直接使用HSTU输出) |
| **Loss函数** | BCE Loss / Cross Entropy | Sampled Softmax Loss |
| **相似度计算** | 无 | 点积相似度 (Dot Product) |
| **负样本** | 无 | In-Batch Negatives |
| **归一化** | 无 | L2归一化 (必需) |
| **结构** | 单塔 | 双塔 (Query + Item) |
| **推理方式** | 在线计算 | 离线构建索引 + ANN检索 |
| **典型评估指标** | AUC, LogLoss | NDCG@K, HR@K, Recall@K |

### 关键差异点

#### 1. **无MLP预测头**
```python
# Ranking模型有:
self._mlp = MLP(...)
logits = self._mlp(hidden_states)

# Retrieval模型没有:
# 直接使用HSTU Block输出的embedding
pred_item_embeddings = jagged_data.values
```

#### 2. **双塔结构**
```python
# Retrieval模型将序列分为两部分:
# Query Tower: 历史序列前n-1个物品 → 预测
# Item Tower: 最后1个物品 → 监督信号 (正样本)

# 训练时计算相似度: query_emb @ item_emb
# 推理时: query_emb @ all_item_embs (从物品库检索)
```

#### 3. **Sampled Softmax Loss**
```python
# Ranking: BCE Loss
losses = BCE(logits, labels)

# Retrieval: Sampled Softmax Loss
losses = SampledSoftmax(
    query_emb,           # [BS, D]
    positive_item_emb,   # [BS, D] 正样本
    negative_item_embs,  # [BS, N, D] 负样本
)
# Loss = -log(exp(sim(q, pos)) / (exp(sim(q, pos)) + Σexp(sim(q, neg))))
```

---

## 代码层级结构

### 文件组织

```
examples/hstu/
├── pretrain_gr_retrieval.py        # 训练入口
├── movielen_retrieval.gin          # 模型配置文件
├── model/
│   ├── __init__.py                 # get_retrieval_model()
│   ├── base_model.py               # BaseModel基类
│   └── retrieval_gr.py             # ⭐ RetrievalGR核心类
├── modules/
│   ├── embedding.py                # ShardedEmbedding
│   ├── hstu_block.py               # HSTUBlock
│   ├── sampled_softmax_loss.py     # SampledSoftmaxLoss
│   ├── negatives_sampler.py        # InBatchNegativesSampler
│   ├── output_postprocessors.py    # L2NormEmbeddingPostprocessor
│   └── similarity/
│       └── dot_product.py          # DotProductSimilarity
└── configs/
    ├── task_config.py              # RetrievalConfig
    └── hstu_config.py              # HSTUConfig
```

---

## 核心类定义

### 1. RetrievalGR (model/retrieval_gr.py)

这是**召回模型的核心类**，定义了"Embedding → HSTU → Similarity → Loss"结构。

```python
class RetrievalGR(BaseModel):
    """
    Retrieval生成推荐模型 (双塔结构)
    
    结构:
        self._embedding_collection  # Embedding层
        self._hstu_block            # HSTU Block
        self._loss_module           # Sampled Softmax Loss
            ├─ negatives_sampler    # In-Batch负样本采样器
            │   └─ norm_func        # L2归一化
            └─ interaction_module   # 点积相似度
    """
    
    def __init__(
        self,
        hstu_config: HSTUConfig,        # HSTU配置
        task_config: RetrievalConfig,   # Retrieval任务配置
    ):
        super().__init__()
        
        # 检查: Retrieval不支持张量并行
        assert self._tp_size == 1, \
            "RetrievalGR does not support tensor model parallel"
        
        self._embedding_dim = hstu_config.hidden_size  # embedding维度
        
        # 第1层: Embedding层
        self._embedding_collection = ShardedEmbedding(
            task_config.embedding_configs
        )
        
        # 第2层: HSTU Block (与Ranking相同)
        self._hstu_block = HSTUBlock(hstu_config)
        
        # 第3层: Sampled Softmax Loss (核心差异)
        self._loss_module = SampledSoftmaxLoss(
            num_to_sample=task_config.num_negatives,      # 负样本数 128
            softmax_temperature=task_config.temperature,   # 温度系数 0.05
            
            # 负样本采样器 (从Batch内采样)
            negatives_sampler=InBatchNegativesSampler(
                # L2归一化函数
                norm_func=L2NormEmbeddingPostprocessor(
                    embedding_dim=self._embedding_dim,
                    eps=task_config.l2_norm_eps,  # 1e-6
                ),
                dedup_embeddings=True,  # 去重
            ),
            
            # 相似度计算模块 (点积)
            interaction_module=DotProductSimilarity(
                dtype=torch.bfloat16 if hstu_config.bf16 else torch.float16
            ),
        )
```

---

### 2. RetrievalGR的前向传播 (forward)

```python
def forward(self, batch: RetrievalBatch):
    """
    完整的前向传播流程
    
    Args:
        batch (RetrievalBatch): 包含features的批次数据
        
    Returns:
        losses: 损失值
        (losses, logits, supervision_item_ids, seqlen): 用于日志和评估
    """
    # 1. 获取query embedding和item embedding
    (
        jagged_item_logit,         # query embedding [BS, D]
        seqlen_after_preprocessor, # 序列长度信息
        supervision_item_ids,      # 监督物品ID [BS]
        supervision_emb,           # 监督物品embedding [BS, D]
    ) = self.get_logit_and_labels(batch)
    
    # 2. 计算Sampled Softmax Loss
    losses = self._loss_module(
        jagged_item_logit.float(),    # query embedding
        supervision_item_ids,          # 正样本物品ID
        supervision_emb.float(),       # 正样本物品embedding
    )
    # 内部流程:
    #   - 采样负样本 (In-Batch Negatives)
    #   - L2归一化 query_emb 和 item_embs
    #   - 计算相似度 scores = query_emb @ item_embs.T
    #   - Softmax with temperature
    #   - 计算交叉熵
    
    # 3. 返回loss和用于评估的信息
    return losses, (
        losses.detach(),
        jagged_item_logit.detach(),
        supervision_item_ids.detach(),
        seqlen_after_preprocessor,
    )
```

---

### 3. get_logit_and_labels (核心逻辑)

这是**双塔结构最清晰的地方**：

```python
def get_logit_and_labels(self, batch: RetrievalBatch):
    """
    完整的 Embedding → HSTU → Split双塔 流程
    """
    
    # ========================================
    # 第1步: Embedding层
    # ========================================
    embeddings = self._embedding_collection(batch.features)
    # embeddings = {
    #     "contextual": JaggedTensor,
    #     "item": JaggedTensor,
    #     "action": JaggedTensor,
    # }
    
    # ========================================
    # 第2步: HSTU Block
    # ========================================
    jagged_data, seqlen_after_preprocessor = self._hstu_block(
        embeddings=embeddings,
        batch=batch,
    )
    # 输出: 所有item token的hidden states
    pred_item_embeddings = jagged_data.values  # [total_items, D]
    pred_item_seqlen = jagged_data.seqlen      # 每个样本的序列长度
    
    # ========================================
    # 第3步: 获取监督信号 (正样本item embedding)
    # ========================================
    # 从原始embedding表中查询监督物品的embedding
    supervision_item_embeddings = embeddings[
        batch.item_feature_name
    ].values()
    supervision_item_ids = batch.features[
        batch.item_feature_name
    ].values()
    
    # ========================================
    # 第4步: Split双塔
    # ========================================
    # Query Tower: 历史序列前n-1个物品的HSTU输出
    # Item Tower: 最后1个物品的原始embedding (监督信号)
    
    # 计算偏移量: 每个样本保留前n-1个
    shift_pred_item_seqlen_offsets = length_to_complete_offsets(
        torch.clamp(pred_item_seqlen - 1, min=0)
    )
    
    # Split: 前n-1个 vs 最后1个
    first_n_pred_item_embeddings, _ = triton_split_2D_jagged(
        pred_item_embeddings,
        pred_item_max_seqlen,
        offsets_a=shift_pred_item_seqlen_offsets,      # 前n-1个
        offsets_b=pred_item_seqlen_offsets - shift_..., # 最后1个
    )
    
    # 同样split监督信号
    _, last_n_supervision_item_embeddings = triton_split_2D_jagged(
        supervision_item_embeddings, ...
    )
    _, last_n_supervision_item_ids = triton_split_2D_jagged(
        supervision_item_ids.view(-1, 1), ...
    )
    
    # ========================================
    # 返回双塔embedding
    # ========================================
    return (
        first_n_pred_item_embeddings.view(-1, self._embedding_dim),  # Query塔
        seqlen_after_preprocessor,
        last_n_supervision_item_ids.view(-1),                        # 正样本ID
        last_n_supervision_item_embeddings.view(-1, self._embedding_dim), # Item塔
    )
```

---

## 前向传播流程

### 详细数据流

```python
# ============================================
# 输入: RetrievalBatch
# ============================================
batch.features = {
    "contextual": KeyedJaggedTensor,  # 用户特征
    "item": KeyedJaggedTensor,        # 物品序列 [item₁, item₂, ..., itemₙ]
    "action": KeyedJaggedTensor,      # 动作序列
}
# 注意: 没有labels! (Retrieval任务的标签是隐式的)

# ============================================
# 第1层: ShardedEmbedding
# ============================================
embeddings = self._embedding_collection(batch.features)
# 同Ranking模型

# ============================================
# 第2层: HSTUBlock
# ============================================
jagged_data, seqlen = self._hstu_block(
    embeddings=embeddings,
    batch=batch,
)
# 输出: pred_item_embeddings [total_items, hidden_size=256]
# 包含所有item token的embedding

# ============================================
# 第3层: Split双塔
# ============================================
# 训练样本: [item₁, item₂, item₃, ..., itemₙ]
# 
# Query Tower: 使用前n-1个物品预测下一个
#   HSTU输出: [emb₁, emb₂, ..., embₙ₋₁]
#   query_emb = embₙ₋₁ (取最后一个作为query)
#
# Item Tower: 最后1个物品作为监督信号
#   正样本: itemₙ 的embedding
#   负样本: 从Batch内其他样本的itemₙ采样

# Split操作:
first_n_pred_item_embeddings: [BS*(n-1), D]  # 前n-1个
last_n_supervision_item_embeddings: [BS, D]   # 最后1个

# 实际上取最后一个作为query:
query_emb = first_n_pred_item_embeddings[-1::n-1]  # [BS, D]

# ============================================
# 第4层: L2归一化
# ============================================
# 在SampledSoftmaxLoss内部自动执行
query_emb = normalize(query_emb, dim=-1)              # [BS, D]
positive_item_emb = normalize(positive_item_emb, dim=-1)  # [BS, D]

# ============================================
# 第5层: 负样本采样
# ============================================
# InBatchNegativesSampler从Batch内采样
# negative_item_embs = [item_emb_1, item_emb_2, ..., item_emb_128]
# shape: [BS, num_negatives=128, D]

# ============================================
# 第6层: 相似度计算
# ============================================
# 正样本相似度
pos_scores = query_emb * positive_item_emb  # [BS, D]
pos_scores = pos_scores.sum(dim=-1)        # [BS]

# 负样本相似度
neg_scores = query_emb @ negative_item_embs.T  # [BS, num_negatives]

# 合并
all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
# shape: [BS, 1+num_negatives]

# ============================================
# 第7层: Softmax with temperature
# ============================================
all_scores = all_scores / temperature  # temperature=0.05
probs = softmax(all_scores, dim=-1)    # [BS, 1+num_negatives]

# ============================================
# 第8层: Loss计算
# ============================================
# 正样本的标签是0 (第一个位置)
labels = torch.zeros(BS, dtype=torch.long)

# 交叉熵
loss = CrossEntropy(probs, labels)
# = -log(probs[:, 0])  # 最大化正样本的概率
```

---

## 配置方式

### 1. Gin配置文件 (movielen_retrieval.gin)

```python
# ========================================
# 网络结构配置
# ========================================
NetworkArgs.dtype_str = "bfloat16"
NetworkArgs.num_layers = 4               # HSTU层数 (比Ranking多)
NetworkArgs.num_attention_heads = 4
NetworkArgs.hidden_size = 256            # embedding维度 (必须一致)
NetworkArgs.kv_channels = 64
NetworkArgs.is_causal = True             # 使用因果mask

# ========================================
# Retrieval任务配置
# ========================================
RetrievalArgs.num_negatives = 128        # 负样本数 ← 关键参数
RetrievalArgs.temperature = 0.05         # Softmax温度 ← 控制分布平滑度
RetrievalArgs.l2_norm_eps = 1e-6         # L2归一化epsilon
RetrievalArgs.eval_metrics = ("NDCG@10", "NDCG@20", "HR@10")  # 评估指标

# 注意: 没有prediction_head_arch! (Retrieval不需要MLP)
```

### 2. 模型实例化 (pretrain_gr_retrieval.py)

```python
# 第1步: 解析配置文件
gin.parse_config_file(args.gin_config_file)

# 第2步: 创建配置对象
retrieval_args = RetrievalArgs()
network_args = NetworkArgs()

# 第3步: 创建HSTU配置
hstu_config = create_hstu_config(network_args, tp_args)

# 第4步: 创建Retrieval配置
retrieval_config = RetrievalConfig(
    embedding_configs=create_embedding_config(...),
    temperature=0.05,               # Softmax温度
    l2_norm_eps=1e-6,               # L2归一化
    num_negatives=128,              # 负样本数
    eval_metrics=("NDCG@10", "HR@10"),
)

# 第5步: 实例化模型
model = get_retrieval_model(
    hstu_config=hstu_config,
    task_config=retrieval_config,
)
# → 返回 RetrievalGR 实例
# → 内部自动创建: Embedding + HSTU Block + Sampled Softmax Loss
```

---

## 双塔结构详解

### 训练阶段

```python
# 输入序列: [item₁, item₂, item₃, item₄, item₅]
#
# 经过HSTU Block:
#   output: [h₁, h₂, h₃, h₄, h₅]  (hidden states)
#
# 双塔split:
#   Query Tower:  [h₁, h₂, h₃, h₄]  → 取h₄作为query_emb
#   Item Tower:   原始item₅的embedding → positive_item_emb
#
# 训练目标: 让query_emb接近positive_item_emb
```

### 为什么这样设计？

#### 1. **Query Tower使用HSTU输出**
- **原因**: HSTU建模了序列依赖，包含了历史行为信息
- **优势**: query_emb = f(item₁, item₂, ..., itemₙ₋₁) 包含丰富上下文

#### 2. **Item Tower使用原始Embedding**
- **原因**: 推理时需要为所有物品构建embedding索引
- **挑战**: 如果Item Tower也用HSTU，每个物品的表示会依赖上下文，无法预先计算
- **解决**: 使用原始embedding，可以离线构建固定的物品索引

### 推理阶段

```python
# 第1步: 离线构建物品索引
all_item_embs = embedding_table["item"]  # [num_items, D]
all_item_embs = normalize(all_item_embs, dim=-1)  # L2归一化
# 存入向量数据库 (如Faiss, HNSW)

# 第2步: 在线计算query embedding
user_history = [item₁, item₂, item₃, item₄]
query_emb = hstu_block(user_history)  # [1, D]
query_emb = normalize(query_emb, dim=-1)

# 第3步: ANN检索topK
scores = query_emb @ all_item_embs.T  # [1, num_items]
topk_indices = scores.topk(k=100).indices  # 召回top-100

# 第4步: 返回召回结果
recommended_items = [item_ids[i] for i in topk_indices]
```

---

## 核心组件详解

### 1. SampledSoftmaxLoss

```python
class SampledSoftmaxLoss(nn.Module):
    """
    带负样本采样的Softmax Loss
    
    公式:
        Loss = -log(exp(sim(q, pos)) / Z)
        Z = exp(sim(q, pos)) + Σᵢ exp(sim(q, negᵢ))
    
    作用: 让query_emb接近positive_item_emb，远离negative_item_embs
    """
    
    def forward(
        self,
        query_emb,           # [BS, D]
        positive_item_ids,   # [BS]
        positive_item_emb,   # [BS, D]
    ):
        # 1. 采样负样本
        negative_item_embs = self.negatives_sampler(
            positive_item_ids,
            positive_item_emb,
        )  # [BS, num_negatives, D]
        
        # 2. L2归一化
        query_emb = self.norm_func(query_emb)
        positive_item_emb = self.norm_func(positive_item_emb)
        negative_item_embs = self.norm_func(negative_item_embs)
        
        # 3. 计算相似度
        pos_scores = (query_emb * positive_item_emb).sum(-1)  # [BS]
        neg_scores = query_emb @ negative_item_embs.T  # [BS, num_negatives]
        
        # 4. Softmax with temperature
        all_scores = torch.cat([
            pos_scores.unsqueeze(1),
            neg_scores
        ], dim=1) / self.temperature  # [BS, 1+num_negatives]
        
        # 5. 交叉熵
        labels = torch.zeros(BS, dtype=torch.long)  # 正样本在第0位
        loss = F.cross_entropy(all_scores, labels)
        
        return loss
```

### 2. InBatchNegativesSampler

```python
class InBatchNegativesSampler(nn.Module):
    """
    从Batch内采样负样本
    
    优势:
        1. 无需额外采样开销
        2. 负样本数随batch size自动增长
        3. 动态负样本，更有效
    
    策略: 
        - 对于样本i，Batch内其他样本的正样本都是i的负样本
        - 自动去重 (避免采样到相同物品)
    """
    
    def forward(
        self,
        positive_item_ids,   # [BS]
        positive_item_emb,   # [BS, D]
    ):
        # 1. Batch内所有正样本都可以作为负样本
        all_candidate_embs = positive_item_emb  # [BS, D]
        
        # 2. 去重 (可选)
        if self.dedup_embeddings:
            unique_item_ids, inverse_indices = torch.unique(
                positive_item_ids, return_inverse=True
            )
            all_candidate_embs = positive_item_emb[inverse_indices]
        
        # 3. 采样 (随机选择num_to_sample个)
        # 实际实现中直接使用所有Batch内样本作为负样本
        negative_embs = all_candidate_embs  # [BS, D]
        
        return negative_embs
```

### 3. L2NormEmbeddingPostprocessor

```python
class L2NormEmbeddingPostprocessor(nn.Module):
    """
    L2归一化
    
    为什么需要?
        1. 让相似度只关注方向，不关注长度
        2. 提高训练稳定性
        3. 使所有embedding在单位球面上
    
    公式:
        x_normalized = x / (||x||₂ + eps)
    """
    
    def forward(self, embeddings):
        # [BS, D] → [BS, D]
        return F.normalize(embeddings, p=2, dim=-1, eps=self.eps)
```

### 4. DotProductSimilarity

```python
class DotProductSimilarity(nn.Module):
    """
    点积相似度
    
    公式:
        sim(x, y) = x · y = Σᵢ xᵢyᵢ
    
    特点:
        - 归一化后等价于余弦相似度
        - 计算高效
        - GPU友好
    """
    
    def forward(self, query_emb, item_embs):
        # query_emb: [BS, D]
        # item_embs: [BS, N, D] or [N, D]
        
        # 点积
        scores = torch.matmul(query_emb, item_embs.T)  # [BS, N]
        
        return scores
```

---

## 实际例子：MovieLens-20M Retrieval

### 数据流示例

```python
# 输入样本
user_id = 1
movie_sequence = [924, 919, 2683, 1584, 1079]  # 5部电影
rating_sequence = [6, 6, 6, 7, 5]

# 训练目标: 根据前4部电影 [924, 919, 2683, 1584]
#           预测第5部电影 [1079]

# ======================================
# 第1步: Embedding
# ======================================
movie_embs = embedding_table["movie"][[924, 919, 2683, 1584, 1079]]
# shape: [5, 256]

# ======================================
# 第2步: HSTU Block
# ======================================
# 输入: [emb₁, emb₂, emb₃, emb₄, emb₅]
# 输出: [h₁, h₂, h₃, h₄, h₅]
hstu_output = hstu_block(movie_embs)  # [5, 256]

# ======================================
# 第3步: Split双塔
# ======================================
# Query Tower: 前4个的HSTU输出
query_emb = hstu_output[3]  # h₄, shape: [256]
# 含义: 基于前4部电影的偏好表示

# Item Tower: 第5个的原始embedding
positive_item_emb = embedding_table["movie"][1079]  # [256]
# 含义: 电影1079的表示

# ======================================
# 第4步: 负样本采样
# ======================================
# 从Batch内其他样本的目标物品采样128个
# negative_item_embs: [128, 256]

# ======================================
# 第5步: L2归一化
# ======================================
query_emb = normalize(query_emb)
positive_item_emb = normalize(positive_item_emb)
negative_item_embs = normalize(negative_item_embs)

# ======================================
# 第6步: 相似度计算
# ======================================
pos_score = query_emb @ positive_item_emb  # 标量
neg_scores = query_emb @ negative_item_embs.T  # [128]

all_scores = [pos_score, neg_scores]  # [129]

# ======================================
# 第7步: Softmax Loss
# ======================================
# 目标: pos_score > neg_scores
probs = softmax(all_scores / 0.05)
loss = -log(probs[0])  # 最大化正样本概率
```

### 推理示例

```python
# 场景: 用户1的新session，看过[924, 919, 2683, 1584]
# 任务: 召回100部可能感兴趣的电影

# 第1步: 计算query embedding
user_history = [924, 919, 2683, 1584]
query_emb = hstu_block(user_history)  # [256]
query_emb = normalize(query_emb)

# 第2步: 从物品库检索 (假设有26744部电影)
all_movie_embs = embedding_table["movie"]  # [26744, 256]
all_movie_embs = normalize(all_movie_embs)

scores = query_emb @ all_movie_embs.T  # [26744]

# 第3步: TopK召回
top100_indices = scores.topk(k=100).indices
recommended_movies = movie_ids[top100_indices]

# 结果: [1079, 2959, 337, ...]  (可能包含真实看过的1079)
```

---

## 关键参数

| 参数 | 配置位置 | 示例值 | 说明 |
|------|----------|--------|------|
| `num_layers` | NetworkArgs | 4 | HSTU层数 (Retrieval通常比Ranking多) |
| `hidden_size` | NetworkArgs | 256 | Embedding维度 (必须与embedding表一致) |
| `num_negatives` | RetrievalArgs | 128 | 负样本数 (影响训练质量和速度) |
| `temperature` | RetrievalArgs | 0.05 | Softmax温度 (越小越陡峭) |
| `l2_norm_eps` | RetrievalArgs | 1e-6 | L2归一化epsilon (数值稳定性) |
| `is_causal` | NetworkArgs | True | 是否使用因果mask |

---

## 训练 vs 推理

### 训练阶段

```python
# 数据: [user_history, target_item]
# 流程:
#   1. HSTU编码user_history → query_emb
#   2. 查表获取target_item_emb (正样本)
#   3. 从Batch采样负样本
#   4. 计算Sampled Softmax Loss
#   5. 反向传播更新参数

# 优化目标:
#   max sim(query_emb, positive_item_emb)
#   min sim(query_emb, negative_item_embs)
```

### 推理阶段

```python
# 离线阶段:
#   1. 为所有物品构建embedding索引
#   all_item_embs = embedding_table["item"]
#   2. 存入向量数据库 (Faiss, HNSW)

# 在线阶段:
#   1. 用户请求到达
#   2. HSTU编码user_history → query_emb
#   3. ANN检索topK最相似物品
#   4. 返回召回结果 (可能进入精排)

# 性能优化:
#   - 使用GPU加速HSTU推理
#   - 使用高效ANN库 (Faiss GPU)
#   - Batch推理 (多个用户并行)
```

---

## 总结

### 模型结构在哪里定义？

1. **核心定义**: `model/retrieval_gr.py` 的 `RetrievalGR.__init__()`
   ```python
   self._embedding_collection = ShardedEmbedding(...)
   self._hstu_block = HSTUBlock(...)
   self._loss_module = SampledSoftmaxLoss(...)  # ← 核心差异
   # 注意: 没有MLP!
   ```

2. **前向流程**: `model/retrieval_gr.py` 的 `get_logit_and_labels()`
   ```python
   embeddings = self._embedding_collection(batch.features)
   hidden_states = self._hstu_block(embeddings, batch)
   query_emb, positive_item_emb = split_towers(hidden_states)  # ← 双塔
   ```

3. **结构配置**: `movielen_retrieval.gin`
   ```python
   NetworkArgs.num_layers = 4
   NetworkArgs.hidden_size = 256
   RetrievalArgs.num_negatives = 128  # ← Retrieval特有
   ```

### 关键文件速查

| 内容 | 文件路径 | 关键代码 |
|------|----------|----------|
| **模型总体定义** | `model/retrieval_gr.py` | `RetrievalGR.__init__()` (46-82行) |
| **前向传播流程** | `model/retrieval_gr.py` | `get_logit_and_labels()` (104-160行) |
| **双塔Split** | `model/retrieval_gr.py` | `triton_split_2D_jagged()` (136-154行) |
| **Sampled Softmax Loss** | `modules/sampled_softmax_loss.py` | `SampledSoftmaxLoss` |
| **负样本采样器** | `modules/negatives_sampler.py` | `InBatchNegativesSampler` |
| **L2归一化** | `modules/output_postprocessors.py` | `L2NormEmbeddingPostprocessor` |
| **配置文件** | `movielen_retrieval.gin` | gin参数 |
| **训练入口** | `pretrain_gr_retrieval.py` | `main()` |

### Retrieval vs Ranking 快速对比

| 特性 | Retrieval | Ranking |
|------|-----------|---------|
| 输出 | Embedding | Logits |
| 预测头 | 无 | MLP |
| Loss | Sampled Softmax | BCE/CE |
| 结构 | 双塔 | 单塔 |
| 负样本 | In-Batch | 无 |
| 归一化 | L2 Norm | 无 |
| 评估 | NDCG, HR | AUC |

---

**现在您应该完全理解Retrieval召回模型的结构了！** 🎉

核心要点：
1. **无MLP**: 直接使用HSTU输出的embedding
2. **双塔结构**: Query Tower (HSTU) + Item Tower (原始embedding)
3. **Sampled Softmax Loss**: 对比学习，拉近正样本，推开负样本
4. **In-Batch Negatives**: 高效的负样本采样策略
5. **L2归一化**: 让相似度计算更稳定

需要进一步解释某个部分吗？例如为什么Retrieval需要L2归一化？🔍

