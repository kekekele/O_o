# 原始序列特征到模型输入表示的融合过程分析

## 1. 输入数据形态

### 1.1 单用户原始序列记录
- 数据源来自 seq.jsonl，每行是一个用户序列。
- 序列中的单条记录形态可概括为：
	- user_id
	- item_id
	- user_feat（字典）
	- item_feat（字典）
	- action_type（曝光或点击）
	- timestamp

### 1.2 训练样本在 __getitem__ 的组装
- 训练集使用 MyDataset.__getitem__ 将用户行为序列转成长度 S=maxlen+1 的定长窗口。
- 返回主干字段（按位置对齐）：
	- seq: 当前 token id，形状 [S]
	- pos: 下一 token 正样本 item id，形状 [S]
	- neg: 负样本 item id，形状 [S]
	- token_type: 当前 token 类型（0=pad, 1=item, 2=user），形状 [S]
	- next_token_type: 下一 token 类型，形状 [S]
	- next_action_type: 下一 token 动作类型，形状 [S]
	- seq_ts: 与 seq 对齐的时间戳，形状 [S]
	- seq_feat, pos_feat, neg_feat: 对齐位置的特征字典数组，形状 [S]
- 为 SSL 额外返回：
	- neg_feat_ssl1, neg_feat_ssl2
	- neg_ssl1, neg_ssl2

### 1.3 collate_fn 后的批次张量
- MyDataset.collate_fn 将上面的每个字段堆叠成 batch：
	- seq, pos, neg, token_type, next_token_type, next_action_type, seq_ts 都是 [B, S]
	- seq_feat, pos_feat, neg_feat 被张量化为分组字典：
		- item_sparse, user_sparse, item_array, user_array, item_continual, user_continual, item_emb
- 重点是 item_emb 位置掩蔽：在 token_type 不是 item 的位置，item_emb 会被置 0，避免用户位置混入 item 多模态向量。

### 1.4 推理输入形态
- MyTestDataset 返回：
	- seq: [S]
	- token_type: [S]
	- seq_feat: 字典数组
	- user_id
	- seq_ts: [S]
- 推理 collate 后进入模型 predict：
	- seq, token_type, seq_feat, seq_ts

## 2. feat2emb：静态特征融合

### 2.1 两条融合路径
- include_user=True（序列主干路径）
	- 同时融合 user 侧与 item 侧静态特征。
	- 最终走 user_item_dnn。
- include_user=False（候选 item 路径）
	- 只融合 item 侧特征。
	- 最终走 item_dnn。

### 2.2 特征拼接构成
- item 侧来源：
	- item_id embedding
	- item_sparse embedding
	- item_array embedding 后按数组维求和
	- item_continual 数值特征
	- item_emb（多模态）先经线性变换到统一维度
- user 侧来源（仅 include_user=True）：
	- user_id embedding
	- user_sparse embedding
	- user_array embedding 后按数组维求和
	- user_continual

### 2.3 时间离散特征在 feat2emb 内的注入
- include_user=True 且提供 seq_ts 时，会把时间戳离散成：
	- hour_idx（1..24）
	- dow_idx（1..7）
	- weekend_idx（1/2）
- 这三个离散特征对应 hour_emb, dow_emb, weekend_emb，并且只在 item 位置保留，用户位置置 0。

### 2.4 DNN 编码器输出
- 拼接后向量经 FeatureInteractionEncoder，输出统一隐藏维。
- 序列主干输出作为后续时序建模输入。
- 正负样本 item 输出用于训练对比学习头。

### 2.5 当前排障中暴露的关键点
- hour_emb 异常不是索引越界触发，而是参数被污染后在前向暴露。
- 现有调试检查会在 feat2emb 的多个子阶段直接抛出具体坏点名称，便于精确定位。

## 3. log2feats：动态时序融合

### 3.1 输入与预处理
- 输入是 include_user=True 的 feat2emb 输出，形状 [B, S, H]。
- 先乘以 sqrt(H) 做尺度调整。

### 3.2 绝对时间编码
- 使用 FourierTimeEncoding 把 seq_ts 映射为多频正余弦，再线性投影到 H。
- 仅在有效 token 位置加入（padding 位置被掩蔽）。

### 3.3 注意力掩码与相对时间偏置
- 构造因果下三角 attention_mask，并结合有效位掩码。
- RelativeTimeBias 根据时间差分桶生成 rel_ts_bias。

### 3.4 HSTU 层堆叠融合
- 多层 HSTU 对序列表征做动态建模。
- 每层包含：
	- Q/K/V/U 投影
	- RoPE 旋转位置编码
	- 注意力项与相对时间项的融合
	- 残差加 RMSNorm

### 3.5 最终归一化
- 经过 F.normalize 得到 log_feats（[B, S, H]），作为序列语义表示。
- 训练时主要在 item 位置抽取 log_feats 参与 InfoNCE。

## 4. 与训练/预测的衔接

### 4.1 训练阶段衔接
- 训练主循环调用 model(seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts)。
- model.forward 内部：
	- log_feats = log2feats(序列主干)
	- pos_embs = feat2emb(pos, pos_feat, include_user=False)
	- neg_embs = feat2emb(neg, neg_feat, include_user=False)
- 然后进入：
	- 主损失 InfoNCE（使用 log_feats 与 pos/neg 表示）
	- 可选 SSL 损失（基于 neg 两视图特征）

### 4.2 验证阶段衔接
- evaluate_hr_ndcg10_and_score 复用相同 forward 输出。
- 在点击 item 位置计算候选打分排序，产出 HR@10、NDCG@10 与综合 score。

### 4.3 推理阶段衔接
- infer 侧通过 test_loader 提供 seq, token_type, seq_feat, seq_ts。
- 调用 model.predict：
	- 内部执行 log2feats
	- 取最后位置向量 final_feat 作为用户检索表示
- 候选 item 向量来自 feat2emb(include_user=False) 离线保存。

## 5. 关键设计总结

### 5.1 融合思想
- 静态融合和动态融合解耦：
	- feat2emb 负责多源特征对齐到统一语义空间
	- log2feats 负责时序依赖与时间结构建模

### 5.2 时间信息的双通道建模
- 离散时间通道：hour/dow/weekend 进入 item 侧 embedding
- 连续时间通道：FourierTimeEncoding 与 RelativeTimeBias
- 两者互补，既保留可解释离散模式，也保留连续周期信息。

### 5.3 训练目标的角色分工
- 主任务 InfoNCE 用序列表示拉近下一个正样本并区分负样本。
- SSL 分支通过负样本两视图提供额外一致性约束。

### 5.4 稳定性经验（结合当前排障）
- 前向非有限多数先在参数污染处暴露（如 hour_emb）。
- 需要同时处理三层防护：
	- 非有限 loss/grad 跳步
	- 非有限参数修复
	- 修复参数对应 optimizer state 重置
- NPU 507015 属于运行时致命故障，应安全停止并重启续训，而不是进程内硬扛。

