from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb

# -------- 相对时间分桶与偏置模块 --------

def make_time_bucketizer(
    num_buckets: int,
    max_exact: int = 24,           # 一天内线性（单位=小时时就是 24 小时）
    unit_seconds: float = 3600.0,  # 把“秒”换算到“小时”（若用分钟则设 60.0）
):
    """
    将非负时间差分桶：
      - 线性段: x < max_exact -> floor(x)
      - 对数段: x >= max_exact，每放大 base 倍桶号 +1
    约定:
      - 正常桶号范围 [0, num_buckets-1]
      - 溢出桶号为 num_buckets（调用方有 ts_w.shape[0] = num_buckets + 1）
    要求:
      - num_buckets > max_exact，否则没有对数段可用
    """
    assert num_buckets > max_exact >= 1, "num_buckets 必须大于 max_exact"
    log_buckets = num_buckets - max_exact
    log_base = 0.301 # follow HSTU

    def bucketize(x_seconds: torch.Tensor) -> torch.Tensor:
        # 1) 换算到目标单位（如小时/分钟），并裁剪到非负
        x = (x_seconds / unit_seconds).clamp(min=0)
        # 2) 线性段
        small = x < max_exact
        lin = x.floor().long().clamp_max(max_exact - 1)  # [0 .. max_exact-1]
        # 3) 对数段（按倍增）
        #    x in [max_exact * base^k, max_exact * base^(k+1)) -> bucket = max_exact + k
        #    注意：对 small 位置用 max_exact 仅为避免 log(0)，不会被最终选择
        z = torch.where(small, torch.as_tensor(float(max_exact), device=x.device, dtype=x.dtype), x)
        k = torch.floor(torch.log(z / max_exact) / log_base).long()  # k >= 0
        k = k.clamp_min(0).clamp_max(log_buckets - 1)               # 限到可用对数桶数
        log_idx = max_exact + k                                      # [max_exact .. num_buckets-1]
        # 4) 合并线性/对数段
        idx = torch.where(small, lin, log_idx)
        # 5) 非有限值 -> 溢出桶
        idx = torch.where(torch.isfinite(x), idx, torch.full_like(idx, num_buckets))
        # 6) 最终裁剪到 [0 .. num_buckets]（含溢出桶）
        return idx.clamp_(0, num_buckets)

    return bucketize


class SeparatedRelativeTimeAndPositionBias(torch.nn.Module):
    """
    生成两类 bias（logits）：
      - rel_pos_bias: [1, S, S]，仅由相对位置决定
      - rel_ts_bias:  [B, S, S]，由 t_{i+1} - t_j 的“时间桶”决定
    """
    def __init__(self, max_seq_len: int, num_buckets: int, bucketization_fn):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.bucketization_fn = bucketization_fn
        # 可学习参数
        self.ts_w = torch.nn.Parameter(torch.empty(num_buckets + 1).normal_(mean=0.0, std=0.02))
        self.pos_w = torch.nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(mean=0.0, std=0.02))

    def forward(self, timestamps: torch.Tensor):
        """
        timestamps: [B, S]（单位例如秒；浮点/整型均可）
        返回：
          rel_pos_bias: [1, S, S]
          rel_ts_bias:  [B, S, S]
        """
        B, S = timestamps.shape

        # 相对位置偏置（Toeplitz），支持 S <= max_seq_len
        pos_vec = self.pos_w[: 2 * S - 1]  # 中心在 S-1
        t = F.pad(pos_vec, [0, S]).repeat(S)
        t = t[..., :-S].reshape(1, S, 3 * S - 2)
        r = (2 * S - 1) // 2
        rel_pos_bias = t[:, :, r:-r]  # [1, S, S]
        rel_pos_bias = rel_pos_bias.expand(B, -1, -1) # 广播到[B, S, S]

        # 相对时间分桶（t_{i+1} - t_j）
        ext = torch.cat([timestamps, timestamps[:, S - 1:S]], dim=1)  # [B, S+1]
        deltas = ext[:, 1:].unsqueeze(2) - ext[:, :-1].unsqueeze(1)   # [B, S, S]
        bucket_ids = torch.clamp(self.bucketization_fn(deltas), 0, self.num_buckets).detach()
        rel_ts_bias = torch.index_select(self.ts_w, 0, bucket_ids.view(-1)).view(B, S, S)  # [B, S, S]
        return rel_pos_bias, rel_ts_bias


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(3 * hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None, rel_pos_bias=None, rel_ts_bias=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            ) # [B,H,S,D]
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        multi_ctx = [attn_output]
        if rel_pos_bias is not None:
            rel_pos_bias = rel_pos_bias * attn_mask
            pos_ctx = torch.matmul(rel_pos_bias.unsqueeze(1), V)  # [B,1,S,S] @ [B,H,S,D] -> [B,H,S,D]
            multi_ctx.append(pos_ctx)

        if rel_ts_bias is not None:
            rel_ts_bias = rel_ts_bias * attn_mask
            ts_ctx = torch.matmul(rel_ts_bias.unsqueeze(1), V)  # [B,1,S,S] @ [B,H,S,D] -> [B,H,S,D]            multi_ctx.append(pos_ctx)
            multi_ctx.append(ts_ctx)

        multi_ctx = torch.cat(multi_ctx, dim=-1)  # [B,H,S,3D]

        # reshape回原来的格式
        multi_ctx = multi_ctx.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units * 3)

        output = self.out_linear(multi_ctx)

        return output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, expansion_factor=2):
        """
        使用 SwiGLU 激活函数的 Point-wise Feed-Forward Network。
        Args:
            hidden_units: 输入和输出的维度。
            dropout_rate: Dropout 概率。
            expansion_factor: 中间层维度的扩展因子。
        """
        super(PointWiseFeedForward, self).__init__()
        self.hidden_units = hidden_units
        self.expansion_factor = expansion_factor
        # 扩展维度，用于门控和值计算
        self.intermediate_dim = expansion_factor * hidden_units

        # 单个线性层，输出维度为 2 * hidden_units (如果 expansion_factor=2)
        # 输出将被分为两部分: 一部分用于门控 (gate), 一部分用于值 (value)
        self.linear1 = torch.nn.Linear(hidden_units, self.intermediate_dim)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)

        # 第二个线性层，将值部分投影回原始维度
        self.linear2 = torch.nn.Linear(self.intermediate_dim // 2, hidden_units)  # 输入是 hidden_units (value 部分)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

        # Swish/SiLU 激活函数，用于门控
        self.swish = torch.nn.SiLU()  # 或者 F.silu

    def forward(self, inputs):
        """
        Args:
            inputs: 输入张量，形状为 (batch_size, seq_len, hidden_units)
        Returns:
            outputs: 输出张量，形状为 (batch_size, seq_len, hidden_units)
        """
        # inputs: [B, S, H]

        x = self.dropout1(self.linear1(inputs))  # [B, S, expansion_factor * H]
        gate, value = x.chunk(2, dim=-1)  # [B, S, H], [B, S, H]
        # SwiGLU: Swish(gate) * value
        activated_gate = self.swish(gate)  # [B, S, H]
        swiglu_output = activated_gate * value  # [B, S, H]
        outputs = self.dropout2(self.linear2(swiglu_output))  # [B, S, H]
        return outputs


class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.embedding_dim, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.embedding_dim, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        userdim = args.embedding_dim * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        itemdim = (
            args.embedding_dim * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.embedding_dim * len(self.ITEM_EMB_FEAT)
        )

        self.userdnn = torch.nn.Sequential(torch.nn.Linear(userdim, args.hidden_units),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(args.hidden_units, args.hidden_units))
        self.itemdnn = torch.nn.Sequential(torch.nn.Linear(itemdim, args.hidden_units),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(args.hidden_units, args.hidden_units))


        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.embedding_dim)

        # -------- 相对 bias（位置 + 时间）--------
        self.ts_num_buckets = getattr(args, "ts_num_buckets", 128)
        self.ts_max_exact = getattr(args, "ts_bucket_max_exact", 16)

        self.time_bucketizer = make_time_bucketizer(
            num_buckets=self.ts_num_buckets,
            max_exact=self.ts_max_exact,
        )
        self.rel_bias = SeparatedRelativeTimeAndPositionBias(
            max_seq_len=self.maxlen + 1,
            num_buckets=self.ts_num_buckets,
            bucketization_fn=self.time_bucketizer,
        )

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]

            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        # pre-compute embedding
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # batch-process all feature types
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]

        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )

        # batch-process each feature type
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)

                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            # collect all data to numpy, then batch-convert
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])

            # pre-allocate tensor
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)

            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]

            # batch-convert and transfer to GPU
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        # merge features
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = self.itemdnn(all_item_emb)
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = self.userdnn(all_user_emb)
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb

        seqs_emb = F.normalize(seqs_emb, dim=-1)
        return seqs_emb

    def _make_rel_bias(self, seq_ts: torch.Tensor):
        """
        计算相对 bias（logits）
        返回：
          rel_pos_bias: [1,S,S]
          rel_ts_bias:  [B,S,S]
        """
        ts = seq_ts.to(self.dev)
        if ts.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            ts = ts.float()
        rel_pos_bias, rel_ts_bias = self.rel_bias(ts)
        return rel_pos_bias.to(ts.dtype), rel_ts_bias.to(ts.dtype)

    def log2feats(self, log_seqs, mask, seq_feature, seq_ts):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs *= self.user_emb.embedding_dim**0.5
        seqs = self.emb_dropout(seqs)

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        rel_pos_bias, rel_ts_bias = self._make_rel_bias(seq_ts)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask, rel_pos_bias=rel_pos_bias, rel_ts_bias=rel_ts_bias)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask, rel_pos_bias=rel_pos_bias, rel_ts_bias=rel_ts_bias)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = F.normalize(seqs, dim=-1)

        return log_feats

    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature, seq_ts
    ):
        """
        训练时调用，计算正负样本的logits

        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list，每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list，每个元素为当前时刻的特征字典

        Returns:
            pos_logits: 正样本logits，形状为 [batch_size, maxlen]
            neg_logits: 负样本logits，形状为 [batch_size, maxlen]
        """
        log_feats = self.log2feats(user_item, mask, seq_feature, seq_ts)

        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        return pos_embs, neg_embs, log_feats

    def predict(self, log_seqs, seq_feature, mask, seq_ts):
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        log_feats = self.log2feats(log_seqs, mask, seq_feature, seq_ts)

        final_feat = log_feats[:, -1, :]

        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)

            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))
