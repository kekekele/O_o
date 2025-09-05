from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb

class RotaryEmbedding(torch.nn.Module):
    """
    标准 RoPE（Rotary Positional Embedding）实现。
    将多头中每个 head 的前 rotary_dim 维进行旋转；其余维度保持不变。
    """
    def __init__(self, head_dim: int, rope_base: float = 10000.0, rope_fraction: float = 1.0):
        """
        Args:
            head_dim: 每个注意力头的维度 d
            rope_base: RoPE 的频率基数（常用 10000）
            rope_fraction: 使用前多少比例维度用于旋转（0~1]；默认为 1.0 即全部 head_dim
        """
        super().__init__()
        rotary_dim = int(head_dim * float(rope_fraction))
        # 需要偶数维；若为奇数，则下取偶数；同时至少保留 2 维
        rotary_dim = max(2, rotary_dim - (rotary_dim % 2))
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        # inv_freq: [rotary_dim/2]
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_cos_sin(self, seq_len: int, device, dtype):
        """
        返回：
            cos, sin: [1, 1, S, rotary_dim]，用于广播到 [B,h,S,rotary_dim]
        """
        # 时刻索引 0..S-1
        t = torch.arange(seq_len, device=device, dtype=dtype)  # [S]
        freqs = torch.einsum("s,f->sf", t, self.inv_freq.to(device=device, dtype=dtype))  # [S, rotary_dim/2]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        # 将 [S, d/2] 交错扩展到 [S, d]
        cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, -1)  # [S, rotary_dim]
        sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, -1)  # [S, rotary_dim]
        # 扩展以便广播到 [B,h,S,rotary_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,S,rotary_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1,1,S,rotary_dim]
        return cos, sin

    @staticmethod
    def _rotate_half(x):
        # x: [..., rotary_dim]，把偶/奇维配对做旋转
        x1, x2 = x[..., ::2], x[..., 1::2]     # [..., d/2], [..., d/2]
        x_rot = torch.stack((-x2, x1), dim=-1) # [..., d/2, 2]
        return x_rot.flatten(-2)               # [..., d]

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        """
        对 Q/K 应用 RoPE。q,k: [B,h,S,d]
        仅对前 rotary_dim 维进行旋转；其余维度保持不变。
        """
        B, H, S, D = q.shape
        assert D == self.head_dim, "Q/K head_dim mismatch"
        if self.rotary_dim == 0:
            return q, k

        # 计算 cos/sin
        cos, sin = self._build_cos_sin(S, q.device, q.dtype)  # [1,1,S,rotary_dim]

        def _apply(x):
            x_rot, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]  # [B,h,S,rd], [B,h,S, D-rd]
            x_rotated = x_rot * cos + self._rotate_half(x_rot) * sin
            return torch.cat([x_rotated, x_pass], dim=-1)

        q = _apply(q)
        k = _apply(k)
        return q, k


class RelativeTimeBias(torch.nn.Module):
    """
    生成两类 bias（logits）：
      - rel_pos_bias: [1, S, S]，仅由相对位置决定（Toeplitz）
      - rel_ts_bias:  [B, S, S]，由 t_i - t_j 的“时间桶”决定（仅在 i>=j 且两端有效时生效）
    """
    def __init__(self, max_seq_len: int, num_buckets: int, bucketization_fn):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.bucketization_fn = bucketization_fn
        self.ts_w = torch.nn.Parameter(torch.empty(num_buckets + 1).normal_(mean=0.0, std=0.02))


    def forward(self, timestamps: torch.Tensor, valid_mask: torch.Tensor):
        """
        timestamps: [B, S]（单位例如秒；浮点/整型均可）
        valid_mask: [B, S]（True=有效 token；False=padding）
        返回：
          rel_pos_bias: [1, S, S]
          rel_ts_bias:  [B, S, S]
        """
        B, S = timestamps.shape

        # 成对相对时间：t_i - t_j（因果：i >= j 时才有效；负数截断为 0）
        ti = timestamps.unsqueeze(2)             # [B, S, 1]
        tj = timestamps.unsqueeze(1)             # [B, 1, S]
        deltas = torch.clamp(ti - tj, min=0)     # [B, S, S]

        # 时间差分桶
        bucket_ids = self.bucketization_fn(deltas)  # [B, S, S]，范围 [0..num_buckets]（0=padding桶，num_buckets=溢出）

        # 构造有效对掩码：两端有效，且下三角（i>=j）
        vi = valid_mask.unsqueeze(2)              # [B, S, 1]
        vj = valid_mask.unsqueeze(1)              # [B, 1, S]
        pair_valid = vi & vj                      # [B, S, S]
        tril_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=timestamps.device))
        pair_valid = pair_valid & tril_mask       # 仅保留下三角

        # 无效位置 -> 0（padding 桶）
        diag = torch.eye(S, dtype=torch.bool, device=timestamps.device)
        pair_valid = pair_valid & (~diag)  # 对角线视作无效对 -> 后续会被置为 0 号 padding 桶
        bucket_ids = torch.where(pair_valid, bucket_ids, torch.zeros_like(bucket_ids))

        # 将桶号映射为标量偏置
        rel_ts_bias = self.ts_w[bucket_ids]       # [B, S, S]
        return rel_ts_bias


class FourierTimeEncoding(torch.nn.Module):
    """
    将绝对时间戳（秒）映射为多频正余弦，再线性投到隐藏维。
    特点：参数量小、可外推、对不同时间尺度（小时~月）建模友好。作为 hour、weekday、is_weekday 的补充
    """

    def __init__(
            self,
            hidden_units: int,
            num_frequencies: int = 12,
            min_period_seconds: float = 86400.0,  # 最短周期：1小时=3600.0，使用 8*86400.0 以防止与 hour、weekday、is_weekday 重复
            max_period_seconds: float = 40 * 86400.0, # 最长周期（数据集似乎 40 天内）
    ):
        super().__init__()
        assert num_frequencies >= 1
        assert max_period_seconds > min_period_seconds > 0

        periods = torch.logspace(
            math.log10(min_period_seconds),
            math.log10(max_period_seconds),
            steps=num_frequencies,
        )  # [L]
        self.register_buffer("periods", periods)  # 常数，不参与训练
        self.register_buffer("two_pi", torch.tensor(2.0 * math.pi))
        self.proj = torch.nn.Linear(2 * num_frequencies, hidden_units, bias=False)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        timestamps: [B, S]，单位：秒（int/float 均可）
        return: [B, S, H]
        """
        t = timestamps.to(torch.float32)  # 数值稳定
        # angle: [B,S,L]
        angle = (t.unsqueeze(-1) / self.periods) * self.two_pi
        s = torch.sin(angle)
        c = torch.cos(angle)
        feats = torch.cat([s, c], dim=-1)  # [B,S,2L]
        out = self.proj(feats)             # [B,S,H]
        return out


class HSTU(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate,
                 rope_fraction: float = 1.0,
                 rope_base: float = 10000.0):
        super(HSTU, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        self.rms_norm = torch.nn.RMSNorm(hidden_units, eps=1e-8)

        # FIX: 输出维度改为 6H（U:3H, V:H, Q:H, K:H）
        self.qkvu_linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_units, hidden_units * 6, bias=False),
            torch.nn.SiLU(),
        )

        self.out_linear = torch.nn.Linear(hidden_units * 3, hidden_units)

        self.rope = RotaryEmbedding(self.head_dim, rope_base=rope_base, rope_fraction=rope_fraction)


    def forward(self, x, attn_mask=None, rel_ts_bias=None):
        """
        x: [B,S,H]
        attn_mask: [B,S,S]，True=允许；False=屏蔽
        rel_pos_bias: [1,S,S] 或 [B,S,S]，相对位置偏置
        rel_ts_bias:  [B,S,S]，相对时间偏置
        """
        batch_size, seq_len, _ = x.size()

        # 计算 U, V, Q, K
        mm_output = self.qkvu_linear(x)  # [B,S,6H]
        U, V, Q, K = torch.split(
            mm_output,
            [
                self.hidden_units * 3,  # U
                self.hidden_units,      # V
                self.hidden_units,      # Q
                self.hidden_units,      # K
            ],
            dim=-1,
        )

        # reshape 为 multi-head 格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B,h,S,d]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B,h,S,d]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B,h,S,d]

        # 应用 RoPE 到 Q/K（点积之前）
        Q_rope, K_rope = self.rope.apply_rotary(Q, K)  # [B,h,S,d], [B,h,S,d]

        # 注意力 logit
        qk_attn = torch.matmul(Q, K.transpose(-2, -1))  # [B,h,S,S]
        qk_attn = F.relu(qk_attn) / seq_len  # [B,h,S,S]

        qk_attn_rope = torch.matmul(Q_rope, K_rope.transpose(-2, -1))  # [B,h,S,S]
        qk_attn_rope = F.relu(qk_attn_rope) / seq_len  # [B,h,S,S]


        if rel_ts_bias is None:
            rel_ts_bias = torch.zeros(batch_size, seq_len, seq_len, device=x.device, dtype=V.dtype)


        if attn_mask is not None:
            # qk_attn: [B,h,S,S] 用 [B,1,S,S] 屏蔽
            mask4attn = attn_mask.unsqueeze(1)  # [B,1,S,S]
            qk_attn = qk_attn.masked_fill(mask4attn.logical_not(), 0.0)
            qk_attn_rope = qk_attn_rope.masked_fill(mask4attn.logical_not(), 0.0)
            # 对两个 bias 用 [B,S,S] 屏蔽
            rel_ts_bias = rel_ts_bias.masked_fill(attn_mask.logical_not(), 0.0)    # [B,S,S]

        # 三路输出
        ts_output = torch.einsum("bnm,bhmd->bnhd", rel_ts_bias, V)    # [B,S,h,d]
        attn_output_rope = torch.einsum("bhnm,bhmd->bnhd", qk_attn_rope, V)     # [B,S,h,d]
        attn_output = torch.einsum("bhnm,bhmd->bnhd", qk_attn, V)     # [B,S,h,d]

        combined_output = torch.cat([attn_output_rope, ts_output, attn_output], dim=-1).contiguous()  # [B,S,h,3d]
        combined_output = combined_output.view(batch_size, seq_len, self.hidden_units * 3)      # [B,S,3H]

        output = self.out_linear(combined_output * U)
        output = self.rms_norm(output + x)  # DeepNorm
        return output

class FeatureInteractionEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, expansion_factor=4):
        """
        Args:
            input_dim: 输入的维度。
            output_dim: 输出的维度。
            dropout_rate: Dropout 概率。
            expansion_factor: 中间层维度的扩展因子。
        """
        super(FeatureInteractionEncoder, self).__init__()
        self.output_dim = output_dim
        self.expansion_factor = expansion_factor
        self.down_proj = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(output_dim, output_dim * expansion_factor, bias=False),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
        )

    def forward(self, X):
        """
        inputs: [B, S, D]
        returns: [B, S, output_dim]
        """
        batch_size, seq_len, D = X.size()
        X = self.down_proj(X)
        H = self.linear(X).view(-1, seq_len, self.expansion_factor, self.output_dim)
        X = torch.einsum("bsd,bsrd->bsd", X, H) + X
        return X


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
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        user_item_dnn: 用户+物品特征交互编码器
        item_dnn: 物品侧特征编码器
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.maxlen = args.maxlen
        self.hidden_units = args.hidden_units

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.embedding_dim, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.embedding_dim, padding_idx=0)

        # 相对时间/位置偏置
        self.rel_time_bias = RelativeTimeBias(
            max_seq_len=args.maxlen + 1,
            num_buckets=128,
            bucketization_fn=lambda x: torch.clamp(
                (torch.log2(x.clamp(min=1.0)).floor() + 1).long(),  # 1,2,...  (delta=1 -> 1)
                min=1, max=128
            ).detach()
        )

        # 绝对时间编码（仍作为 add-on 特征加到序列表示上）
        self.time_abs_enc = FourierTimeEncoding(hidden_units=args.hidden_units)

        # 将 hour/dow/weekend 作为“item 侧稀疏特征”输入到 user_item_dnn（注意：embedding_dim）
        self.hour_emb = torch.nn.Embedding(24 + 1, args.embedding_dim, padding_idx=0)
        self.dow_emb = torch.nn.Embedding(7 + 1, args.embedding_dim, padding_idx=0)
        # 1=工作日, 2=周末, 0=pad
        self.weekend_emb = torch.nn.Embedding(2 + 1, args.embedding_dim, padding_idx=0)
        # 新增：用户侧序列时间跨度特征（log1p 后线性映射到 embedding_dim）
        self.user_time_span_proj = torch.nn.Linear(1, args.embedding_dim, bias=False)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        self.attention_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        # 计算两条路径的输入维度
        userdim = args.embedding_dim * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT) + 1) \
                  + len(self.USER_CONTINUAL_FEAT)

        itemdim_for_itemdnn = (
                args.embedding_dim * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
                + len(self.ITEM_CONTINUAL_FEAT)
                + args.embedding_dim * len(self.ITEM_EMB_FEAT)
        )
        # user_item 路径：在原有基础上 + hour/dow/weekend + 交叉特征
        itemdim_for_user_itemdnn = itemdim_for_itemdnn + 3 * args.embedding_dim + args.embedding_dim * len(self.ITEM_CROSS_FEAT)
        all_dim_for_user_itemdnn = userdim + itemdim_for_user_itemdnn

        # 两套编码器：user_item 使用包含时间特征与交叉特征的维度；item 使用原始维度（不含交叉特征）
        self.user_item_dnn = FeatureInteractionEncoder(all_dim_for_user_itemdnn, args.hidden_units, args.dropout_rate)
        self.item_dnn = FeatureInteractionEncoder(itemdim_for_itemdnn, args.hidden_units, args.dropout_rate)

        for _ in range(args.num_blocks):
            new_attn_layer = HSTU(args.hidden_units, args.num_heads, args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

        # ... embedding 表创建（增加 ITEM_CROSS_FEAT）
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        # 新增：交叉特征的 embedding 表
        for k in self.ITEM_CROSS_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_CROSS_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.embedding_dim)

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']

        # 新增：单独记录交叉特征（若数据侧未提供，回退为空）
        cross_keys = feat_types.get('item_cross', [])
        self.ITEM_CROSS_FEAT = {k: feat_statistics[k] for k in cross_keys}

        # 普通 item 稀疏特征：从 item_sparse 中排除交叉特征
        self.ITEM_SPARSE_FEAT = {
            k: feat_statistics[k]
            for k in feat_types['item_sparse']
            if k not in self.ITEM_CROSS_FEAT
        }

        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度

    def feat2emb(self, seq, feature_batch, mask=None, include_user=False, seq_ts=None):
        """
        Args:
            seq: [B,S]
            feature_batch: Dataset.collate_fn 预张量化特征
            mask: [B,S]（1=item, 2=user, 0=pad）
            include_user: 是否处理用户特征（user_item 路径）
            seq_ts: [B,S] 绝对时间戳（秒）；仅在 include_user=True 时用于构造 item 时间离散特征
        Returns:
            seqs_emb: [B, S, H]（经对应的 DNN 编码后）
        """
        seq = seq.to(self.dev)

        # 初始化 item/user id embedding
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

        ft = feature_batch  # 预张量化特征

        # item 特征（不含时间离散特征）
        # 仅保留普通 item 稀疏特征；交叉特征只在 user_item 路径启用
        for k, tens in ft.get('item_sparse', {}).items():
            if (k in self.ITEM_SPARSE_FEAT) or (include_user and (k in self.ITEM_CROSS_FEAT)):
                item_feat_list.append(self.sparse_emb[k](tens.to(self.dev)))
        for k, tens in ft.get('item_array', {}).items():
            item_feat_list.append(self.sparse_emb[k](tens.to(self.dev)).sum(2))
        for k, tens in ft.get('item_continual', {}).items():
            item_feat_list.append(tens.to(self.dev).unsqueeze(2))
        for k in self.ITEM_EMB_FEAT:
            if k in ft.get('item_emb', {}):
                t = ft['item_emb'][k].to(self.dev)
                item_feat_list.append(self.emb_transform[k](t))

        # 将 hour / dow / weekend 作为“item 侧稀疏特征”拼到 user_item 路径
        if include_user and (seq_ts is not None):
            ts = seq_ts.to(self.dev).long()  # [B,S]
            is_item = (mask == 1).to(torch.bool).to(self.dev)
            hour_idx = ((ts % 86400) // 3600) + 1
            dow_idx = (((ts // 86400) + 4) % 7) + 1
            weekend_idx = (dow_idx >= 6).long() + 1
            hour_idx = torch.where(is_item, hour_idx, torch.zeros_like(hour_idx))
            dow_idx = torch.where(is_item, dow_idx, torch.zeros_like(dow_idx))
            weekend_idx = torch.where(is_item, weekend_idx, torch.zeros_like(weekend_idx))
            item_feat_list.append(self.hour_emb(hour_idx))
            item_feat_list.append(self.dow_emb(dow_idx))
            item_feat_list.append(self.weekend_emb(weekend_idx))

            # 新增：序列级“时间跨度”用户特征（最后一个 item ts - 第一个 item ts）
            valid_item = is_item & (ts > 0)
            has_any = valid_item.any(dim=1)  # [B]

            INF = torch.iinfo(ts.dtype).max
            ts_min = torch.where(valid_item, ts, torch.full_like(ts, INF))
            first_ts = ts_min.min(dim=1).values  # [B]
            ts_max = torch.where(valid_item, ts, torch.zeros_like(ts))
            last_ts = ts_max.max(dim=1).values  # [B]

            # 若没有有效 item，则跨度视为 0（first_ts 用 last_ts 兜底）
            first_ts = torch.where(has_any, first_ts, last_ts)
            span = (last_ts - first_ts).clamp(min=0).to(torch.float32)  # [B]
            span_log1p = torch.log1p(span).unsqueeze(-1)  # [B,1]

            # 线性映射到 embedding_dim，并仅在用户位置保留
            span_emb = self.user_time_span_proj(span_log1p)  # [B,E]
            span_emb = span_emb.unsqueeze(1).expand(-1, seq.size(1), -1)  # [B,S,E]
            user_pos_mask = (mask == 2).to(self.dev).unsqueeze(-1).to(span_emb.dtype)  # [B,S,1]
            span_emb = span_emb * user_pos_mask

            user_feat_list.append(span_emb)

        # user 特征（仅在 include_user=True 时使用）
        if include_user:
            for k, tens in ft.get('user_sparse', {}).items():
                user_feat_list.append(self.sparse_emb[k](tens.to(self.dev)))
            for k, tens in ft.get('user_array', {}).items():
                user_feat_list.append(self.sparse_emb[k](tens.to(self.dev)).sum(2))
            for k, tens in ft.get('user_continual', {}).items():
                user_feat_list.append(tens.to(self.dev).unsqueeze(2))

        if include_user:
            all_user_item_emb = torch.concat(user_feat_list + item_feat_list, dim=-1)  # [B,S,D_user_item]
            seqs_emb = self.user_item_dnn(all_user_item_emb)
        else:
            all_item_emb = torch.concat(item_feat_list, dim=-1)                        # [B,S,D_item]
            seqs_emb = self.item_dnn(all_item_emb)

        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature, seq_ts):
        """
        将日志序列（含特征）编码为序列 hidden states
        """

        # 加入 seq_ts，使 hour/dow/weekend 作为 item 特征进入 user_item_dnn
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True, seq_ts=seq_ts)
        seqs *= self.hidden_units ** 0.5

        # 仅保留绝对时间傅里叶特征作为加性偏置（padding 位置为 0）
        ts = seq_ts.to(self.dev).long()                         # [B,S]
        valid = (mask != 0).to(torch.bool).to(self.dev)         # [B,S]
        time_abs = self.time_abs_enc(ts) * valid.unsqueeze(-1)  # [B,S,H]
        seqs += time_abs

        seqs = self.emb_dropout(seqs)

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)  # [S,S]
        key_query_valid = (mask != 0).to(torch.bool).to(self.dev)  # [B,S]
        attention_mask = (
            attention_mask_tril.unsqueeze(0)
            & key_query_valid.unsqueeze(2)
            & key_query_valid.unsqueeze(1)
        )  # [B,S,S]

        # 相对时间/位置偏置
        rel_ts_bias = self.rel_time_bias(ts, key_query_valid)

        for i in range(len(self.attention_layers)):
            seqs = self.attention_layers[i](
                seqs, attn_mask=attention_mask, rel_ts_bias=rel_ts_bias
            )
        log_feats = F.normalize(seqs, dim=-1)
        return log_feats

    def forward(
            self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature,
            neg_feature, seq_ts
    ):
        """
        训练时调用，计算正负样本的logits
        """
        log_feats = self.log2feats(user_item, mask, seq_feature, seq_ts)
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        pos_embs = F.normalize(pos_embs, dim=-1)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)
        neg_embs = F.normalize(neg_embs, dim=-1)

        return pos_embs, neg_embs, log_feats

    def predict(self, log_seqs, seq_feature, mask, seq_ts):
        """
        计算用户序列的表征
        """
        log_feats = self.log2feats(log_seqs, mask, seq_feature, seq_ts)
        final_feat = log_feats[:, -1, :]
        return final_feat

    def _pack_item_features_for_seq(self, feature_array_list):
        """
        将 save_item_emb 中的“list[dict]（长度L）”打包成与 DataLoader 输出一致的预张量化结构（B=1）。
        仅包含 item 侧特征。
        """
        L = len(feature_array_list)
        B = 1
        out = {
            'item_sparse': {},
            'user_sparse': {},
            'item_array': {},
            'user_array': {},
            'item_continual': {},
            'user_continual': {},
            'item_emb': {},
        }
        # item_sparse
        for k in self.ITEM_SPARSE_FEAT:
            arr = np.zeros((B, L), dtype=np.int64)
            for s in range(L):
                v = feature_array_list[s].get(k, 0)
                arr[0, s] = int(v if not isinstance(v, list) else (v[0] if len(v) > 0 else 0))
            out['item_sparse'][k] = torch.from_numpy(arr)
        # item_array（本任务 item_array 为空，保留逻辑）
        for k in self.ITEM_ARRAY_FEAT:
            max_len = 1
            for s in range(L):
                v = feature_array_list[s].get(k, [0])
                if isinstance(v, list):
                    max_len = max(max_len, len(v))
            arr = np.zeros((B, L, max_len), dtype=np.int64)
            for s in range(L):
                v = feature_array_list[s].get(k, [0])
                if isinstance(v, list) and len(v) > 0:
                    Lc = min(max_len, len(v))
                    arr[0, s, :Lc] = np.asarray(v[:Lc], dtype=np.int64)
            out['item_array'][k] = torch.from_numpy(arr)
        # item_continual（当前为空）
        for k in self.ITEM_CONTINUAL_FEAT:
            arr = np.zeros((B, L), dtype=np.float32)
            for s in range(L):
                v = feature_array_list[s].get(k, 0.0)
                try:
                    arr[0, s] = float(v)
                except Exception:
                    arr[0, s] = 0.0
            out['item_continual'][k] = torch.from_numpy(arr)
        # item_emb
        for k, dim in self.ITEM_EMB_FEAT.items():
            arr = np.zeros((B, L, dim), dtype=np.float32)
            for s in range(L):
                vec = np.asarray(feature_array_list[s].get(k, np.zeros(dim, dtype=np.float32)), dtype=np.float32)
                if vec.shape[0] != dim:
                    Lc = min(dim, vec.shape[0])
                    if Lc > 0:
                        arr[0, s, :Lc] = vec[:Lc]
                else:
                    arr[0, s] = vec
            out['item_emb'][k] = torch.from_numpy(arr)
        return out

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

            # [1, N] 的“序列”，与打包后的特征对齐
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)

            # 将本批的 list[dict] -> 预张量化结构（B=1, S=N）
            batch_feat_list = []
            for i in range(start_idx, end_idx):
                batch_feat_list.append(feat_dict[i])
            prepacked_feat = self._pack_item_features_for_seq(batch_feat_list)

            # 直接走模型的快速路径
            batch_emb = self.feat2emb(item_seq, prepacked_feat, include_user=False).squeeze(0)
            batch_emb = F.normalize(batch_emb, dim=-1)

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))