from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb


class FourierTimeEncoding(torch.nn.Module):
    """
    将绝对时间戳（秒）映射为多频正余弦，再线性投到隐藏维。
    特点：参数量小、可外推、对不同时间尺度（小时~月）建模友好。作为hour、weekday、is_weekday补充
    """

    def __init__(
            self,
            hidden_units: int,
            num_frequencies: int = 8,
            min_period_seconds: float = 8 * 86400.0,  # 最短周期：1小时=3600.0，使用8 * 86400.0以防止与hour、weekday、is_weekday重复
            max_period_seconds: float = 40 * 86400.0,  # 最长周期 似乎数据集是40天内的
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
        out = self.proj(feats)  # [B,S,H]
        return out


class HSTU(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(HSTU, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        self.rms_norm = torch.nn.RMSNorm(hidden_units, eps=1e-8)

        self.qkvu_linear = torch.nn.Sequential(torch.nn.Linear(hidden_units, hidden_units * 4),
                                               torch.nn.SiLU())
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units * 3, hidden_units)

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.size()

        # 计算Q, K, V
        Q, K, V, U = torch.chunk(self.qkvu_linear(x), dim=-1, chunks=4)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = F.relu(torch.matmul(Q, K.transpose(-2, -1))) / seq_len

        if attn_mask is not None:
            attn_weights.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float(0.0))

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        output = self.out_linear(torch.cat([U, attn_output, attn_output * U], dim=-1))
        # output = F.dropout(output, p=self.dropout_rate, training=self.training)
        output = self.rms_norm(output + x)  # DeepNorm
        return output


class FeatureInteractionEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, expansion_factor=4):
        """
        Args:
            all_dim: 输入的维度。
            hidden_units: 输出的维度。
            dropout_rate: Dropout 概率。
            expansion_factor: 中间层维度的扩展因子。
        """
        super(FeatureInteractionEncoder, self).__init__()
        self.output_dim = output_dim
        self.expansion_factor = expansion_factor
        self.down_proj = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear = torch.nn.Sequential(torch.nn.Linear(output_dim, output_dim * expansion_factor, bias=False),
                                          torch.nn.ReLU(),
                                          torch.nn.Dropout(dropout_rate))

    def forward(self, X):
        """
        Args:
            inputs: 输入张量，形状为 (batch_size, seq_len, all_emb_dim)
        Returns:
            outputs: 输出张量，形状为 (batch_size, seq_len, hidden_units)
        """
        # inputs: [B, S, D]
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
        self.maxlen = args.maxlen
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.embedding_dim, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.embedding_dim, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        # 新增：绝对时间编码 + 周期特征嵌入（weekday / is_weekend / hour）
        self.time_abs_enc = FourierTimeEncoding(hidden_units=args.hidden_units)
        # 周期特征嵌入（+1 预留0给padding）
        self.hour_emb = torch.nn.Embedding(24 + 1, args.hidden_units, padding_idx=0)
        self.dow_emb = torch.nn.Embedding(7 + 1, args.hidden_units, padding_idx=0)
        # 1=工作日, 2=周末, 0=pad
        self.weekend_emb = torch.nn.Embedding(2 + 1, args.hidden_units, padding_idx=0)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        self.attention_layernorms = torch.nn.ModuleList()
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
        all_dim = userdim + itemdim

        self.user_item_dnn = FeatureInteractionEncoder(all_dim, args.hidden_units, args.dropout_rate)

        self.item_dnn = FeatureInteractionEncoder(itemdim, args.hidden_units, args.dropout_rate)

        for _ in range(args.num_blocks):
            new_attn_layer = HSTU(args.hidden_units, args.num_heads, args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

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

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度

    def feat2emb(self, seq, feature_batch, mask=None, include_user=False):
        """
        Args:
            seq: 序列ID [B,S]
            feature_batch: 预张量化的特征字典（由 Dataset.collate_fn 生成）
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征

        Returns:
            seqs_emb: [B, S, H]
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

        # 直接消费预张量化特征（CPU->GPU 一次性拷贝，避免 Python 循环）
        ft = feature_batch  # dict

        # item 特征
        for k, tens in ft.get('item_sparse', {}).items():
            item_feat_list.append(self.sparse_emb[k](tens.to(self.dev)))
        for k, tens in ft.get('item_array', {}).items():
            item_feat_list.append(self.sparse_emb[k](tens.to(self.dev)).sum(2))
        for k, tens in ft.get('item_continual', {}).items():
            item_feat_list.append(tens.to(self.dev).unsqueeze(2))
        for k in self.ITEM_EMB_FEAT:
            if k in ft.get('item_emb', {}):
                t = ft['item_emb'][k].to(self.dev)
                item_feat_list.append(self.emb_transform[k](t))

        # user 特征（仅在 include_user=True 时使用）
        if include_user:
            for k, tens in ft.get('user_sparse', {}).items():
                user_feat_list.append(self.sparse_emb[k](tens.to(self.dev)))
            for k, tens in ft.get('user_array', {}).items():
                user_feat_list.append(self.sparse_emb[k](tens.to(self.dev)).sum(2))
            for k, tens in ft.get('user_continual', {}).items():
                user_feat_list.append(tens.to(self.dev).unsqueeze(2))

        if include_user:
            all_user_item_emb = torch.concat(user_feat_list + item_feat_list, dim=-1)  # B S D
            seqs_emb = self.user_item_dnn(all_user_item_emb)
        else:
            all_item_emb = torch.concat(item_feat_list, dim=-1)
            seqs_emb = self.item_dnn(all_item_emb)

        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature, seq_ts):
        """
        将日志序列（含特征）编码为序列 hidden states
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= log_seqs != 0
        seqs += self.pos_emb(poss)

        # =========== 绝对时间编码 + 三个周期特征 ===========
        ts = seq_ts.to(self.dev, non_blocking=True)  # [B,S] 绝对时间戳（秒）
        valid = (mask != 0).to(torch.bool).to(self.dev)  # [B,S] True=有效token

        # 周期时间特征：hour(1..24), dow(1..7, ISO: 周一=1), is_weekend(1=工作日, 2=周末)
        hour = ((ts % 86400) // 3600) + 1  # 1..24
        dow = (((ts // 86400) + 4) % 7) + 1  # 1..7（1970-01-01是周四(+4)）
        weekend_flag = (dow >= 6).long() + 1  # 1=工作日, 2=周末

        # padding 置 0（embedding 的 padding_idx=0）
        hour = torch.where(valid, hour, torch.zeros_like(hour))
        dow = torch.where(valid, dow, torch.zeros_like(dow))
        weekend = torch.where(valid, weekend_flag, torch.zeros_like(weekend_flag))

        # 绝对时间傅里叶编码（零出 padding 位置）
        time_abs = self.time_abs_enc(ts)  # [B,S,H]
        time_abs = time_abs * valid.unsqueeze(-1)
        seqs += self.hour_emb(hour) + self.dow_emb(dow) + self.weekend_emb(weekend) + time_abs

        seqs = self.emb_dropout(seqs)

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            seqs = self.attention_layers[i](seqs, attn_mask=attention_mask)
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