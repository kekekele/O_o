import json
import pickle
import struct
import math
import os
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from tqdm import tqdm

# 优先使用 orjson 加速解析；不可用则回退到内置 json
try:
    import orjson as _fastjson

    def JSON_LOADS(b: bytes):
        # orjson.loads 接受 bytes，返回 Python 原生对象
        return _fastjson.loads(b)
except Exception:
    def JSON_LOADS(b: bytes):
        # 内置 json.loads 接受 str，因此需 decode
        return json.loads(b.decode("utf-8", errors="ignore"))


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典（外层 key 为 int 的 item_id）
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_mats: 多模态特征矩阵字典（feat_id -> np.ndarray[item_reid, dim]）
        mm_emb_dict: 多模态字典形式，feat_id -> {creative_id(raw): np.ndarray[dim]}，供 infer 使用
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id 原始ID)
        indexer_u_rev: 用户索引字典 (reid -> user_id 原始ID)
        indexer: 索引字典
        feature_default_value: 特征缺省值（包含 user+item+emb 的全量字段）
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        # 先加载 indexer，便于构建 mm_emb 致密矩阵
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
        self.indexer = indexer
        self.itemnum = len(indexer['i'])
        self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}

        # item_feat_dict：外层 key 转 int，避免后续 str(t) 转换
        with open(Path(data_dir, "item_feat_dict.json"), 'r', encoding='utf-8') as f:
            _raw_item_feat = json.load(f)
        self.item_feat_dict = {int(k): v for k, v in _raw_item_feat.items()}

        # 预构建可用 item 列表与掩码（用于负采样）
        self.valid_item_ids = np.fromiter(self.item_feat_dict.keys(), dtype=np.int32)
        self.valid_item_mask = np.zeros(self.itemnum + 1, dtype=bool)
        _clip_ids = self.valid_item_ids[self.valid_item_ids <= self.itemnum]
        self.valid_item_mask[_clip_ids] = True

        # 统计点击并准备分桶映射
        self.CLICK_BUCKETS = 16
        self.item_click_counts = self._compute_item_click_counts()
        self._click_bucket_map = self._load_click_bucket_map()

        # 加载多模态特征为致密矩阵（feat_id -> np.ndarray[item_reid, dim]）
        self.mm_emb_mats, self.mm_emb_dict = load_mm_emb(
            Path(data_dir, "creative_emb"),
            self.mm_emb_ids,
            indexer_i=indexer['i'],
            itemnum=self.itemnum
        )

        # 初始化特征信息（依赖 mm_emb_mats 确定默认 embedding 维度）
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

        # 为快速特征填充准备“全量模板”和简单缓存（每个 worker 独立）
        self._build_feat_templates()
        self._item_feat_cache = OrderedDict()
        self._item_feat_cache_max = 400000  # 可根据内存调节，如 20000/100000

    def _load_data_and_offsets(self):
        """
        加载用户序列偏移量；不在此处打开文件，避免在多进程间传递已打开句柄
        """
        self._data_file_path = self.data_dir / "seq.jsonl"
        self.data_file = None  # 懒打开
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _ensure_file_open(self):
        """
        在各 worker 进程内懒加载打开数据文件
        """
        if getattr(self, "data_file", None) is None:
            # 使用二进制模式读取，配合 JSON_LOADS
            self.data_file = open(self._data_file_path, 'rb')

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = JSON_LOADS(line)
        return data

    def _random_neq(self, hist_set):
        """
        生成一个不在序列 hist_set 中的随机 item（仅从有特征的合法 item 中采样）
        """
        if self.valid_item_ids.size == 0:
            return 0
        # 快速拒绝采样（采用有效 item 列表）
        for _ in range(16):
            cand = int(self.valid_item_ids[np.random.randint(0, self.valid_item_ids.size)])
            if cand not in hist_set:
                return cand
        for _ in range(256):
            cand = int(self.valid_item_ids[np.random.randint(0, self.valid_item_ids.size)])
            if cand not in hist_set:
                return cand
        return 0

    # ======================== 新增：点击统计与分桶工具 ========================

    def _compute_item_click_counts(self):
        """
        统计每个 item 的点击次数（action_type==1）；
        优先从 data_dir/seq.jsonl 读取，否则回退到当前数据文件（如 predict_seq.jsonl）。
        """
        counts = np.zeros(self.itemnum + 1, dtype=np.int32)
        prefer = self.data_dir / "seq.jsonl"
        use_path = prefer if prefer.exists() else self._data_file_path
        try:
            with open(use_path, 'rb') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        seq = JSON_LOADS(line)
                    except Exception:
                        continue
                    # record: [user_id, item_id, user_feat, item_feat, action_type, timestamp]
                    for rec in seq:
                        if not isinstance(rec, (list, tuple)) or len(rec) < 6:
                            continue
                        i = rec[1]
                        a = rec[4]
                        if isinstance(i, int) and 0 < i <= self.itemnum and a == 1:
                            counts[i] += 1
        except FileNotFoundError:
            pass
        return counts

    def _click_count_to_bucket(self, c: int) -> int:
        """
        将点击次数映射到 [1..CLICK_BUCKETS] 的对数分桶；0 次并非 pad，这里归入第1桶。
        pad 位置由上游 token_type 控制；item_id=0 时返回 0。
        """
        if c <= 0:
            return 1
        b = int(math.floor(math.log2(c))) + 1
        return int(max(1, min(self.CLICK_BUCKETS, b)))

    def _map_action_to_id(self, a) -> int:
        """
        将原始 action_type（0=曝光，1=点击，或 None）映射到嵌入索引：
        0=pad，1=曝光，2=点击
        """
        if a is None:
            return 0
        return 1 if int(a) == 0 else 2

    def _load_click_bucket_map(self):
        """
        装载 item_click_bucket.json，优先从 USER_CACHE_PATH，其次 TRAIN_CKPT_PATH、数据目录。
        返回 {item_reid: bucket_int}
        """
        candidates = []
        cache_root = os.environ.get("USER_CACHE_PATH", None)
        if cache_root:
            candidates.append(Path(cache_root) / "item_click_bucket.json")
        ckpt_root = os.environ.get("TRAIN_CKPT_PATH", None)
        if ckpt_root:
            candidates.append(Path(ckpt_root) / "item_click_bucket.json")
        candidates.append(self.data_dir / "item_click_bucket.json")

        for p in candidates:
            try:
                if p.exists():
                    with open(p, 'r', encoding='utf-8') as f:
                        d = json.load(f)
                    print(f"Loaded click bucket map from {p}")
                    return {int(k): int(v) for k, v in d.items()}
            except Exception as e:
                print(f"warn: failed loading {p}: {e}")
        return {}

    # ======================================================================

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式
        """
        # 关键：在 worker 内懒打开文件，支持 num_workers>0
        self._ensure_file_open()

        user_sequence = self._load_user_data(uid)

        # 构造等价于原来 insert(0, ...) + append(...) 的序列：
        user_tokens = []
        item_tokens = []
        item_ids_in_hist = set()

        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            if u and user_feat:
                user_tokens.append((u, user_feat, 2, action_type, timestamp))
            if i and item_feat:
                item_tokens.append((i, item_feat, 1, action_type, timestamp))
                item_ids_in_hist.add(i)

        ext_user_sequence = list(reversed(user_tokens)) + item_tokens
        if not ext_user_sequence:
            return self._empty_return()

        S = self.maxlen + 1
        seq = np.zeros([S], dtype=np.int32)
        pos = np.zeros([S], dtype=np.int32)
        neg = np.zeros([S], dtype=np.int32)
        token_type = np.zeros([S], dtype=np.int32)
        next_token_type = np.zeros([S], dtype=np.int32)
        next_action_type = np.zeros([S], dtype=np.int32)
        seq_ts = np.zeros([S], dtype=np.int64)

        # 预填全量默认字典，保证所有位置都含有所有字段键（如 '100'）
        seq_feat = np.empty([S], dtype=object)
        pos_feat = np.empty([S], dtype=object)
        neg_feat = np.empty([S], dtype=object)
        seq_feat.fill(self.feature_default_value)
        pos_feat.fill(self.feature_default_value)
        neg_feat.fill(self.feature_default_value)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        # 反向填充；与原实现一致：丢弃最后一个 token 仅作为“下一步”
        for record_tuple in reversed(ext_user_sequence[:-1]):
            # 当前步
            i, feat, type_, act_type, ts_cur = record_tuple
            # 下一步（用于 pos/next_*）
            next_i, next_feat_raw, next_type, next_act_type, ts_next = nxt

            # 当前 token 特征（基于全量模板）
            if type_ == 1:
                feat_filled = self._make_item_feat(i, feat)
                # 动态加入 action_type 嵌入索引 '900'（0=pad,1=曝光,2=点击）
                feat_filled = feat_filled.copy()
                feat_filled['900'] = self._map_action_to_id(act_type)
            else:
                feat_filled = self._make_user_feat(feat)

            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat_filled
            seq_ts[idx] = int(ts_cur) if ts_cur is not None else 0

            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = self._make_item_feat(next_i, next_feat_raw)
                neg_id = self._random_neq(item_ids_in_hist)
                neg[idx] = neg_id
                neg_raw = self.item_feat_dict.get(neg_id, {})
                neg_feat[idx] = self._make_item_feat(neg_id, neg_raw)

            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts

    def _empty_return(self):
        S = self.maxlen + 1
        zeros_i32 = np.zeros([S], dtype=np.int32)
        zeros_i64 = np.zeros([S], dtype=np.int64)
        seq_feat = np.empty([S], dtype=object); seq_feat.fill(self.feature_default_value)
        pos_feat = np.empty([S], dtype=object); pos_feat.fill(self.feature_default_value)
        neg_feat = np.empty([S], dtype=object); neg_feat.fill(self.feature_default_value)
        token_type = zeros_i32.copy()
        next_token_type = zeros_i32.copy()
        next_action_type = zeros_i32.copy()
        return zeros_i32, zeros_i32.copy(), zeros_i32.copy(), token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, zeros_i64

    def __len__(self):
        """
        返回数据集长度，即用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        # 新增：action_type（900）与点击次数分桶（901）
        feat_types['item_sparse'] += ['900', '901']

        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            if feat_id in ('900', '901'):
                # 在下方分别填充
                continue
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            # 使用致密矩阵的列维度作为默认 emb 长度
            emb_dim = self.mm_emb_mats[feat_id].shape[1]
            feat_default_value[feat_id] = np.zeros(emb_dim, dtype=np.float32)

        # 新增字段的默认与统计
        # 900: action_type（0=pad, 1=曝光, 2=点击） -> 2 类，Embedding 侧 +1 预留 pad
        feat_default_value['900'] = 0
        feat_statistics['900'] = 2
        # 901: 点击次数分桶（[1..CLICK_BUCKETS]，0=pad）
        feat_default_value['901'] = 0
        feat_statistics['901'] = self.CLICK_BUCKETS

        return feat_default_value, feat_types, feat_statistics

    def _build_feat_templates(self):
        """
        构建全量模板字典，包含 user+item+emb 所有字段，便于快速 copy+update
        """
        all_feat_ids = []
        for v in self.feature_types.values():
            all_feat_ids.extend(v)
        # 全量模板：确保任意 token 都包含所有键（例如 '100'）
        self._full_template = {fid: self.feature_default_value[fid] for fid in all_feat_ids}

    def _make_item_feat(self, item_id, base_feat_dict):
        """
        仅组合非 item_emb 字段；
        新增：写入 '901'（item 点击次数分桶，静态，与 item_id 相关且可缓存）。
        优先使用 _click_bucket_map；若无则用本地统计；仍无则置 0。
        """
        cache = self._item_feat_cache
        if item_id in cache:
            out = cache[item_id].copy()
            if base_feat_dict:
                out.update(base_feat_dict)
            cache.move_to_end(item_id, last=True)
            return out

        filled = self._full_template.copy()
        if base_feat_dict:
            filled.update(base_feat_dict)

        # 写入静态点击分桶 '901'
        if isinstance(item_id, int) and 0 < item_id <= self.itemnum:
            if self._click_bucket_map:
                filled['901'] = int(self._click_bucket_map.get(item_id, 0))
                if filled['901'] == 0:
                    # 若映射中没有该 item，回退到本地统计
                    cnt = int(self.item_click_counts[item_id]) if hasattr(self, 'item_click_counts') else 0
                    filled['901'] = self._click_count_to_bucket(cnt) if cnt > 0 else 0
            else:
                cnt = int(self.item_click_counts[item_id]) if hasattr(self, 'item_click_counts') else 0
                filled['901'] = self._click_count_to_bucket(cnt) if cnt > 0 else 0
        else:
            filled['901'] = 0

        # 放入 LRU（存一份轻量 copy）
        cache[item_id] = filled.copy()
        if len(cache) > self._item_feat_cache_max:
            cache.popitem(last=False)  # 弹出最旧条目
        return filled

    def _make_user_feat(self, base_feat_dict):
        """
        结合 base_feat_dict，快速生成用户特征字典（基于全量模板，保证包含 item 字段）。
        """
        filled = self._full_template.copy()
        if base_feat_dict:
            filled.update(base_feat_dict)
        return filled

    def fill_missing_feat(self, feat, item_id):
        """
        兼容接口：对于原始数据中缺失的特征进行填充缺省值。
        与旧实现一致：任何 token 都包含全量字段；若是合法 item，则覆盖 emb。
        """
        if feat is None:
            feat = {}
        filled = self._full_template.copy()
        filled.update(feat)
        if item_id != 0 and 0 <= item_id <= self.itemnum:
            for fid, mat in self.mm_emb_mats.items():
                filled[fid] = mat[item_id]
        return filled

    def _tensorize_feature_list(self, feat_list, ids=None, token_type=None, groups=None):
        """
        向量化张量化：将批内“字典数组”转为稠密张量。
        groups: 只构建需要的组，缺省构建全部。
        ids: 与位置对齐的 ID（例如 seq/pos/neg），用于查表 item_emb
        token_type: 与位置对齐的 token 类型（1=item, 2=user），用于在 seq 的用户位置屏蔽 item_emb
        """
        if groups is None:
            groups = ['item_sparse', 'user_sparse', 'item_array', 'user_array',
                      'item_continual', 'user_continual', 'item_emb']

        B = len(feat_list)
        S = len(feat_list[0]) if B > 0 else 0

        out = {
            'item_sparse': {},
            'user_sparse': {},
            'item_array': {},
            'user_array': {},
            'item_continual': {},
            'user_continual': {},
            'item_emb': {},
        }
        ft = self.feature_types

        # 1) sparse
        if 'item_sparse' in groups:
            for fid in ft['item_sparse']:
                arr = np.zeros((B, S), dtype=np.int64)
                for b in range(B):
                    row = feat_list[b]
                    vals = [row[s].get(fid, 0) for s in range(S)]
                    v = np.fromiter(
                        (int(v[0] if isinstance(v, list) and len(v) > 0 else (0 if isinstance(v, list) else int(v)))
                         for v in vals),
                        count=S, dtype=np.int64
                    )
                    arr[b] = v
                out['item_sparse'][fid] = torch.from_numpy(arr)

        if 'user_sparse' in groups:
            for fid in ft['user_sparse']:
                arr = np.zeros((B, S), dtype=np.int64)
                for b in range(B):
                    row = feat_list[b]
                    vals = [row[s].get(fid, 0) for s in range(S)]
                    v = np.fromiter(
                        (int(v[0] if isinstance(v, list) and len(v) > 0 else (0 if isinstance(v, list) else int(v)))
                         for v in vals),
                        count=S, dtype=np.int64
                    )
                    arr[b] = v
                out['user_sparse'][fid] = torch.from_numpy(arr)

        # 2) array
        if 'item_array' in groups:
            for fid in ft['item_array']:
                max_len = 1
                for b in range(B):
                    row = feat_list[b]
                    for s in range(S):
                        v = row[s].get(fid, [0])
                        if isinstance(v, list) and len(v) > max_len:
                            max_len = len(v)
                arr = np.zeros((B, S, max_len), dtype=np.int64)
                for b in range(B):
                    row = feat_list[b]
                    for s in range(S):
                        v = row[s].get(fid, [0])
                        if isinstance(v, list):
                            L = min(len(v), max_len)
                            if L > 0:
                                arr[b, s, :L] = np.asarray(v[:L], dtype=np.int64)
                        else:
                            arr[b, s, 0] = int(v)
                out['item_array'][fid] = torch.from_numpy(arr)

        if 'user_array' in groups:
            for fid in ft['user_array']:
                max_len = 1
                for b in range(B):
                    row = feat_list[b]
                    for s in range(S):
                        v = row[s].get(fid, [0])
                        if isinstance(v, list) and len(v) > max_len:
                            max_len = len(v)
                arr = np.zeros((B, S, max_len), dtype=np.int64)
                for b in range(B):
                    row = feat_list[b]
                    for s in range(S):
                        v = row[s].get(fid, [0])
                        if isinstance(v, list):
                            L = min(len(v), max_len)
                            if L > 0:
                                arr[b, s, :L] = np.asarray(v[:L], dtype=np.int64)
                        else:
                            arr[b, s, 0] = int(v)
                out['user_array'][fid] = torch.from_numpy(arr)

        # 3) continual
        if 'item_continual' in groups:
            for fid in ft['item_continual']:
                arr = np.zeros((B, S), dtype=np.float32)
                for b in range(B):
                    row = feat_list[b]
                    vals = [row[s].get(fid, 0.0) for s in range(S)]
                    v = np.fromiter(
                        ((float(v) if isinstance(v, (int, float, np.number)) else 0.0) for v in vals),
                        count=S, dtype=np.float32
                    )
                    arr[b] = v
                out['item_continual'][fid] = torch.from_numpy(arr)

        if 'user_continual' in groups:
            for fid in ft['user_continual']:
                arr = np.zeros((B, S), dtype=np.float32)
                for b in range(B):
                    row = feat_list[b]
                    vals = [row[s].get(fid, 0.0) for s in range(S)]
                    v = np.fromiter(
                        ((float(v) if isinstance(v, (int, float, np.number)) else 0.0) for v in vals),
                        count=S, dtype=np.float32
                    )
                    arr[b] = v
                out['user_continual'][fid] = torch.from_numpy(arr)

        # 4) item_emb（修复：在非 item 位置屏蔽为 0；并做越界保护）
        if 'item_emb' in groups:
            for fid in self.feature_types['item_emb']:
                dim = int(self.feature_default_value[fid].shape[0])
                if ids is not None:
                    ids_np = np.asarray(ids, dtype=np.int64)  # [B,S]
                    # 在用户位置（token_type != 1）置 0 行（padding）
                    if token_type is not None:
                        tt = np.asarray(token_type, dtype=np.int64)
                        ids_masked = ids_np.copy()
                        ids_masked[tt != 1] = 0
                    else:
                        ids_masked = ids_np
                    # 越界保护：非法 id -> 0
                    ids_masked = np.where((ids_masked >= 0) & (ids_masked <= self.itemnum), ids_masked, 0)
                    mat = self.mm_emb_mats[fid]  # [itemnum+1, dim]
                    arr = mat[ids_masked]  # [B,S,dim]
                    out['item_emb'][fid] = torch.from_numpy(arr)
                else:
                    out['item_emb'][fid] = torch.zeros((B, S, dim), dtype=torch.float32)

        return out

    def collate_fn(self, batch):
        """
        Args:
            batch: 多个 __getitem__ 返回的数据

        Returns:
            seq: 用户序列ID, torch.LongTensor [B, S]
            pos: 正样本ID, torch.LongTensor [B, S]
            neg: 负样本ID, torch.LongTensor [B, S]
            token_type: 用户序列类型, torch.LongTensor [B, S]
            next_token_type: 下一个token类型, torch.LongTensor [B, S]
            next_action_type: 下一个token动作类型, torch.LongTensor [B, S]
            seq_feat: 预张量化的用户序列特征字典
            pos_feat: 预张量化的正样本特征字典
            neg_feat: 预张量化的负样本特征字典
            seq_ts: 与 seq 对齐的时间戳, torch.LongTensor [B, S]
        """
        (seq, pos, neg, token_type, next_token_type, next_action_type,
         seq_feat, pos_feat, neg_feat, seq_ts) = zip(*batch)

        # 先全部保持在 NumPy
        seq_np = np.stack(seq)
        pos_np = np.stack(pos)
        neg_np = np.stack(neg)
        tt_np = np.stack(token_type)
        ntt_np = np.stack(next_token_type)
        nat_np = np.stack(next_action_type)
        ts_np = np.stack(seq_ts)

        # 向量化：直接传 NumPy，避免 tensor.numpy() 往返
        seq_feat = self._tensorize_feature_list(
            list(seq_feat), ids=seq_np, token_type=tt_np,
            groups=['item_sparse', 'user_sparse', 'item_array', 'user_array',
                    'item_continual', 'user_continual', 'item_emb']
        )
        pos_feat = self._tensorize_feature_list(
            list(pos_feat), ids=pos_np,
            groups=['item_sparse', 'item_array', 'item_continual', 'item_emb']
        )
        neg_feat = self._tensorize_feature_list(
            list(neg_feat), ids=neg_np,
            groups=['item_sparse', 'item_array', 'item_continual', 'item_emb']
        )

        # 再一次性转成 Torch
        seq = torch.from_numpy(seq_np).long()
        pos = torch.from_numpy(pos_np).long()
        neg = torch.from_numpy(neg_np).long()
        token_type = torch.from_numpy(tt_np).long()
        next_token_type = torch.from_numpy(ntt_np).long()
        next_action_type = torch.from_numpy(nat_np).long()
        seq_ts = torch.from_numpy(ts_np).long()

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        # 与训练集保持一致：lazy open
        self._data_file_path = self.data_dir / "predict_seq.jsonl"
        self.data_file = None  # 懒打开
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if isinstance(feat_value, list):
                processed_feat[feat_id] = [0 if isinstance(v, str) else v for v in feat_value]
            elif isinstance(feat_value, str):
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在 self.data_file 中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_xxxxxx（便于对照答案）
            seq_ts: 与 seq 对齐的时间戳（int64），padding 位置为 0
        """
        # 懒打开
        self._ensure_file_open()

        user_sequence = self._load_user_data(uid)

        user_tokens = []
        item_tokens = []
        user_id = None

        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action, timestamp = record_tuple

            if u:
                if isinstance(u, str):  # 字符串：user_id
                    user_id = u
                else:                   # re_id -> 原始 user_id
                    user_id = self.indexer_u_rev[u]

            if u and user_feat:
                uid_val = 0 if isinstance(u, str) else u
                user_feat = self._process_cold_start_feat(user_feat) if user_feat else user_feat
                user_tokens.append((uid_val, user_feat, 2, timestamp))

            if i and item_feat:
                if i > self.itemnum:
                    i = 0
                item_feat = self._process_cold_start_feat(item_feat) if item_feat else item_feat
                # 动态加入历史动作特征 900（0=pad, 1=曝光, 2=点击）
                feat_dict = dict(item_feat) if item_feat else {}
                feat_dict['900'] = self._map_action_to_id(action)
                item_tokens.append((i, feat_dict, 1, timestamp))

        ext_user_sequence = list(reversed(user_tokens)) + item_tokens

        S = self.maxlen + 1
        seq = np.zeros([S], dtype=np.int32)
        token_type = np.zeros([S], dtype=np.int32)
        seq_feat = np.empty([S], dtype=object)
        seq_feat.fill(self.feature_default_value)
        seq_ts = np.zeros([S], dtype=np.int64)

        idx = self.maxlen

        # 从后向前填充到固定长度；在测试时不要丢弃下一步，因为测试集序列已经丢弃了下一步
        if ext_user_sequence:
            for record_tuple in reversed(ext_user_sequence):
                i, feat, type_, ts_cur = record_tuple
                feat_filled = self._make_item_feat(i, feat) if type_ == 1 else self._make_user_feat(feat)
                seq[idx] = i
                token_type[idx] = type_
                seq_feat[idx] = feat_filled
                seq_ts[idx] = int(ts_cur) if ts_cur is not None else 0
                idx -= 1
                if idx == -1:
                    break

        return seq, token_type, seq_feat, user_id, seq_ts

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        return len(self.seq_offsets)

    def collate_fn(self, batch):
        """
        将多个 __getitem__ 返回的数据拼接成一个 batch

        Returns:
            seq: torch.LongTensor [B, S]
            token_type: torch.LongTensor [B, S]
            seq_feat: 预张量化的特征字典
            user_id: tuple(str)，长度 B
            seq_ts: torch.LongTensor [B, S]
        """
        seq, token_type, seq_feat, user_id, seq_ts = zip(*batch)
        seq = torch.from_numpy(np.stack(seq)).long()
        token_type = torch.from_numpy(np.stack(token_type)).long()
        seq_ts = torch.from_numpy(np.stack(seq_ts)).long()

        # 将“字典数组”转为批内张量
        seq_feat = self._tensorize_feature_list(list(seq_feat), ids=seq.numpy(), token_type=token_type.numpy())
        return seq, token_type, seq_feat, user_id, seq_ts


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids, indexer_i, itemnum):
    """
    加载多模态特征Embedding：
      - 返回致密矩阵（feat_id -> np.ndarray[itemnum+1, dim]）
      - 同时返回按 creative_id(raw) 索引的字典（feat_id -> {raw_id: np.ndarray[dim]}}）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mats = {}
    dicts = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        dim = SHAPE_DICT[feat_id]
        mat = np.zeros((itemnum + 1, dim), dtype=np.float32)  # 0 行作为缺省
        dct = {}

        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{dim}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'rb') as file:
                        for line in file:
                            data_dict_origin = JSON_LOADS(line.strip())
                            raw_id = data_dict_origin['anonymous_cid']
                            vec = np.asarray(data_dict_origin['emb'], dtype=np.float32)
                            dct[raw_id] = vec
                            reid = indexer_i.get(raw_id, None)
                            if reid is not None and 0 <= reid <= itemnum:
                                mat[reid] = vec
            except Exception as e:
                print(f"transfer error: {e}")
        else:
            with open(Path(mm_path, f'emb_{feat_id}_{dim}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)  # {raw_item_id: np.ndarray / list}
            for raw_id, vec in emb_dict.items():
                vec = np.asarray(vec, dtype=np.float32)
                dct[raw_id] = vec
                reid = indexer_i.get(raw_id, None)
                if reid is not None and 0 <= reid <= itemnum:
                    mat[reid] = vec
        mats[feat_id] = mat
        dicts[feat_id] = dct
        print(f'Loaded #{feat_id} mm_emb')
    return mats, dicts


# def load_mm_emb(mm_path, feat_ids, indexer_i, itemnum):
#     """
#     加载多模态特征Embedding：
#       - 返回致密矩阵（feat_id -> np.ndarray[itemnum+1, dim]）
#       - 同时返回按 creative_id(raw) 索引的字典（feat_id -> {raw_id: np.ndarray[dim]}}）
#     """
#     SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
#     mats = {}
#     dicts = {}
#     for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
#         dim = SHAPE_DICT[feat_id]
#         mat = np.zeros((itemnum + 1, dim), dtype=np.float32)  # 0 行作为缺省
#         dct = {}
#
#         # if feat_id != '81':
#         try:
#             base_path = Path(mm_path, f'emb_{feat_id}_{dim}')
#             # for json_file in base_path.glob('*.json'):
#             for json_file in base_path.glob('part-*'):
#                 with open(json_file, 'rb') as file:
#                     for line in file:
#                         data_dict_origin = JSON_LOADS(line.strip())
#                         raw_id = data_dict_origin['anonymous_cid']
#                         vec = np.asarray(data_dict_origin['emb'], dtype=np.float32)
#                         dct[raw_id] = vec
#                         reid = indexer_i.get(raw_id, None)
#                         if reid is not None and 0 <= reid <= itemnum:
#                             mat[reid] = vec
#         except Exception as e:
#             print(f"transfer error: {e}")
#
#         mats[feat_id] = mat
#         dicts[feat_id] = dct
#         print(f'Loaded #{feat_id} mm_emb')
#     return mats, dicts