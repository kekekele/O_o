import argparse
import json
import os
import struct
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import BaselineModel


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--seed', default=20252026, type=int)

    # Baseline Model construction
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_units', default=256, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)

    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--neg_pop_alpha', default=0.15, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # Torch ANN（新增）
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--torch_query_bs', default=None, type=int, help='查询分块大小（默认: cuda=1024, cpu=256）')
    parser.add_argument('--torch_item_bs', default=None, type=int, help='库向量分块大小（默认: cuda=16384, cpu=8192）')
    parser.add_argument('--use_fp16', default=False, action='store_true', help='在 CUDA 上使用半精度进行相似度计算')

    # DataLoader workers（推理）
    parser.add_argument('--num_workers', default=8, type=int)

    # 可选：直接在推理端声明组合（若未能读取到清单文件时作为兜底）
    parser.add_argument('--feature_crosses', nargs='*', default=["118+120", "116+118"],
                        help='例如: ["118+120"]；缺省(None)表示不使用交叉特征')

    args = parser.parse_args()
    return args


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def _load_click_bucket_map_for_infer() -> Dict[int, int]:
    """
    加载 item_click_bucket.json（方案A）
    搜索顺序：
      - USER_CACHE_PATH
      - EVAL_RESULT_PATH
      - MODEL_OUTPUT_PATH
      - TRAIN_CKPT_PATH
      - EVAL_DATA_PATH
    返回：{item_reid: bucket}
    """
    candidates: List[Path] = []
    env_keys = ["USER_CACHE_PATH", "EVAL_RESULT_PATH", "MODEL_OUTPUT_PATH", "TRAIN_CKPT_PATH", "EVAL_DATA_PATH"]
    for k in env_keys:
        v = os.environ.get(k, None)
        if v:
            candidates.append(Path(v) / "item_click_bucket.json")
    for p in candidates:
        try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                print(f"Loaded item_click_bucket.json from {p}")
                return {int(k): int(v) for k, v in d.items()}
        except Exception as e:
            print(f"warn: failed reading {p}: {e}")
    print("warn: item_click_bucket.json not found. Will use bucket=0 as fallback.")
    return {}


# ======================== 泛化的特征组合（推理侧加载） ========================

def _search_paths_for_infer():
    paths = []
    for k in ["USER_CACHE_PATH", "EVAL_RESULT_PATH", "MODEL_OUTPUT_PATH", "TRAIN_CKPT_PATH", "EVAL_DATA_PATH"]:
        v = os.environ.get(k, None)
        if v:
            paths.append(Path(v))
    return paths

def _load_json_from_paths_infer(filename: str):
    for root in _search_paths_for_infer():
        p = root / filename
        try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f), p
        except Exception as e:
            print(f"warn: failed loading {filename} from {p}: {e}")
    return None, None

def _cross_key_from_fields(fields: List[str]) -> str:
    fields = [str(f) for f in fields]
    return "x" + "_".join(fields)

def _load_cross_specs_from_manifest_infer():
    """
    在候选路径中搜索 feature_cross_manifest__*.json，
    选择一个有效清单并返回 list[{'fields': [...], 'key': 'x...'}]
    """
    best = None
    best_len = -1
    for root in _search_paths_for_infer():
        try:
            for p in root.glob("feature_cross_manifest__*.json"):
                with open(p, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                xs = []
                for item in manifest.get("crosses", []):
                    fields = [str(x) for x in item.get("fields", [])]
                    fields = sorted(fields, key=lambda x: int(x) if x.isdigit() else x)
                    key = str(item.get("key", _cross_key_from_fields(fields)))
                    xs.append({"fields": fields, "key": key})
                if xs and len(xs) > best_len:
                    best = xs
                    best_len = len(xs)
                    print(f"Loaded cross manifest: {p}")
        except Exception as e:
            print(f"warn: failed scanning manifests under {root}: {e}")
    return best if best is not None else []


def _parse_cross_specs_for_infer(args) -> List[Dict]:
    """
    优先读取 args.feature_crosses / FEATURE_CROSSES；
    若为空，则兜底读取任意 feature_cross_manifest__*.json
    """
    raw = getattr(args, "feature_crosses", None)
    if raw is None:
        raw = os.environ.get("FEATURE_CROSSES", None)

    # None/空 -> 尝试从 manifest 兜底加载
    if raw is None or raw == "" or (isinstance(raw, (list, tuple)) and len(raw) == 0):
        specs = _load_cross_specs_from_manifest_infer()
        if specs:
            return specs
        print("[feature crosses] infer: disabled (feature_crosses is None/empty)")
        return []

    # 显式配置 -> 正常解析
    if isinstance(raw, str):
        items = [s for s in raw.strip().replace(",", "+").split() if s]
    else:
        items = list(raw)

    specs = []
    for s in items:
        s = s.strip().replace(",", "+")
        fields = [t for t in s.split("+") if t]
        if len(fields) >= 2:
            fields = sorted([str(f) for f in fields], key=lambda x: int(x) if x.isdigit() else x)
            specs.append({"fields": fields, "key": _cross_key_from_fields(fields)})

    uniq = {sp["key"]: sp for sp in specs}
    return list(uniq.values())

def _load_cross_maps_for_infer(args):
    """
    返回：
      - cross_specs: list[{'fields': [...], 'key': 'x...'}]
      - cross_maps: dict[key -> dict['v1_v2_...' -> id]]
    """
    specs = _parse_cross_specs_for_infer(args)
    maps = {}
    for sp in specs:
        key = sp['key']
        fname = f"item_cross_{key}_map.json"
        data, p = _load_json_from_paths_infer(fname)
        if data is not None:
            maps[key] = {str(k): int(v) for k, v in data.items()}
            sp['size'] = max(1, len(maps[key]))
            print(f"Loaded cross map {key} from {p}")
        else:
            maps[key] = {}
            sp['size'] = 0
            print(f"warn: cross map {key} not found. Will set its value to 0.")
    return specs, maps

# ==========================================================


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model, args):
    """
    生产候选库item的id和embedding，并落盘 embedding.fbin / id.u64bin
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    # 载入点击桶映射
    click_bucket_map = _load_click_bucket_map_for_infer()
    # 载入特征组合映射（修复：直接用 args + manifest 兜底）
    cross_specs, cross_maps = _load_cross_maps_for_infer(args)

    # 可选：与数据集/模型中声明的 item_cross 对齐（若存在）
    allowed = set(feat_types.get('item_cross', [])) if isinstance(feat_types, dict) else set()
    if allowed:
        cross_specs = [sp for sp in cross_specs if sp['key'] in allowed]

    def _to_int_safe(v):
        if type(v) == list:
            return int(v[0]) if len(v) > 0 else 0
        try:
            return int(v)
        except Exception:
            return 0

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0

            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            # 点击次数分桶（901）
            try:
                feature['901'] = int(click_bucket_map.get(int(item_id), 0))
            except Exception:
                feature['901'] = 0

            # 组合特征（按 specs + 对应 map 填充）
            for sp in cross_specs:
                key = sp['key']
                vals = []
                ok = True
                for fid in sp['fields']:
                    v = _to_int_safe(feature.get(fid, 0))
                    if v <= 0:
                        ok = False
                        break
                    vals.append(v)
                feature[key] = int(cross_maps.get(key, {}).get("_".join(map(str, vals)), 0)) if ok else 0

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id

def read_matrix_bin(file_path: Path, dtype=np.float32) -> np.ndarray:
    """
    读取 save_emb 写出的 .fbin/.u64bin 文件（前两个 uint32 头：num_points, dim）
    """
    with open(file_path, 'rb') as f:
        num_points = struct.unpack('I', f.read(4))[0]
        dim = struct.unpack('I', f.read(4))[0]
        arr = np.fromfile(f, dtype=dtype, count=num_points * dim)
    return arr.reshape(num_points, dim)


@torch.no_grad()
def batched_topk_torch(
    queries: torch.Tensor,         # [Q, D] (已归一化)
    items: torch.Tensor,           # [I, D] (已归一化)
    top_k: int,
    device: torch.device,
    query_bs: int,
    item_bs: int,
    use_fp16: bool = False,
) -> torch.Tensor:
    """
    纯 PyTorch 分块 Top-K 检索：
    - 双分块（查询/库）避免显存爆炸
    - 对每个查询批，跨库分块维护运行中 top-k（二次 topk 合并）
    """
    Q, D = queries.shape
    I, D2 = items.shape
    assert D == D2, f"Dim mismatch: {D} vs {D2}"

    on_cuda = (device.type == 'cuda')
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.float16 if (use_fp16 and on_cuda) else torch.float32

    out_indices = torch.empty((Q, top_k), dtype=torch.long)

    for q_start in tqdm(range(0, Q, query_bs), desc='ANN(query-chunks)'):
        q_end = min(q_start + query_bs, Q)
        q_chunk = queries[q_start:q_end].to(device=device, dtype=dtype, non_blocking=True)
        qb = q_chunk.shape[0]

        best_scores = torch.full((qb, top_k), -1e9, device=device, dtype=torch.float32)
        best_indices = torch.full((qb, top_k), -1, device=device, dtype=torch.long)

        for i_start in range(0, I, item_bs):
            i_end = min(i_start + item_bs, I)
            item_block = items[i_start:i_end].to(device=device, dtype=dtype, non_blocking=True)

            sims = torch.matmul(q_chunk, item_block.t())
            sims32 = sims.float()

            k_local = min(top_k, item_block.shape[0])
            block_scores, block_pos = torch.topk(sims32, k=k_local, dim=1)
            block_indices = (block_pos + i_start)

            comb_scores = torch.cat([best_scores, block_scores], dim=1)
            comb_indices = torch.cat([best_indices, block_indices], dim=1)
            best_scores, sel = torch.topk(comb_scores, k=top_k, dim=1)
            best_indices = torch.gather(comb_indices, 1, sel)

            del sims, sims32, block_scores, block_pos, block_indices, comb_scores, comb_indices, sel
            if on_cuda:
                torch.cuda.empty_cache()

        out_indices[q_start:q_end] = best_indices.cpu()

        del q_chunk, best_scores, best_indices
        if on_cuda:
            torch.cuda.empty_cache()

    return out_indices


def infer():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    # 设定分块默认值
    if args.torch_query_bs is None:
        args.torch_query_bs = 1024 if device.type == 'cuda' else 256
    if args.torch_item_bs is None:
        args.torch_item_bs = 16384 if device.type == 'cuda' else 8192

    data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(device)
    model.eval()

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # 生成查询（用户）向量
    all_embs = []
    user_list = []
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Encoding queries'):
        seq, token_type, seq_feat, user_id, seq_ts = batch
        seq = seq.to(device, non_blocking=True)
        with torch.no_grad():
            emb = model.predict(seq, seq_feat, token_type, seq_ts)  # [B, D]，已归一化
        all_embs.append(emb.detach().cpu().numpy().astype(np.float32))
        user_list += user_id

    all_embs = np.concatenate(all_embs, axis=0)  # [Q, D]
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))

    # 生成候选库的embedding 以及 id文件（embedding.fbin / id.u64bin）
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
        args,  # 新增
    )

    # 读取候选库向量与检索ID
    emb_path = Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin")
    id_path = Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin")
    item_embs = read_matrix_bin(emb_path, dtype=np.float32)       # [I, D]
    item_ids_u64 = read_matrix_bin(id_path, dtype=np.uint64).reshape(-1)  # [I]
    assert item_embs.shape[0] == item_ids_u64.shape[0], "embedding 与 id 数量不一致"

    # 转为 torch，准备检索
    queries_t = torch.from_numpy(all_embs)        # [Q, D]
    items_t = torch.from_numpy(item_embs)         # [I, D]

    # 分块 Top-K
    topk_idx = batched_topk_torch(
        queries=queries_t,
        items=items_t,
        top_k=args.top_k,
        device=device,
        query_bs=args.torch_query_bs,
        item_bs=args.torch_item_bs,
        use_fp16=args.use_fp16,
    )

    # 映射为检索ID -> creative_id
    top10s: List[List[int]] = []
    I = item_ids_u64.shape[0]
    for row in topk_idx.numpy():
        row_creative: List[int] = []
        for col in row:
            idx = int(col)
            if 0 <= idx < I:
                rid = int(item_ids_u64[idx])
                row_creative.append(retrieve_id2creative_id.get(rid, 0))
            else:
                row_creative.append(0)
        top10s.append(row_creative)

    return top10s, user_list