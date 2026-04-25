import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm


"""
输入目录结构（HuggingFace 数据下载后本地目录，按子目录组织）：
1. seq/: 用户行为序列表。每条记录形如
     {
         "user_id": <int>,
         "seq": [
             {"item_id": <int>, "action_type": <int>, "timestamp": <int>},
             ...
         ]
     }
2. user_feat/: 用户侧特征表，至少包含 user_id 与若干特征字段。
3. item_feat/: 物品侧特征表，至少包含 item_id 与若干特征字段。
4. candidate/: （可选）候选集，至少包含 item_id 与 retrieval_id。

输出目录结构（本项目 dataset.py / infer.py 可直接读取）：
1. indexer.pkl: {"u": 原始user_id->reid, "i": 原始item_id->reid, "f": 特征值映射}
2. item_feat_dict.json: {"item_reid": {"feature_id": feature_value, ...}, ...}
3. seq.jsonl + seq_offsets.pkl: 训练序列（每行一个用户）与行偏移。
4. predict_seq.jsonl + predict_seq_offsets.pkl: （可选）推理序列。
5. predict_set.jsonl: （可选）候选库，字段为 creative_id/retrieval_id/features。
6. creative_emb/: （可选，需开启 --with-mm-emb）多模态向量目录
    - emb_81_32.pkl
    - emb_82_1024/part-xxxxx.json ... emb_86_3584/part-xxxxx.json
"""


# [2026-04-25] Clean rewrite: TencentGR-1M schema aligned + large-data friendly.

# ================================================================
# Config block (change here only if source schema changes)
# ================================================================
SEQ_DIR = "seq"
PREDICT_SEQ_DIR = "predict_seq"
USER_FEAT_DIR = "user_feat"
ITEM_FEAT_DIR = "item_feat"
CANDIDATE_DIR = "candidate"
MM_EMB_DIR = "mm_emb"

USER_ID_COL = "user_id"
ITEM_ID_COL = "item_id"
TIMESTAMP_COL = "timestamp"
ACTION_COL = "action_type"

# Candidate table columns
CANDIDATE_ITEM_ID_COL = "item_id"
CANDIDATE_RETRIEVAL_ID_COL = "retrieval_id"

# Feature ids used by baseline
ITEM_SPARSE_FEAT_IDS = [
    "100", "117", "111", "118", "101", "102", "119", "120", "114", "112", "121", "115", "122", "116"
]
USER_SPARSE_FEAT_IDS = ["103", "104", "105", "109"]
USER_ARRAY_FEAT_IDS = ["106", "107", "108", "110"]

SORT_BY_TIMESTAMP = True
GENERATE_DUMMY_EMB81 = False
VERBOSE = False

MM_EMB_DIMS = {
    "81": 32,
    "82": 1024,
    "83": 3584,
    "84": 4096,
    "85": 3584,
    "86": 3584,
}
# ================================================================


def _log(msg: str, force: bool = False) -> None:
    if not (VERBOSE or force):
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert TencentGR HF folders to O_o project format")
    p.add_argument("--input", required=True, type=str, help="Input root path")
    p.add_argument("--output", required=True, type=str, help="Output directory")
    p.add_argument("--strict", action="store_true", help="Fail when optional folders are missing")
    p.add_argument("--verbose", action="store_true", help="Enable detailed progress logs")
    p.add_argument("--with-mm-emb", action="store_true", help="Convert mm_emb directory to creative_emb outputs")
    p.add_argument(
        "--mm-emb-ids",
        nargs="+",
        default=["81", "82", "83", "84", "85", "86"],
        type=str,
        choices=[str(s) for s in range(81, 87)],
        help="Select which mm emb ids to convert, e.g. --mm-emb-ids 81 84",
    )
    p.add_argument(
        "--only-mm-emb",
        action="store_true",
        help="Only convert mm_emb -> creative_emb; skip regular seq/user/item/candidate conversion",
    )
    return p.parse_args()


def _safe_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, float):
        if np.isnan(v):
            return default
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return default
        try:
            return int(float(s))
        except Exception:
            return default
    return default


def _norm_feat_dict(d: Any) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in d.items():
        fid = str(k)
        if isinstance(v, list):
            out[fid] = [_safe_int(x, 0) for x in v]
        else:
            out[fid] = _safe_int(v, 0)
    return out


def _iter_jsonl_file(path: Path) -> Iterable[Any]:
    with open(path, "rb") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s.decode("utf-8", errors="ignore"))


def _iter_json_file(path: Path) -> Iterable[Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for x in obj:
            yield x
    else:
        yield obj


def _iter_arrow_file(path: Path) -> Iterable[Any]:
    from datasets import Dataset

    ds = Dataset.from_file(str(path))
    for row in ds:
        yield row


def _iter_parquet_file(path: Path) -> Iterable[Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError(f"Reading parquet requires pyarrow: {e}")

    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=10000):
        for row in batch.to_pylist():
            yield row


def _iter_records_from_dir(dir_path: Path) -> Iterable[Any]:
    """按目录递归读取记录，支持 jsonl/json/arrow/parquet。"""
    if not dir_path.exists():
        return
    files = sorted([p for p in dir_path.rglob("*") if p.is_file()])
    _log(f"扫描目录: {dir_path.as_posix()}，文件数={len(files)}")
    for p in files:
        suf = p.suffix.lower()
        _log(f"读取文件: {p.as_posix()}")
        if suf == ".jsonl":
            for r in _iter_jsonl_file(p):
                yield r
        elif suf == ".json":
            for r in _iter_json_file(p):
                yield r
        elif suf == ".arrow":
            for r in _iter_arrow_file(p):
                yield r
        elif suf == ".parquet":
            for r in _iter_parquet_file(p):
                yield r


def _iter_records_from_file(path: Path) -> Iterable[Any]:
    """按单文件读取记录，支持 jsonl/json/arrow/parquet。"""
    suf = path.suffix.lower()
    if suf == ".jsonl":
        for r in _iter_jsonl_file(path):
            yield r
    elif suf == ".json":
        for r in _iter_json_file(path):
            yield r
    elif suf == ".arrow":
        for r in _iter_arrow_file(path):
            yield r
    elif suf == ".parquet":
        for r in _iter_parquet_file(path):
            yield r


def _iter_mm_source_files(mm_dir: Path) -> List[Path]:
    files = []
    for p in mm_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in [".arrow", ".parquet", ".jsonl", ".json"]:
            files.append(p)
    return sorted(files)


def _collect_mm_sources(input_root: Path, selected_ids: set) -> Dict[str, List[Path]]:
    """
    自动收集 mm_emb 输入来源：
    1) 新目录形态：--input/mm_emb_<fid>_<dim>
    2) 兼容旧目录：--input/mm_emb
    返回: fid -> files, 其中 '*' 表示旧目录中的混合来源。
    """
    source_map: Dict[str, List[Path]] = {fid: [] for fid in selected_ids}

    # 新形态：每个 emb 一个独立目录
    for fid in selected_ids:
        dim = MM_EMB_DIMS.get(fid)
        if dim is None:
            continue
        d = input_root / f"mm_emb_{fid}_{dim}"
        if d.exists() and d.is_dir():
            source_map[fid].extend(_iter_mm_source_files(d))

    # 兼容旧形态：混合目录
    legacy = input_root / MM_EMB_DIR
    if legacy.exists() and legacy.is_dir():
        mixed = _iter_mm_source_files(legacy)
        if mixed:
            source_map.setdefault("*", [])
            source_map["*"].extend(mixed)

    return source_map


def _coerce_emb_vec(v: Any, dim: int) -> List[float]:
    if isinstance(v, dict):
        if "emb" in v:
            v = v["emb"]
        elif "vector" in v:
            v = v["vector"]
        elif "value" in v:
            v = v["value"]

    if isinstance(v, np.ndarray):
        arr = v.astype(np.float32).reshape(-1)
    elif isinstance(v, list):
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
    else:
        return []

    if arr.size == dim:
        return arr.tolist()
    if arr.size > dim:
        return arr[:dim].tolist()
    if arr.size == 0:
        return []
    out = np.zeros((dim,), dtype=np.float32)
    out[:arr.size] = arr
    return out.tolist()


def _extract_mm_emb_from_row(row: Any, selected_ids: set, source_fid_hint: str = "") -> Tuple[str, Dict[str, List[float]]]:
    if not isinstance(row, dict):
        return "", {}

    raw_id = ""
    for k in ["anonymous_cid", "creative_id", "item_id", "cid", "i"]:
        if k in row and row[k] is not None:
            raw_id = str(row[k])
            break

    if not raw_id:
        return "", {}

    out: Dict[str, List[float]] = {}

    feat_id = row.get("feat_id", None)
    if feat_id is not None and "emb" in row:
        fid = str(feat_id)
        if fid in MM_EMB_DIMS and fid in selected_ids:
            vec = _coerce_emb_vec(row.get("emb"), MM_EMB_DIMS[fid])
            if vec:
                out[fid] = vec

    for container_key in ["mm_emb", "features", "embeddings"]:
        c = row.get(container_key)
        if not isinstance(c, dict):
            continue
        for k, v in c.items():
            fid = str(k)
            if fid.startswith("emb_"):
                fid = fid.replace("emb_", "", 1)
            if fid in MM_EMB_DIMS and fid in selected_ids:
                vec = _coerce_emb_vec(v, MM_EMB_DIMS[fid])
                if vec:
                    out[fid] = vec

    for fid, dim in MM_EMB_DIMS.items():
        if fid not in selected_ids:
            continue
        for key in [f"emb_{fid}", fid, f"mm_emb_{fid}"]:
            if key in row:
                vec = _coerce_emb_vec(row.get(key), dim)
                if vec:
                    out[fid] = vec
                break

    # 若记录里没有显式 feat_id，但来源目录已明确对应某个 fid，则按目录 hint 兜底解析。
    if source_fid_hint and source_fid_hint in selected_ids and source_fid_hint in MM_EMB_DIMS and source_fid_hint not in out:
        vec = _coerce_emb_vec(row.get("emb", row.get("vector", row.get("value", None))), MM_EMB_DIMS[source_fid_hint])
        if vec:
            out[source_fid_hint] = vec

    return raw_id, out


def _convert_mm_emb_dir(input_root: Path, out_dir: Path, strict: bool, mm_emb_ids: List[str]) -> None:
    """
    将 mm_emb 目录转换为项目可读取的 creative_emb 结构。
    - 81: 写为 emb_81_32.pkl
    - 82~86: 写为 emb_<fid>_<dim>/part-xxxxx.json（每行一条）
    """
    selected_ids = set(mm_emb_ids)
    if not selected_ids:
        _log("未选择任何 mm_emb_id，跳过 mm_emb 转换", force=True)
        return

    source_map = _collect_mm_sources(input_root, selected_ids)
    total_files = sum(len(v) for v in source_map.values())
    if total_files == 0:
        if strict:
            raise ValueError(
                f"mm_emb has no supported files under input root: {input_root.as_posix()} "
                f"(expected mm_emb_<id>_<dim> or {MM_EMB_DIR}/)"
            )
        _log(f"输入目录下未找到可解析 mm_emb 文件，跳过: {input_root.as_posix()}", force=True)
        return

    creative_root = out_dir / "creative_emb"
    creative_root.mkdir(parents=True, exist_ok=True)
    for fid, dim in MM_EMB_DIMS.items():
        if fid == "81" or fid not in selected_ids:
            continue
        (creative_root / f"emb_{fid}_{dim}").mkdir(parents=True, exist_ok=True)

    _log(
        "开始转换 mm_emb: "
        f"files={total_files}, selected_ids={sorted(selected_ids)}, "
        f"source_dirs=mm_emb_<id>_<dim> + optional {MM_EMB_DIR}/",
        force=True,
    )

    emb81: Dict[str, np.ndarray] = {}
    per_fid_written = {fid: 0 for fid in MM_EMB_DIMS.keys()}
    total_rows = 0

    file_tasks: List[Tuple[str, Path]] = []
    for fid, f_list in source_map.items():
        for p in f_list:
            file_tasks.append((fid, p))

    for idx, (fid_hint, src) in enumerate(file_tasks, start=1):
        _log(f"mm_emb 文件 {idx}/{len(file_tasks)}: {src.as_posix()} (hint={fid_hint})", force=True)
        writers: Dict[str, Any] = {}
        try:
            for row in tqdm(_iter_records_from_file(src), desc=f"mm_emb {idx}/{len(file_tasks)}", dynamic_ncols=True, disable=not VERBOSE):
                total_rows += 1
                raw_id, mm_map = _extract_mm_emb_from_row(
                    row,
                    selected_ids=selected_ids,
                    source_fid_hint=(fid_hint if fid_hint != "*" else ""),
                )
                if not raw_id or not mm_map:
                    continue

                for fid, vec in mm_map.items():
                    if fid == "81":
                        emb81[raw_id] = np.asarray(vec, dtype=np.float32)
                        per_fid_written[fid] += 1
                    else:
                        if fid not in writers:
                            part = creative_root / f"emb_{fid}_{MM_EMB_DIMS[fid]}" / f"part-{idx:05d}.json"
                            writers[fid] = open(part, "w", encoding="utf-8")
                        line = {"anonymous_cid": raw_id, "emb": vec}
                        writers[fid].write(json.dumps(line, ensure_ascii=False) + "\n")
                        per_fid_written[fid] += 1

                if total_rows % 200000 == 0:
                    _log(
                        "mm_emb 转换进度: "
                        f"rows={total_rows}, emb81_items={len(emb81)}, "
                        f"written={{81:{per_fid_written['81']},82:{per_fid_written['82']},83:{per_fid_written['83']},84:{per_fid_written['84']},85:{per_fid_written['85']},86:{per_fid_written['86']}}}",
                        force=True,
                    )
        finally:
            for _, wf in writers.items():
                wf.close()

    if "81" in selected_ids:
        emb81_path = creative_root / "emb_81_32.pkl"
        if len(emb81) > 0:
            with open(emb81_path, "wb") as f:
                pickle.dump(emb81, f)
            _log(f"写出 mm_emb 81 文件: {emb81_path.as_posix()} (items={len(emb81)})", force=True)
        else:
            _log("未解析到 emb_81 数据，未写 emb_81_32.pkl", force=True)

    _log(
        "mm_emb 转换完成: "
        f"rows={total_rows}, written={{81:{per_fid_written['81']},82:{per_fid_written['82']},83:{per_fid_written['83']},84:{per_fid_written['84']},85:{per_fid_written['85']},86:{per_fid_written['86']}}}",
        force=True,
    )


def _extract_id_and_feat(row: Any, id_candidates: List[str], feat_candidates: List[str]) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(row, dict):
        return "", {}

    rid = ""
    for k in id_candidates:
        if k in row and row[k] is not None:
            rid = str(row[k])
            break

    feat: Dict[str, Any] = {}
    for k in feat_candidates:
        if k in row and isinstance(row[k], dict):
            feat = _norm_feat_dict(row[k])
            break

    if not feat:
        tmp: Dict[str, Any] = {}
        for k, v in row.items():
            if k in id_candidates:
                continue
            if isinstance(v, (dict, list, int, float, str, np.integer, np.floating)):
                tmp[str(k)] = v
        feat = _norm_feat_dict(tmp)

    return rid, feat


def _load_feature_map_from_dir(dir_path: Path, is_user: bool) -> Dict[str, Dict[str, Any]]:
    """加载 user_feat / item_feat 为内存字典，供后续快速查特征。"""
    out: Dict[str, Dict[str, Any]] = {}
    if not dir_path.exists():
        return out

    if is_user:
        id_candidates = ["user_id", "uid", "u", "wuid", "anonymous_uid"]
        feat_candidates = ["user_feat", "features", "feat", "feature"]
        role = "user"
    else:
        id_candidates = ["item_id", "creative_id", "cid", "anonymous_cid", "i"]
        feat_candidates = ["item_feat", "features", "feat", "feature"]
        role = "item"

    _log(f"开始加载{role}特征映射: {dir_path.as_posix()}")
    n_rows = 0
    for row in _iter_records_from_dir(dir_path):
        n_rows += 1
        rid, feat = _extract_id_and_feat(row, id_candidates, feat_candidates)
        if not rid:
            continue
        if rid not in out:
            out[rid] = feat
        else:
            base = out[rid]
            for k, v in feat.items():
                if k not in base:
                    base[k] = v
        if n_rows % 200000 == 0:
            _log(f"{role}特征解析进度: rows={n_rows}, unique_ids={len(out)}")

    _log(f"{role}特征映射加载完成: rows={n_rows}, unique_ids={len(out)}")
    return out


def _scan_seq_ids_from_dir(seq_dir: Path) -> Tuple[set, set, int]:
    """第一遍扫描 seq，仅提取 user_id/item_id 集合，避免占用大内存。"""
    users = set()
    items = set()
    rows = 0
    _log("第一遍扫描 seq：提取 user/item ID")

    for row in _iter_records_from_dir(seq_dir):
        rows += 1
        if isinstance(row, dict) and isinstance(row.get("seq"), list):
            u = row.get(USER_ID_COL)
            if u is not None:
                users.add(str(u))
            for ev in row.get("seq", []):
                if isinstance(ev, dict):
                    i = ev.get(ITEM_ID_COL)
                    if i is not None:
                        items.add(str(i))
        elif isinstance(row, dict):
            u = row.get(USER_ID_COL)
            i = row.get(ITEM_ID_COL)
            if u is not None:
                users.add(str(u))
            if i is not None:
                items.add(str(i))

        if rows % 200000 == 0:
            _log(f"第一遍进度: rows={rows}, users={len(users)}, items={len(items)}")

    _log(f"第一遍完成: rows={rows}, users={len(users)}, items={len(items)}")
    return users, items, rows


def _build_feature_vocab_from_maps(user_feat_map: Dict[str, Dict[str, Any]], item_feat_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """从 user/item 特征映射中构建 indexer['f'] 的离散值重映射。"""
    values: Dict[str, set] = {k: set() for k in USER_SPARSE_FEAT_IDS + USER_ARRAY_FEAT_IDS + ITEM_SPARSE_FEAT_IDS}

    _log("构建特征词表（indexer['f']）")
    for _, uf in tqdm(user_feat_map.items(), total=len(user_feat_map), desc="f_vocab(user)", dynamic_ncols=True):
        for fid in USER_SPARSE_FEAT_IDS:
            v = _safe_int(uf.get(fid, 0), 0)
            if v > 0:
                values[fid].add(v)
        for fid in USER_ARRAY_FEAT_IDS:
            arr = uf.get(fid, [])
            if not isinstance(arr, list):
                arr = [arr]
            for x in arr:
                iv = _safe_int(x, 0)
                if iv > 0:
                    values[fid].add(iv)

    for _, itf in tqdm(item_feat_map.items(), total=len(item_feat_map), desc="f_vocab(item)", dynamic_ncols=True):
        for fid in ITEM_SPARSE_FEAT_IDS:
            v = _safe_int(itf.get(fid, 0), 0)
            if v > 0:
                values[fid].add(v)

    out: Dict[str, Dict[str, int]] = {}
    for fid, s in values.items():
        out[fid] = {str(v): idx + 1 for idx, v in enumerate(sorted(s))}
    return out


def _write_item_feat_dict_from_map(path: Path, i_map: Dict[str, int], item_feat_map: Dict[str, Dict[str, Any]]) -> int:
    """按 item reid 写 item_feat_dict.json（流式写，减少中间对象）。"""
    cnt = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("{")
        first = True
        for raw_i, rid in tqdm(i_map.items(), total=len(i_map), desc="write item_feat_dict", dynamic_ncols=True):
            feat = item_feat_map.get(str(raw_i), {})
            if not isinstance(feat, dict):
                feat = {}
            if not first:
                f.write(",")
            first = False
            f.write(json.dumps(str(rid), ensure_ascii=False))
            f.write(":")
            f.write(json.dumps(feat, ensure_ascii=False))
            cnt += 1
        f.write("}")
    return cnt


def _write_seq_from_hf_seq_dir(
    seq_dir: Path,
    out_jsonl: Path,
    out_offsets: Path,
    u_map: Dict[str, int],
    i_map: Dict[str, int],
    user_feat_map: Dict[str, Dict[str, Any]],
    item_feat_map: Dict[str, Dict[str, Any]],
    sort_by_timestamp: bool,
) -> int:
    """
    第二遍扫描 seq 并直接写目标 seq.jsonl 与 seq_offsets.pkl。
    输出单条 record 结构：
        [user_reid, item_reid, user_feat(dict), item_feat(dict), action_type, timestamp]
    """
    offsets: List[int] = []
    written = 0

    with open(out_jsonl, "wb") as f:
        for row in _iter_records_from_dir(seq_dir):
            if not (isinstance(row, dict) and isinstance(row.get("seq"), list)):
                continue

            raw_u = row.get(USER_ID_COL)
            if raw_u is None:
                continue
            u_key = str(raw_u)
            u_reid = u_map.get(u_key, 0)
            if u_reid <= 0:
                continue

            uf = user_feat_map.get(u_key, {})
            recs: List[List[Any]] = []
            for ev in row.get("seq", []):
                if not isinstance(ev, dict):
                    continue
                raw_i = ev.get(ITEM_ID_COL)
                if raw_i is None:
                    continue
                i_key = str(raw_i)
                i_reid = i_map.get(i_key, 0)
                if i_reid <= 0:
                    continue

                itf = item_feat_map.get(i_key, {})
                a = _safe_int(ev.get(ACTION_COL, 0), 0)
                ts = _safe_int(ev.get(TIMESTAMP_COL, 0), 0)
                recs.append([u_reid, i_reid, uf, itf, a, ts])

            if sort_by_timestamp:
                recs.sort(key=lambda x: _safe_int(x[5], 0))

            offsets.append(f.tell())
            f.write(json.dumps(recs, ensure_ascii=False).encode("utf-8") + b"\n")
            written += 1
            if written % 200000 == 0:
                _log(f"写入 seq 进度: users={written}")

    with open(out_offsets, "wb") as ff:
        pickle.dump(offsets, ff)

    return written


def _extract_candidate_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """提取 candidate 的稀疏特征字典，兼容 features 嵌套和 feature_value 结构。"""
    if not isinstance(row, dict):
        return {}

    if "features" in row and isinstance(row.get("features"), dict):
        return _norm_feat_dict(row.get("features", {}))

    out: Dict[str, Any] = {}
    for fid in ITEM_SPARSE_FEAT_IDS:
        if fid not in row:
            continue
        v = row.get(fid)
        if isinstance(v, dict):
            if "feature_value" in v:
                out[fid] = v.get("feature_value")
            elif "value" in v:
                out[fid] = v.get("value")
            else:
                out[fid] = 0
        else:
            out[fid] = v

    return _norm_feat_dict(out)


def _write_indexer(path: Path, u_map: Dict[str, int], i_map: Dict[str, int], f_map: Dict[str, Dict[str, int]]) -> None:
    with open(path, "wb") as f:
        pickle.dump({"u": u_map, "i": i_map, "f": f_map}, f)


def _write_predict_set_from_dir(dir_path: Path, out_path: Path) -> int:
    """将 candidate 表转换为 infer.py 需要的 predict_set.jsonl。"""
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in _iter_records_from_dir(dir_path):
            if not isinstance(r, dict):
                continue
            creative = r.get(CANDIDATE_ITEM_ID_COL)
            retrieval = r.get(CANDIDATE_RETRIEVAL_ID_COL)
            if creative is None or retrieval is None:
                continue
            feat = _extract_candidate_features(r)
            line = {
                "creative_id": str(creative),
                "retrieval_id": _safe_int(retrieval, 0),
                "features": feat,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            n += 1
            if n % 200000 == 0:
                _log(f"写入 predict_set 进度: rows={n}")
    return n


def _write_dummy_emb81(path: Path, item_raw_ids: Iterable[str]) -> None:
    emb = {}
    z = np.zeros((32,), dtype=np.float32)
    for rid in item_raw_ids:
        emb[str(rid)] = z.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(emb, f)


def main() -> None:
    global VERBOSE
    args = parse_args()
    VERBOSE = args.verbose
    input_root = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only_mm_emb:
        _log("仅 mm_emb 模式：跳过常规转换", force=True)
        _convert_mm_emb_dir(
            input_root,
            out_dir,
            strict=args.strict,
            mm_emb_ids=args.mm_emb_ids,
        )
        _log("全部阶段完成", force=True)
        print("Done")
        return

    _log("阶段 1/7：校验输入目录", force=True)
    if not (input_root / SEQ_DIR).exists():
        raise ValueError(f"missing seq directory: {(input_root / SEQ_DIR).as_posix()}")
    _log(
        "输入目录结构约定: "
        f"{SEQ_DIR}/, {USER_FEAT_DIR}/, {ITEM_FEAT_DIR}/, 可选 {PREDICT_SEQ_DIR}/, 可选 {CANDIDATE_DIR}/"
        , force=True
    )
    _log(
        "输出文件结构: indexer.pkl, item_feat_dict.json, seq.jsonl, seq_offsets.pkl, "
        "可选 predict_seq.jsonl/predict_seq_offsets.pkl, 可选 predict_set.jsonl"
        , force=True
    )

    _log("阶段 2/7：加载 user/item 特征侧表", force=True)
    user_feat_map = _load_feature_map_from_dir(input_root / USER_FEAT_DIR, is_user=True)
    item_feat_map = _load_feature_map_from_dir(input_root / ITEM_FEAT_DIR, is_user=False)

    _log("阶段 3/7：第一遍扫描序列 ID", force=True)
    train_users, train_items, _ = _scan_seq_ids_from_dir(input_root / SEQ_DIR)
    predict_users, predict_items = set(), set()
    if (input_root / PREDICT_SEQ_DIR).exists():
        predict_users, predict_items, _ = _scan_seq_ids_from_dir(input_root / PREDICT_SEQ_DIR)
    elif args.strict:
        raise ValueError(f"missing predict sequence directory: {(input_root / PREDICT_SEQ_DIR).as_posix()}")

    _log("阶段 4/7：构建 user/item 重映射", force=True)
    all_users = train_users | predict_users | set(user_feat_map.keys())
    all_items = train_items | predict_items | set(item_feat_map.keys())
    u_map = {u: idx + 1 for idx, u in enumerate(sorted(all_users))}
    i_map = {i: idx + 1 for idx, i in enumerate(sorted(all_items))}
    print(f"Built id maps: users={len(u_map)}, items={len(i_map)}")

    if args.strict and len(train_users) == 0:
        raise ValueError("no train rows parsed from seq directory")

    _log("阶段 5/7：构建特征词表并写 indexer/item_feat_dict", force=True)
    f_map = _build_feature_vocab_from_maps(user_feat_map=user_feat_map, item_feat_map=item_feat_map)
    _write_indexer(out_dir / "indexer.pkl", u_map, i_map, f_map)
    print(f"Wrote {(out_dir / 'indexer.pkl').as_posix()}")

    item_cnt = _write_item_feat_dict_from_map(out_dir / "item_feat_dict.json", i_map=i_map, item_feat_map=item_feat_map)
    print(f"Wrote {(out_dir / 'item_feat_dict.json').as_posix()} (items with feats={item_cnt})")

    _log("阶段 6/7：第二遍写训练序列文件", force=True)
    train_written = _write_seq_from_hf_seq_dir(
        seq_dir=input_root / SEQ_DIR,
        out_jsonl=out_dir / "seq.jsonl",
        out_offsets=out_dir / "seq_offsets.pkl",
        u_map=u_map,
        i_map=i_map,
        user_feat_map=user_feat_map,
        item_feat_map=item_feat_map,
        sort_by_timestamp=SORT_BY_TIMESTAMP,
    )
    print(f"Wrote seq files for train users={train_written}")

    if (input_root / PREDICT_SEQ_DIR).exists():
        _log("阶段 7/7：第二遍写推理序列文件", force=True)
        pred_written = _write_seq_from_hf_seq_dir(
            seq_dir=input_root / PREDICT_SEQ_DIR,
            out_jsonl=out_dir / "predict_seq.jsonl",
            out_offsets=out_dir / "predict_seq_offsets.pkl",
            u_map=u_map,
            i_map=i_map,
            user_feat_map=user_feat_map,
            item_feat_map=item_feat_map,
            sort_by_timestamp=SORT_BY_TIMESTAMP,
        )
        print(f"Wrote seq files for predict users={pred_written}")

    if (input_root / CANDIDATE_DIR).exists():
        n_candidate = _write_predict_set_from_dir(input_root / CANDIDATE_DIR, out_dir / "predict_set.jsonl")
        print(f"Wrote {(out_dir / 'predict_set.jsonl').as_posix()} (rows={n_candidate})")
    elif args.strict:
        raise ValueError(f"missing candidate directory: {(input_root / CANDIDATE_DIR).as_posix()}")

    if args.with_mm_emb:
        _log("阶段 8/8：转换 mm_emb 到 creative_emb", force=True)
        _convert_mm_emb_dir(
            input_root,
            out_dir,
            strict=args.strict,
            mm_emb_ids=args.mm_emb_ids,
        )

    if GENERATE_DUMMY_EMB81:
        p = out_dir / "creative_emb" / "emb_81_32.pkl"
        _write_dummy_emb81(p, i_map.keys())
        print(f"Wrote dummy mm emb file {(p).as_posix()}")

    _log("全部阶段完成", force=True)
    print("Done")


if __name__ == "__main__":
    main()
