import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# [2026-04-24] 新增：将 HuggingFace 数据转换为本项目可直接读取的数据目录结构。
# 目标产物：
# - indexer.pkl
# - item_feat_dict.json
# - seq.jsonl + seq_offsets.pkl
# - 可选：predict_seq.jsonl + predict_seq_offsets.pkl
# - 可选：predict_set.jsonl（供 infer.py 读取）
# - 可选：creative_emb/emb_81_32.pkl（占位多模态特征）

# ================================================================
# [2026-04-24] 配置区块（后续若要适配新数据，只改这里）
# ================================================================

# 目录名约定
SEQ_DIR = "seq"
PREDICT_SEQ_DIR = "predict_seq"
USER_FEAT_DIR = "user_feat"
ITEM_FEAT_DIR = "item_feat"
CANDIDATE_DIR = "predict_set"

# 序列字段约定
USER_ID_COL = "user_id"
ITEM_ID_COL = "item_id"
TIMESTAMP_COL = "timestamp"
ACTION_COL = "action_type"
USER_FEAT_COL = "user_feat"
ITEM_FEAT_COL = "item_feat"

# 候选集字段约定
CANDIDATE_CREATIVE_ID_COL = "creative_id"
CANDIDATE_RETRIEVAL_ID_COL = "retrieval_id"
CANDIDATE_FEATURES_COL = "features"

# 行为开关
SORT_BY_TIMESTAMP = True
GENERATE_DUMMY_EMB81 = False

# ================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert HuggingFace data to O_o project format")
    p.add_argument("--input", required=True, type=str, help="Input root path")
    p.add_argument("--output", required=True, type=str, help="Output directory")
    p.add_argument("--strict", action="store_true", help="Strict mode: missing optional dirs or zero rows will raise error")
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
        if s == "":
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
            if not s:
                continue
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
    for r in ds:
        yield r


def _iter_records_from_dir(dir_path: Path) -> Iterable[Any]:
    if not dir_path.exists():
        return
    files = sorted([p for p in dir_path.rglob("*") if p.is_file()])
    for p in files:
        suffix = p.suffix.lower()
        if suffix == ".jsonl":
            for r in _iter_jsonl_file(p):
                yield r
        elif suffix == ".json":
            for r in _iter_json_file(p):
                yield r
        elif suffix == ".arrow":
            for r in _iter_arrow_file(p):
                yield r


def _extract_id_and_feat(row: Any, id_candidates: List[str], feat_candidates: List[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    if isinstance(row, dict):
        rid = None
        for k in id_candidates:
            if k in row and row[k] is not None:
                rid = str(row[k])
                break
        feat = {}
        for k in feat_candidates:
            if k in row and isinstance(row[k], dict):
                feat = _norm_feat_dict(row[k])
                break
        if not feat:
            # 若没有嵌套字段，则尝试把其余字段视作 feature_id -> value
            tmp = {}
            for k, v in row.items():
                if k in id_candidates:
                    continue
                if isinstance(v, (dict, list, int, float, str, np.integer, np.floating)):
                    tmp[str(k)] = v
            feat = _norm_feat_dict(tmp)
        return rid, feat
    return None, {}


def _load_feature_map_from_dir(dir_path: Path, is_user: bool) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not dir_path.exists():
        return out

    if is_user:
        id_candidates = ["user_id", "uid", "u", "wuid", "anonymous_uid"]
        feat_candidates = ["user_feat", "features", "feat", "feature"]
    else:
        id_candidates = ["item_id", "creative_id", "cid", "anonymous_cid", "i"]
        feat_candidates = ["item_feat", "features", "feat", "feature"]

    for row in _iter_records_from_dir(dir_path):
        rid, feat = _extract_id_and_feat(row, id_candidates=id_candidates, feat_candidates=feat_candidates)
        if rid is None:
            continue
        if rid not in out:
            out[rid] = feat
        else:
            base = out[rid]
            for k, v in feat.items():
                if k not in base:
                    base[k] = v
    return out


def _row_to_event(
    row: Any,
    user_col: str,
    item_col: str,
    ts_col: str,
    action_col: str,
    user_feat_col: str,
    item_feat_col: str,
    user_feat_map: Dict[str, Dict[str, Any]],
    item_feat_map: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    # 博文示例：record = [user_id, item_id, user_feature, item_feature, action_type, timestamp]
    if isinstance(row, (list, tuple)) and len(row) >= 6:
        u_raw = row[0]
        i_raw = row[1]
        uf = _norm_feat_dict(row[2]) if isinstance(row[2], dict) else {}
        itf = _norm_feat_dict(row[3]) if isinstance(row[3], dict) else {}
        act = _safe_int(row[4], 0)
        ts = _safe_int(row[5], 0)

        u_key = str(u_raw) if u_raw is not None else None
        i_key = str(i_raw) if i_raw is not None else None

        if not uf and u_key in user_feat_map and i_raw in (None, 0):
            uf = user_feat_map[u_key]
        if not itf and i_key in item_feat_map:
            itf = item_feat_map[i_key]

        return {
            "u_raw": u_key,
            "i_raw": i_key,
            "user_feat": uf,
            "item_feat": itf,
            "action": act,
            "ts": ts,
        }

    if isinstance(row, dict):
        u_raw = row.get(user_col, None)
        i_raw = row.get(item_col, None)
        ts = _safe_int(row.get(ts_col, 0), 0)
        act = _safe_int(row.get(action_col, 0), 0)

        uf = _norm_feat_dict(row.get(user_feat_col, {}))
        itf = _norm_feat_dict(row.get(item_feat_col, {}))

        u_key = str(u_raw) if u_raw is not None else None
        i_key = str(i_raw) if i_raw is not None else None

        # 目录模式下常见：序列里不带完整特征，使用 side-table 回填。
        if not uf and u_key in user_feat_map and i_raw in (None, 0):
            uf = user_feat_map[u_key]
        if not itf and i_key in item_feat_map:
            itf = item_feat_map[i_key]

        return {
            "u_raw": u_key,
            "i_raw": i_key,
            "user_feat": uf,
            "item_feat": itf,
            "action": act,
            "ts": ts,
        }

    return None


def _load_events_from_seq_dir(
    seq_dir: Path,
    user_col: str,
    item_col: str,
    ts_col: str,
    action_col: str,
    user_feat_col: str,
    item_feat_col: str,
    user_feat_map: Dict[str, Dict[str, Any]],
    item_feat_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for row in _iter_records_from_dir(seq_dir):
        ev = _row_to_event(
            row,
            user_col=user_col,
            item_col=item_col,
            ts_col=ts_col,
            action_col=action_col,
            user_feat_col=user_feat_col,
            item_feat_col=item_feat_col,
            user_feat_map=user_feat_map,
            item_feat_map=item_feat_map,
        )
        if ev is not None:
            events.append(ev)
    return events


def _events_to_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for e in events:
        out.append(
            {
                "user_id": e.get("u_raw", None),
                "item_id": e.get("i_raw", None),
                "timestamp": e.get("ts", 0),
                "action_type": e.get("action", 0),
                "user_feat": e.get("user_feat", {}) or {},
                "item_feat": e.get("item_feat", {}) or {},
            }
        )
    return out


def _build_id_maps(train_rows: List[Dict[str, Any]], predict_rows: List[Dict[str, Any]],
                   user_col: str, item_col: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    users = set()
    items = set()

    for rows in (train_rows, predict_rows):
        for r in rows:
            u = r.get(user_col, None)
            i = r.get(item_col, None)
            if u is not None:
                users.add(str(u))
            if i is not None:
                items.add(str(i))

    u_map = {u: idx + 1 for idx, u in enumerate(sorted(users))}
    i_map = {i: idx + 1 for idx, i in enumerate(sorted(items))}
    return u_map, i_map


def _collect_feature_vocab(rows: List[Dict[str, Any]], user_feat_col: str, item_feat_col: str) -> Dict[str, Dict[str, int]]:
    # 固定与项目对齐的特征组
    user_sparse = ["103", "104", "105", "109"]
    user_array = ["106", "107", "108", "110"]
    item_sparse = ["100", "117", "111", "118", "101", "102", "119", "120", "114", "112", "121", "115", "122", "116"]

    values: Dict[str, set] = {k: set() for k in user_sparse + user_array + item_sparse}

    for r in rows:
        uf = _norm_feat_dict(r.get(user_feat_col, {}))
        itf = _norm_feat_dict(r.get(item_feat_col, {}))

        for fid in user_sparse:
            v = _safe_int(uf.get(fid, 0), 0)
            if v > 0:
                values[fid].add(v)

        for fid in user_array:
            arr = uf.get(fid, [])
            if not isinstance(arr, list):
                arr = [arr]
            for x in arr:
                iv = _safe_int(x, 0)
                if iv > 0:
                    values[fid].add(iv)

        for fid in item_sparse:
            v = _safe_int(itf.get(fid, 0), 0)
            if v > 0:
                values[fid].add(v)

    out: Dict[str, Dict[str, int]] = {}
    for fid, s in values.items():
        sorted_vals = sorted(s)
        out[fid] = {str(v): idx + 1 for idx, v in enumerate(sorted_vals)}
    return out


def _build_item_feat_dict(rows: List[Dict[str, Any]], item_col: str, item_feat_col: str,
                          i_map: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    item_feat_by_reid: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        raw_i = r.get(item_col, None)
        if raw_i is None:
            continue
        raw_i_str = str(raw_i)
        if raw_i_str not in i_map:
            continue
        i_reid = i_map[raw_i_str]

        feat = _norm_feat_dict(r.get(item_feat_col, {}))
        if i_reid not in item_feat_by_reid:
            item_feat_by_reid[i_reid] = feat
        else:
            # 后出现的样本补齐缺失键
            base = item_feat_by_reid[i_reid]
            for k, v in feat.items():
                if k not in base:
                    base[k] = v

    return {str(k): v for k, v in item_feat_by_reid.items()}


def _build_user_sequences(rows: List[Dict[str, Any]],
                          u_map: Dict[str, int], i_map: Dict[str, int],
                          user_col: str, item_col: str, ts_col: str, action_col: str,
                          user_feat_col: str, item_feat_col: str,
                          sort_by_timestamp: bool) -> Dict[int, List[List[Any]]]:
    seqs: Dict[int, List[List[Any]]] = defaultdict(list)

    for r in rows:
        raw_u = r.get(user_col, None)
        raw_i = r.get(item_col, None)
        if raw_u is None or raw_i is None:
            continue

        u_reid = u_map.get(str(raw_u), 0)
        i_reid = i_map.get(str(raw_i), 0)
        if u_reid <= 0 or i_reid <= 0:
            continue

        ts = _safe_int(r.get(ts_col, 0), 0)
        act = _safe_int(r.get(action_col, 0), 0)
        uf = _norm_feat_dict(r.get(user_feat_col, {}))
        itf = _norm_feat_dict(r.get(item_feat_col, {}))

        rec = [u_reid, i_reid, uf, itf, act, ts]
        seqs[u_reid].append(rec)

    if sort_by_timestamp:
        for u in list(seqs.keys()):
            seqs[u].sort(key=lambda x: _safe_int(x[5], 0))

    return seqs


def _write_seq_and_offsets(path_jsonl: Path, path_offsets: Path, seqs_by_uid: Dict[int, List[List[Any]]]) -> None:
    # 以 uid 升序写，每个 uid 一行。
    offsets: List[int] = []
    with open(path_jsonl, "wb") as f:
        for uid in sorted(seqs_by_uid.keys()):
            offsets.append(f.tell())
            line = json.dumps(seqs_by_uid[uid], ensure_ascii=False).encode("utf-8") + b"\n"
            f.write(line)

    with open(path_offsets, "wb") as f:
        pickle.dump(offsets, f)


def _write_indexer(path: Path, u_map: Dict[str, int], i_map: Dict[str, int], f_map: Dict[str, Dict[str, int]]) -> None:
    obj = {
        "u": u_map,
        "i": i_map,
        "f": f_map,
    }
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _write_predict_set(path: Path, candidate_rows: List[Dict[str, Any]],
                       creative_col: str, retrieval_col: str, features_col: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in candidate_rows:
            creative = r.get(creative_col, None)
            retrieval = r.get(retrieval_col, None)
            feat = _norm_feat_dict(r.get(features_col, {}))
            if creative is None or retrieval is None:
                continue
            line = {
                "creative_id": str(creative),
                "retrieval_id": _safe_int(retrieval, 0),
                "features": feat,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _write_dummy_emb81(path: Path, item_raw_ids: Iterable[str]) -> None:
    # 与项目读取逻辑兼容：creative_emb/emb_81_32.pkl，key 是 raw item id。
    emb = {}
    z = np.zeros((32,), dtype=np.float32)
    for rid in item_raw_ids:
        emb[str(rid)] = z.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(emb, f)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_root = Path(args.input)

    train_rows: List[Dict[str, Any]]
    predict_rows: List[Dict[str, Any]]
    candidate_rows: List[Dict[str, Any]]

    if not (input_root / SEQ_DIR).exists():
        raise ValueError(f"missing seq directory: {(input_root / SEQ_DIR).as_posix()}")

    user_feat_map = _load_feature_map_from_dir(input_root / USER_FEAT_DIR, is_user=True)
    item_feat_map = _load_feature_map_from_dir(input_root / ITEM_FEAT_DIR, is_user=False)

    train_events = _load_events_from_seq_dir(
        seq_dir=input_root / SEQ_DIR,
        user_col=USER_ID_COL,
        item_col=ITEM_ID_COL,
        ts_col=TIMESTAMP_COL,
        action_col=ACTION_COL,
        user_feat_col=USER_FEAT_COL,
        item_feat_col=ITEM_FEAT_COL,
        user_feat_map=user_feat_map,
        item_feat_map=item_feat_map,
    )
    train_rows = _events_to_rows(train_events)

    if PREDICT_SEQ_DIR and (input_root / PREDICT_SEQ_DIR).exists():
        predict_events = _load_events_from_seq_dir(
            seq_dir=input_root / PREDICT_SEQ_DIR,
            user_col=USER_ID_COL,
            item_col=ITEM_ID_COL,
            ts_col=TIMESTAMP_COL,
            action_col=ACTION_COL,
            user_feat_col=USER_FEAT_COL,
            item_feat_col=ITEM_FEAT_COL,
            user_feat_map=user_feat_map,
            item_feat_map=item_feat_map,
        )
        predict_rows = _events_to_rows(predict_events)
    else:
        if args.strict:
            raise ValueError(f"missing predict sequence directory: {(input_root / PREDICT_SEQ_DIR).as_posix()}")
        predict_rows = []

    if CANDIDATE_DIR and (input_root / CANDIDATE_DIR).exists():
        candidate_rows = list(_iter_records_from_dir(input_root / CANDIDATE_DIR))
    else:
        if args.strict:
            raise ValueError(f"missing candidate directory: {(input_root / CANDIDATE_DIR).as_posix()}")
        candidate_rows = []

    print(f"Loaded rows: train={len(train_rows)}, predict={len(predict_rows)}, candidates={len(candidate_rows)}")

    u_map, i_map = _build_id_maps(
        train_rows=train_rows,
        predict_rows=predict_rows,
        user_col=USER_ID_COL,
        item_col=ITEM_ID_COL,
    )
    print(f"Built id maps: users={len(u_map)}, items={len(i_map)}")

    if args.strict and len(train_rows) == 0:
        raise ValueError("no train rows parsed from seq directory")

    f_map = _collect_feature_vocab(
        rows=train_rows + predict_rows,
        user_feat_col=USER_FEAT_COL,
        item_feat_col=ITEM_FEAT_COL,
    )
    _write_indexer(out_dir / "indexer.pkl", u_map, i_map, f_map)
    print(f"Wrote {(out_dir / 'indexer.pkl').as_posix()}")

    item_feat_dict = _build_item_feat_dict(
        rows=train_rows + predict_rows,
        item_col=ITEM_ID_COL,
        item_feat_col=ITEM_FEAT_COL,
        i_map=i_map,
    )
    with open(out_dir / "item_feat_dict.json", "w", encoding="utf-8") as f:
        json.dump(item_feat_dict, f, ensure_ascii=False)
    print(f"Wrote {(out_dir / 'item_feat_dict.json').as_posix()} (items with feats={len(item_feat_dict)})")

    train_seqs = _build_user_sequences(
        rows=train_rows,
        u_map=u_map,
        i_map=i_map,
        user_col=USER_ID_COL,
        item_col=ITEM_ID_COL,
        ts_col=TIMESTAMP_COL,
        action_col=ACTION_COL,
        user_feat_col=USER_FEAT_COL,
        item_feat_col=ITEM_FEAT_COL,
        sort_by_timestamp=SORT_BY_TIMESTAMP,
    )
    _write_seq_and_offsets(
        out_dir / "seq.jsonl",
        out_dir / "seq_offsets.pkl",
        train_seqs,
    )
    print(f"Wrote seq files for train users={len(train_seqs)}")

    if predict_rows:
        predict_seqs = _build_user_sequences(
            rows=predict_rows,
            u_map=u_map,
            i_map=i_map,
            user_col=USER_ID_COL,
            item_col=ITEM_ID_COL,
            ts_col=TIMESTAMP_COL,
            action_col=ACTION_COL,
            user_feat_col=USER_FEAT_COL,
            item_feat_col=ITEM_FEAT_COL,
            sort_by_timestamp=SORT_BY_TIMESTAMP,
        )
        _write_seq_and_offsets(
            out_dir / "predict_seq.jsonl",
            out_dir / "predict_seq_offsets.pkl",
            predict_seqs,
        )
        print(f"Wrote seq files for predict users={len(predict_seqs)}")

    if candidate_rows:
        _write_predict_set(
            out_dir / "predict_set.jsonl",
            candidate_rows,
            creative_col=CANDIDATE_CREATIVE_ID_COL,
            retrieval_col=CANDIDATE_RETRIEVAL_ID_COL,
            features_col=CANDIDATE_FEATURES_COL,
        )
        print(f"Wrote {(out_dir / 'predict_set.jsonl').as_posix()}")

    if GENERATE_DUMMY_EMB81:
        p = out_dir / "creative_emb" / "emb_81_32.pkl"
        _write_dummy_emb81(p, i_map.keys())
        print(f"Wrote dummy mm emb file {(p).as_posix()}")

    print("Done")


if __name__ == "__main__":
    main()
