# -*- coding: utf-8 -*-
"""
RQ-VAE / RK-Means（Streaming 版，带 tqdm 进度条与 ETA）
优化版（修复 _assign_one_layer 卡住；消除 pop(0) 副作用；更稳健的 DataLoader 生命周期）

要点：
- DataLoader: persistent_workers=False，避免长时间运行中 worker 悬挂导致主线程等待
- _compute_residual: 不再对 sid_mmaps 做 pop(0)，改为按 zip 对齐遍历
- _assign_one_layer: 避免对 pinned tensor 的 numpy 视图做高级索引；写 memmap 前对 rid 排序，减少随机写放大
- 其余优化（orjson 并行解析、TF32、autocast）保留

新增能力：
- 支持逐层不同的 codebook_size（如 [1024, 512, 256, 128, 64]）
- build_semantic_id_84 可仅传 codebook_size 列表自动推断层数与 feature_ids；
  仍向后兼容：单一 int 时使用 num_layers 重复该 K。
"""

import os
import json
import math
import pickle
import contextlib
from pathlib import Path
from typing import List, Tuple, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# ---------- orjson 优先 ----------
try:
    import orjson as _fastjson
except Exception:
    _fastjson = None

# 复用 dataset 中的 JSON_LOADS（若存在），但优先 orjson
try:
    from new.dataset import JSON_LOADS as _DATASET_JSON_LOADS
except Exception:
    _DATASET_JSON_LOADS = None

def JSON_LOADS(b: bytes):
    if _fastjson is not None:
        return _fastjson.loads(b)
    if _DATASET_JSON_LOADS is not None:
        return _DATASET_JSON_LOADS(b)
    return json.loads(b.decode("utf-8", errors="ignore"))

# ---------- CUDA/TF32 性能开关 ----------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ====================== 日志与小工具 ======================

def _emit(msg: str, log_file=None):
    try:
        print(msg, flush=True)
    except Exception:
        pass
    if log_file is not None:
        try:
            log_file.write(msg + "\n")
            log_file.flush()
        except Exception:
            pass


def _pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor, force_fp32: bool = False) -> torch.Tensor:
    """
    a: [B,D], b: [K,D] -> [B,K] 的平方欧氏距离
    - 不强制 cast 到 float32，便于配合 autocast(bfloat16/fp16)
    """
    if force_fp32:
        a = a.float()
        b = b.float()
    a2 = (a * a).sum(dim=1, keepdim=True)               # [B,1]
    b2 = (b * b).sum(dim=1, keepdim=False).unsqueeze(0) # [1,K]
    d  = a @ b.t()                                      # [B,K]
    dist = (a2 + b2 - 2.0 * d).clamp_min_(0.0)
    return dist


def _open_sid_memmaps(user_cache_path: Path, num_layers: int, itemnum: int, create: bool, sid_prefix: str) -> List[np.memmap]:
    from numpy.lib.format import open_memmap
    mms = []
    user_cache_path.mkdir(parents=True, exist_ok=True)
    for i in range(num_layers):
        p = user_cache_path / f"{sid_prefix}_l{i+1}.npy"
        if create:
            mm = open_memmap(str(p), mode="w+", dtype=np.int32, shape=(itemnum + 1,))
            mm[:] = 0
            mm.flush()
        else:
            if not p.exists():
                raise FileNotFoundError(f"missing file: {p}")
            mm = open_memmap(str(p), mode="r+", dtype=np.int32, shape=(itemnum + 1,))
        mms.append(mm)
    return mms


def _estimate_tuple_collision_rate(
    sid_paths: List[Path],
    itemnum: int,
    sample_cap: int = 200_000,
    seed: int = 2025,
    ignore_zero: bool = False,
) -> float:
    """
    估计 tuple 碰撞率：1 - (唯一 tuple 数 / 样本有效 tuple 数)
    - ignore_zero=True 时，会跳过任何一层为 0 的 tuple（常用于排除“未覆盖 item”带来的全 0 元组）
    """
    if not sid_paths:
        return float("nan")
    rng = np.random.RandomState(seed)
    S = min(sample_cap, itemnum)
    if S <= 0:
        return float("nan")

    idx = rng.choice(np.arange(1, itemnum + 1, dtype=np.int64), size=S, replace=False)
    mats = [np.load(p, mmap_mode="r") for p in sid_paths]

    seen = set()
    eff = 0  # 有效样本数
    for rid in idx:
        tup = tuple(int(mm[rid]) for mm in mats)
        if ignore_zero and any(v <= 0 for v in tup):
            continue
        seen.add(tup)
        eff += 1

    if eff == 0:
        return float("nan")
    uniq = len(seen)
    return 1.0 - (float(uniq) / float(eff))

# ====================== 流式 emb_84 迭代器（DataLoader 版） ======================
SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 32, "85": 3584, "86": 3584}

class MM84Stream:
    def __init__(self,
                 data_dir: Path,
                 indexer_i: dict,
                 itemnum: int,
                 mm_id: str = "84",            # 新增：多模态特征编号
                 emb_dim: int = None,          # 可选；缺省时按 SHAPE_DICT 推断
                 num_workers: int = 8,
                 prefetch_factor: int = 8,
                 persistent_workers: bool = False):
        mm_id = str(mm_id)
        if emb_dim is None:
            if mm_id not in SHAPE_DICT:
                raise ValueError(f"unknown mm_id={mm_id}, available={list(SHAPE_DICT.keys())}")
            emb_dim = SHAPE_DICT[mm_id]

        base = data_dir / "creative_emb" / f"emb_{mm_id}_{emb_dim}"
        if not base.exists():
            raise FileNotFoundError(f"mm_emb_id={mm_id} not found under {base}")
        self.mm_id = mm_id
        self.base = base
        self.files = sorted(list(base.glob("*.json")))
        self.indexer_i = indexer_i
        self.itemnum = int(itemnum)
        self.emb_dim = int(emb_dim)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = bool(persistent_workers)

    def _build_loader(self, batch_size: int, shuffle_files: bool, seed: int, return_vec: bool) -> DataLoader:
        ds = JSON84IterableDataset(
            files=self.files, indexer_i=self.indexer_i, itemnum=self.itemnum,
            emb_dim=self.emb_dim, shuffle_files=shuffle_files, seed=seed, return_vec=return_vec
        )

        if return_vec:
            # 训练/打标 collate：拼成张量（pinned）
            def _collate_train(batch):
                if not batch:
                    return None
                rids = torch.tensor([rid for rid, _ in batch], dtype=torch.long)
                vecs = torch.from_numpy(np.stack([vec for _, vec in batch], axis=0))
                return rids, vecs.contiguous()
            collate_fn = _collate_train
        else:
            # 计数 collate：仅返回本批样本数，避免构造大数组
            def _collate_count(batch):
                return 0 if batch is None else len(batch)
            collate_fn = _collate_count

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=return_vec,               # 只在训练需要 pinned
            persistent_workers=False,            # 避免悬挂
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2,
            drop_last=False,
            collate_fn=collate_fn,
            timeout=0
        )
        return loader

    def iter_batches(self, batch_size: int, shuffle_files: bool = False, seed: int = 2025) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        loader = self._build_loader(batch_size=batch_size, shuffle_files=shuffle_files, seed=seed, return_vec=True)
        for batch in loader:
            if batch is None:
                continue
            yield batch

    def count_vectors(self, cache_to: Optional[Path] = None, refresh: bool = False, log_file=None) -> int:
        """
        并行快速计数（不构造向量）。首跑缓存到 JSON，后续复用。
        """
        if cache_to is not None and cache_to.exists() and not refresh:
            try:
                with open(cache_to, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if int(obj.get("emb_dim", -1)) == self.emb_dim:
                    return int(obj.get("count", 0))
            except Exception:
                pass

        count_bs = 65536
        loader = self._build_loader(batch_size=count_bs, shuffle_files=False, seed=2025, return_vec=False)
        total = 0
        pbar = tqdm(desc=f"Counting emb_{self.mm_id} parallel ({self.emb_dim})", dynamic_ncols=True, leave=False)
        for n in loader:
            total += int(n)
            pbar.update(int(n))
        pbar.close()

        if cache_to is not None:
            try:
                with open(cache_to, "w", encoding="utf-8") as f:
                    json.dump({"count": int(total), "emb_dim": int(self.emb_dim)}, f)
            except Exception as e:
                _emit(f"warn: failed to save count cache: {e}", log_file)
        return int(total)


# ====================== 并行 JSON 读取 IterableDataset ======================

class JSON84IterableDataset(IterableDataset):
    """
    并行顺序读取 creative_emb/emb_84_<emb_dim>/*.json。
    - return_vec=True: 产出 (rid:int, vec:np.float32[emb_dim])（训练/打标）
    - return_vec=False: 产出 rid:int（计数用，避免构造大向量，显著提速）
    """
    def __init__(self, files: List[Path], indexer_i: dict, itemnum: int, emb_dim: int = 4096,
                 shuffle_files: bool = False, seed: int = 2025, return_vec: bool = True):
        super().__init__()
        self.files = list(files)
        self.indexer_i = indexer_i
        self.itemnum = int(itemnum)
        self.emb_dim = int(emb_dim)
        self.shuffle_files = bool(shuffle_files)
        self.seed = int(seed)
        self.return_vec = bool(return_vec)

    def _iter_files_for_worker(self) -> List[Path]:
        files = list(self.files)
        if self.shuffle_files and len(files) > 1:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(files)
        wi = get_worker_info()
        if wi is None:
            return files
        per = int(math.ceil(len(files) / wi.num_workers))
        beg = wi.id * per
        end = min(beg + per, len(files))
        return files[beg:end]

    def __iter__(self):
        files = self._iter_files_for_worker()
        for p in files:
            try:
                with open(p, "rb") as fb:
                    for line in fb:
                        if not line:
                            continue
                        try:
                            obj = JSON_LOADS(line)
                            raw_id = obj["anonymous_cid"]
                            emb = obj["emb"]
                        except Exception:
                            continue
                        # 仅检查长度，计数模式下不构造 np.ndarray
                        if not isinstance(emb, (list, tuple)) or len(emb) != self.emb_dim:
                            continue
                        rid = self.indexer_i.get(raw_id, None)
                        if rid is None or not (1 <= int(rid) <= self.itemnum):
                            continue
                        if self.return_vec:
                            vec = np.asarray(emb, dtype=np.float32)
                            yield (int(rid), vec)
                        else:
                            yield int(rid)
            except Exception:
                continue

# 放在 SHAPE_DICT 附近
from typing import List, Dict, Union

def _normalize_mm_ids(mm_id: Union[str, int, List[Union[str, int]], tuple]) -> List[str]:
    if isinstance(mm_id, (list, tuple)):
        return [str(x) for x in mm_id]
    return [str(mm_id)]

def _mm_ids_to_prefix(mm_ids: List[str]) -> str:
    # 保持用户给定顺序，便于可重复
    return "_".join(mm_ids)

# 复用 JSON_LOADS 与 SHAPE_DICT
def _load_mm_raw_dict(mm_root: Path, mm_id: str, emb_dim: int) -> Dict[str, np.ndarray]:
    dct: Dict[str, np.ndarray] = {}
    if mm_id != "81":
        base = mm_root / f"emb_{mm_id}_{emb_dim}"
        if not base.exists():
            print(f"warn: mm emb path not found: {base}")
            return dct
        for json_file in base.glob("*.json"):
            try:
                with open(json_file, "rb") as f:
                    for line in f:
                        if not line:
                            continue
                        try:
                            obj = JSON_LOADS(line)
                            raw = obj["anonymous_cid"]
                            vec = np.asarray(obj["emb"], dtype=np.float32)
                            if vec.ndim == 1 and vec.shape[0] == emb_dim:
                                dct[raw] = vec
                        except Exception:
                            continue
            except Exception as e:
                print(f"warn: reading {json_file} failed: {e}")
    else:
        pkl = mm_root / f"emb_{mm_id}_{emb_dim}.pkl"
        if pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    raw_dict = pickle.load(f)
                for raw, v in raw_dict.items():
                    vec = np.asarray(v, dtype=np.float32)
                    if vec.ndim == 1 and vec.shape[0] == emb_dim:
                        dct[raw] = vec
            except Exception as e:
                print(f"warn: reading {pkl} failed: {e}")
    return dct

class MMConcatStream:
    """
    以 anchor_stream 为主迭代，同时把其它 mm_id 的向量按 rid->raw_id 查字典并逐行拼接。
    拼接顺序严格按 mm_ids_order 给定顺序。
    """
    def __init__(self,
                 anchor_stream: MM84Stream,
                 mm_ids_order: List[str],
                 emb_dims: Dict[str, int],
                 extra_dicts: Dict[str, Dict[str, np.ndarray]],
                 indexer_i_rev: Dict[int, str]):
        self.anchor_stream = anchor_stream
        self.mm_ids_order = list(mm_ids_order)
        self.emb_dims = dict(emb_dims)
        self.extra_dicts = dict(extra_dicts)  # mm_id -> {raw_id: vec}
        self.indexer_i_rev = dict(indexer_i_rev)
        self.anchor_id = anchor_stream.mm_id

    def iter_batches(self, batch_size: int, shuffle_files: bool = False, seed: int = 2025):
        for rids, vecs in self.anchor_stream.iter_batches(batch_size=batch_size,
                                                          shuffle_files=shuffle_files, seed=seed):
            B = int(vecs.size(0))
            parts: List[torch.Tensor] = []
            rid_np = rids.detach().cpu().numpy().astype(np.int64, copy=True)

            for mm in self.mm_ids_order:
                D = int(self.emb_dims[mm])
                if mm == self.anchor_id:
                    # anchor 的向量直接用
                    parts.append(vecs)
                else:
                    arr = np.zeros((B, D), dtype=np.float32)
                    dct = self.extra_dicts.get(mm, {})
                    for i in range(B):
                        raw = self.indexer_i_rev.get(int(rid_np[i]), None)
                        if raw is None:
                            continue
                        v = dct.get(raw, None)
                        if v is not None and v.shape[0] == D:
                            arr[i] = v
                    parts.append(torch.from_numpy(arr))

            combo = torch.cat(parts, dim=1).contiguous()  # [B, sumD]
            yield rids, combo


# ====================== 真·流式 Residual K-Means（含 tqdm ETA） ======================

class ResidualKMeansStream(torch.nn.Module):
    def __init__(self,
                 num_layers: int = 3,
                 codebook_size = 256,                # 支持 int 或 序列（list/tuple/np.ndarray）
                 streaming_epochs: int = 4,
                 streaming_batch_size: int = 16384,
                 streaming_eval_bs: int = 16384,
                 device: str = "cuda",
                 tolerance: float = 1e-4,
                 seed: int = 2025,
                 log_file=None,
                 writer=None,
                 verbose: bool = True,
                 total_vectors: Optional[int] = None,
                 # ---- 新增：初始化策略 ----
                 init_method: str = "balanced",  # ["balanced", "random"]
                 init_pool_multiplier: float = 4.0,  # 采样池规模 ~ 4*K
                 init_max_pool: int = 32768,  # 池上限，防 OOM
                 init_bkmeans_iters: int = 50,  # BalancedKmeans 迭代轮数（初始化用）
                 init_bkmeans_tolerance: float = 1e-3  # BalancedKmeans 收敛阈值（初始化用）
                 ):
        super().__init__()
        # 兼容：codebook_size 可为单个 int，也可为序列（分层 K）
        if isinstance(codebook_size, (list, tuple, np.ndarray)):
            self.Ks = [int(k) for k in codebook_size]
            self.L = len(self.Ks)
        else:
            self.L = int(num_layers)
            self.Ks = [int(codebook_size)] * self.L

        self.epochs = int(streaming_epochs)
        self.bs = int(streaming_batch_size)
        self.eval_bs = int(streaming_eval_bs)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.tol = float(tolerance)
        self.seed = int(seed)
        self.log_file = log_file
        self.writer = writer
        self.verbose = bool(verbose)
        self.total_vectors = total_vectors
        self.autocast_dtype = torch.bfloat16 if self.device.type == "cuda" else None
        self.codebooks: List[torch.Tensor] = []
        self._last_fit_info = {}
        # 新增：初始化超参
        self.init_method = str(init_method).lower()
        self.init_pool_multiplier = float(init_pool_multiplier)
        self.init_max_pool = int(init_max_pool)
        self.init_bkmeans_iters = int(init_bkmeans_iters)
        self.init_bkmeans_tolerance = float(init_bkmeans_tolerance)

    def _log(self, s: str):
        _emit(s, self.log_file)

    # ---- 新增：从流中收集初始化池 ----
    @torch.no_grad()
    def _gather_init_pool(self, stream: 'MM84Stream', target: int, seed: int) -> Optional[torch.Tensor]:
        buf = []
        total = 0
        for rids, vecs in stream.iter_batches(batch_size=self.bs, shuffle_files=True, seed=seed):
            buf.append(vecs)
            total += vecs.size(0)
            if total >= target:
                break
        if not buf:
            return None
        pool = torch.cat(buf, dim=0)[:target]  # [M, D]（CPU）
        return pool

    # ---- 重写初始化：每层使用对应的 K ----
    @torch.no_grad()
    def _init_centers_from_stream_once(self, stream: 'MM84Stream', D: int, device: torch.device,
                                       layer_idx: int = 0) -> torch.Tensor:
        K = int(self.Ks[layer_idx])
        # 目标池规模：min(上限, max(K, K*multiplier))
        target_M = int(min(self.init_max_pool, max(K, int(self.init_pool_multiplier * K))))
        pool_seed = self.seed + 97 * (layer_idx + 1)
        pool = self._gather_init_pool(stream, target_M, seed=pool_seed)

        if pool is None:
            self._log(f"[RKMeans/init] no pool collected. use random N({K},{D})")
            return torch.randn(K, D, device=device)

        M = int(pool.size(0))
        if self.init_method == "balanced":
            try:
                # 复用本文件中的 BalancedKmeans（在小池上跑少量迭代拿初始中心）
                bk = BalancedKmeans(
                    num_clusters=K,
                    kmeans_iters=self.init_bkmeans_iters,
                    tolerance=self.init_bkmeans_tolerance,
                    device=str(device),
                    logger=(self._log if self.verbose else None)
                )
                codebook, _labels = bk.fit(pool.to(device=device, non_blocking=True), verbose=False)
                self._log(
                    f"[RKMeans/init] BalancedKmeans on pool M={M}, iters={self.init_bkmeans_iters} -> init centers.")
                return codebook.detach().to(device, non_blocking=True)
            except Exception as e:
                self._log(f"[RKMeans/init] BalancedKmeans init failed: {e}. fallback to random subset.")

        # 回退：从池中随机抽样 K 个
        g = torch.Generator(device='cpu'); g.manual_seed(self.seed + 17 * (layer_idx + 1))
        if M < K:
            add = torch.randn(K - M, D, dtype=pool.dtype)
            init = torch.cat([pool, add], dim=0)[:K].to(device, non_blocking=True)
        else:
            idx = torch.randperm(M, generator=g)[:K]
            init = pool[idx].to(device, non_blocking=True)
        return init.clone()

    @torch.no_grad()
    def _compute_residual(self,
                          vecs: torch.Tensor,          # [B,D] (已在 CPU pinned)
                          rids: torch.Tensor,          # [B] (CPU pinned)
                          prev_codebooks_dev: List[torch.Tensor],  # [K,D] on device
                          sid_mmaps: List[np.memmap],              # prev-layer sid memmaps (CPU)
                          device: torch.device) -> torch.Tensor:
        """
        不修改 sid_mmaps（无 pop），逐层 zip 对齐；避免副作用导致的潜在阻塞。
        同时避免对 pinned tensor 的 numpy 视图做复杂操作：统一显式拷贝到普通内存。
        """
        if not prev_codebooks_dev:
            return vecs.to(device, non_blocking=True)
        res = vecs.to(device, non_blocking=True)
        # 显式复制 rid 到普通 CPU 内存（非 pinned），避免后续 numpy 高级索引潜在问题
        rid_np = rids.detach().cpu().numpy().astype(np.int64, copy=True)
        for cb, sid_mm in zip(prev_codebooks_dev, sid_mmaps):
            sid_np = np.asarray(sid_mm[rid_np], dtype=np.int64)  # 1-based
            sid_t = torch.from_numpy(sid_np).to(device=device, dtype=torch.long)
            mask = (sid_t > 0)
            if mask.any():
                idx = (sid_t[mask] - 1).clamp_min(0)
                chosen = cb.index_select(0, idx)
                res[mask] = res[mask] - chosen
        return res

    @torch.no_grad()
    def _train_one_layer(self,
                         layer_idx: int,
                         stream: MM84Stream,
                         itemnum: int,
                         prev_codebooks: List[torch.Tensor],
                         sid_mmaps_prev: List[np.memmap],
                         D: int) -> torch.Tensor:
        if self.verbose:
            self._log(f"[RKMeans/stream] === Train Layer {layer_idx+1}/{self.L} ===")
        device = self.device
        K = int(self.Ks[layer_idx])

        centers = self._init_centers_from_stream_once(stream, D, device=device, layer_idx=layer_idx)
        prev = centers.clone()

        prev_cbs_dev = [cb_cpu.to(device, non_blocking=True) for cb_cpu in prev_codebooks]
        total_items = int(self.total_vectors) if self.total_vectors is not None else None

        for ep in range(1, self.epochs + 1):
            sums = torch.zeros((K, D), device=device, dtype=torch.float32)
            cnts = torch.zeros((K,), device=device, dtype=torch.float32)

            pbar = tqdm(
                total=total_items, desc=f"L{layer_idx+1} Train epoch {ep}/{self.epochs}",
                dynamic_ncols=True, leave=False, bar_format='{l_bar}{bar}{r_bar}\n'
            ) if total_items is not None else tqdm(
                desc=f"L{layer_idx+1} Train epoch {ep}/{self.epochs}",
                dynamic_ncols=True, leave=False, bar_format='{l_bar}{bar}{r_bar}\n'
            )

            for rids, vecs in stream.iter_batches(batch_size=self.bs, shuffle_files=True, seed=self.seed + ep):
                res = self._compute_residual(vecs, rids, prev_cbs_dev, sid_mmaps_prev, device=device)
                with (torch.autocast(device_type='cuda', dtype=self.autocast_dtype) if self.autocast_dtype else contextlib.nullcontext()):
                    dists = _pairwise_sq_dists(res, centers, force_fp32=False)
                labels = torch.argmin(dists, dim=1)
                cnts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
                sums.index_add_(0, labels, res)
                bsz = int(res.size(0))
                pbar.update(bsz)

            pbar.close()

            nonempty = (cnts > 0.0)
            centers_new = centers.clone()
            if nonempty.any():
                centers_new[nonempty] = sums[nonempty] / cnts[nonempty].unsqueeze(1).clamp_min(1e-12)
            if (~nonempty).any():
                k_empty = int((~nonempty).sum().item())
                noise = 1e-2 * torch.randn((k_empty, D), device=device, dtype=torch.float32)
                src = centers[nonempty]
                if src.size(0) > 0:
                    g = torch.Generator(device=device); g.manual_seed(self.seed + 131 * ep)
                    sel = torch.randint(0, src.size(0), (k_empty,), device=device, generator=g)
                    centers_new[~nonempty] = src.index_select(0, sel) + noise
                else:
                    centers_new[~nonempty] = noise

            delta = torch.norm(centers_new - prev, p=2).item()
            empty_cnt = int((~nonempty).sum().item())
            centers = centers_new
            prev = centers_new.clone()

            if self.verbose:
                self._log(f"[RKMeans/stream] L{layer_idx+1} epoch={ep}/{self.epochs}  delta={delta:.6f}  empty={empty_cnt}")

            if delta < self.tol:
                if self.verbose:
                    self._log(f"[RKMeans/stream] L{layer_idx+1} early-stop at epoch={ep} (delta < {self.tol})")
                break

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return centers.detach().to('cpu', dtype=torch.float32)

    @torch.no_grad()
    def _assign_one_layer(self,
                          layer_idx: int,
                          stream: MM84Stream,
                          itemnum: int,
                          codebook_cpu: torch.Tensor,            # [K,D] CPU
                          prev_codebooks: List[torch.Tensor],
                          sid_mmaps_prev: List[np.memmap],
                          sid_mm_out: np.memmap,                 # [itemnum+1]
                          D: int) -> dict:
        device = self.device
        cb = codebook_cpu.to(device, non_blocking=True)
        prev_cbs_dev = [cb_cpu.to(device, non_blocking=True) for cb_cpu in prev_codebooks]
        K = int(self.Ks[layer_idx])

        counts = torch.zeros(K, dtype=torch.float64, device=device)
        Nv = 0
        sum_res_before = torch.zeros((), device=device, dtype=torch.float64)
        sum_res_after  = torch.zeros((), device=device, dtype=torch.float64)
        sum_mse        = torch.zeros((), device=device, dtype=torch.float64)

        total_items = int(self.total_vectors) if self.total_vectors is not None else None
        pbar = tqdm(
            total=total_items, desc=f"L{layer_idx+1} Assign",
            dynamic_ncols=True, leave=False, bar_format='{l_bar}{bar}{r_bar}\n'
        ) if total_items is not None else tqdm(
            desc=f"L{layer_idx+1} Assign",
            dynamic_ncols=True, leave=False, bar_format='{l_bar}{bar}{r_bar}\n'
        )

        for rids, vecs in stream.iter_batches(batch_size=self.eval_bs, shuffle_files=False, seed=self.seed + 777):
            res = self._compute_residual(vecs, rids, prev_cbs_dev, sid_mmaps_prev, device=device)

            with (torch.autocast(device_type='cuda', dtype=self.autocast_dtype) if self.autocast_dtype else contextlib.nullcontext()):
                dists = _pairwise_sq_dists(res, cb, force_fp32=False)
            labels = torch.argmin(dists, dim=1)

            # GPU 累计计数
            counts += torch.bincount(labels, minlength=K).to(torch.float64)

            # CPU 写 memmap：显式复制 rid/labels，避免 pinned 共享 + 升序写减少随机 IO
            rid_np = rids.detach().cpu().numpy().astype(np.int64, copy=True)
            labels_cpu = labels.to('cpu', non_blocking=False).contiguous()
            sid_np_out = (labels_cpu.numpy().astype(np.int32, copy=True) + 1)  # 1-based

            order = np.argsort(rid_np, kind='mergesort')  # 稳定排序
            sid_mm_out[rid_np[order]] = sid_np_out[order]

            chosen = cb.index_select(0, labels)
            diff   = (res - chosen)
            sum_res_before += (res * res).sum().to(torch.float64)
            sum_res_after  += (diff * diff).sum().to(torch.float64)
            sum_mse        += ((chosen - res) * (chosen - res)).sum().to(torch.float64)

            bsz = int(res.size(0))
            Nv += bsz
            pbar.update(bsz)

        pbar.close()

        counts_f = counts.to('cpu')
        used = int((counts_f > 0).sum().item())
        empty_rate = 1.0 - used / float(K) if K > 0 else 0.0

        if Nv > 0:
            p = (counts_f / float(Nv)).clamp(min=1e-12)
            entropy = float((-(p * torch.log(p))).sum().item())
            perplexity = float(math.exp(entropy)) if np.isfinite(entropy) else 0.0
            util = perplexity / float(K) if K > 0 else 0.0
        else:
            perplexity = util = 0.0

        sum_res_before_f = float(sum_res_before.to('cpu').item())
        sum_res_after_f  = float(sum_res_after.to('cpu').item())
        sum_mse_f        = float(sum_mse.to('cpu').item())

        if Nv > 0 and D > 0:
            mse_per_dim = sum_mse_f / float(Nv * D)
            res_before = sum_res_before_f / float(Nv * D)
            res_after  = sum_res_after_f / float(Nv * D)
            explained = float(max(0.0, (res_before - res_after) / (res_before + 1e-12)))
        else:
            mse_per_dim = res_before = res_after = explained = 0.0

        metrics = {
            "layer": layer_idx + 1, "K": int(K), "used_codes": used, "empty_rate": float(empty_rate),
            "perplexity": float(perplexity), "utilization": float(util),
            "cluster_size_min": float(counts_f.min().item()) if K > 0 else 0.0,
            "cluster_size_max": float(counts_f.max().item()) if K > 0 else 0.0,
            "cluster_size_mean": float(counts_f.mean().item()) if K > 0 else 0.0,
            "cluster_size_std": float(counts_f.float().std(unbiased=False).item()) if K > 0 else 0.0,
            "mse_per_dim": float(mse_per_dim),
            "res_norm_before": float(res_before), "res_norm_after": float(res_after),
            "explained_ratio": float(explained),
            "Nv": int(Nv),
        }

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return metrics

    @torch.no_grad()
    def fit(self,
            stream: MM84Stream,
            itemnum: int,
            user_cache_path: Path,
            feature_ids: Optional[List[str]] = None,
            sid_prefix: str = "sid84",  # 新增，形如 "sid{mm_id}"
            metric_tag_prefix: str = "sid84"  # 新增，TensorBoard tag 前缀
            ) -> Tuple[List[torch.Tensor], List[dict], float]:
        sid_mmaps = _open_sid_memmaps(user_cache_path, self.L, itemnum, create=True, sid_prefix=sid_prefix)

        D = None
        for rids, vecs in stream.iter_batches(batch_size=max(32, self.bs//4), shuffle_files=False, seed=self.seed):
            D = int(vecs.size(1))
            break
        if D is None:
            raise RuntimeError("No valid vectors found in emb_84 stream.")

        self.codebooks = []
        per_layer_metrics: List[dict] = []

        prev_cbs: List[torch.Tensor] = []
        prev_sid_mmaps: List[np.memmap] = []

        for l in range(self.L):
            if self.verbose:
                self._log(f"[RKMeans/stream] ===== Layer {l+1}/{self.L} =====")

            cb_l_cpu = self._train_one_layer(
                layer_idx=l, stream=stream, itemnum=itemnum,
                prev_codebooks=prev_cbs, sid_mmaps_prev=prev_sid_mmaps, D=D
            )
            self.codebooks.append(cb_l_cpu.clone())

            metrics_l = self._assign_one_layer(
                layer_idx=l, stream=stream, itemnum=itemnum, codebook_cpu=cb_l_cpu,
                prev_codebooks=prev_cbs, sid_mmaps_prev=prev_sid_mmaps, sid_mm_out=sid_mmaps[l], D=D
            )
            per_layer_metrics.append(metrics_l)

            if self.verbose:
                m = metrics_l
                self._log(("[RKMeans][L{l}/{L}] used={used}/{K} empty={empty:.4f} util={util:.4f} "
                           "ppx={ppx:.1f} mse={mse:.6f} res↓={exp:.4f} Nv={Nv}").format(
                    l=l+1, L=self.L, used=m["used_codes"], K=m["K"], empty=m["empty_rate"],
                    util=m["utilization"], ppx=m["perplexity"], mse=m["mse_per_dim"], exp=m["explained_ratio"], Nv=m["Nv"]
                ))
            if self.writer is not None:
                m = metrics_l
                layer = l + 1
                self.writer.add_scalar("sid84/empty_rate_by_layer", float(m["empty_rate"]), layer)
                self.writer.add_scalar("sid84/utilization_by_layer", float(m["utilization"]), layer)
                self.writer.add_scalar("sid84/perplexity_by_layer", float(m["perplexity"]), layer)
                self.writer.add_scalar("sid84/mse_per_dim_by_layer", float(m["mse_per_dim"]), layer)
                self.writer.add_scalar("sid84/explained_ratio_by_layer", float(m["explained_ratio"]), layer)

            prev_cbs.append(cb_l_cpu.clone())
            prev_sid_mmaps.append(sid_mmaps[l])

        sid_paths = [user_cache_path / f"{sid_prefix}_l{i + 1}.npy" for i in range(self.L)]

        # 1) emb_84 覆盖/缺失率（用第 1 层是否被赋值作为是否“见到向量”的近似）
        try:
            sid_l1 = np.load(sid_paths[0], mmap_mode="r")
            nonzero_l1 = int(np.count_nonzero(sid_l1[1:]))  # 排除 0 号
            emb84_covered_rate = nonzero_l1 / float(itemnum) if itemnum > 0 else float("nan")
            emb84_missing_rate = 1.0 - emb84_covered_rate if np.isfinite(emb84_covered_rate) else float("nan")
        except Exception as e:
            emb84_covered_rate = float("nan")
            emb84_missing_rate = float("nan")
            self._log(f"warn: failed computing emb84 coverage: {e}")

        # 2) 两种口径的 tuple 碰撞率
        tuple_collision_all = _estimate_tuple_collision_rate(
            sid_paths, itemnum=itemnum, sample_cap=200_000, seed=self.seed, ignore_zero=False
        )
        tuple_collision_ignore0 = _estimate_tuple_collision_rate(
            sid_paths, itemnum=itemnum, sample_cap=200_000, seed=self.seed, ignore_zero=True
        )

        if self.verbose:
            self._log(
                f"[RKMeans/stream] emb84-missing-rate ≈ {emb84_missing_rate:.6f} (covered={emb84_covered_rate:.6f})")
            self._log(f"[RKMeans/stream] tuple-collision-rate(all) ≈ {tuple_collision_all:.6f}")
            self._log(f"[RKMeans/stream] tuple-collision-rate(ignore_zero) ≈ {tuple_collision_ignore0:.6f}")

        if self.writer is not None:
            try:
                self.writer.add_scalar(f"{metric_tag_prefix}/emb_missing_rate", float(emb84_missing_rate), self.L)
                self.writer.add_scalar(f"{metric_tag_prefix}/tuple_collision_rate_all", float(tuple_collision_all), self.L)
                self.writer.add_scalar(f"{metric_tag_prefix}/tuple_collision_rate_ignore_zero", float(tuple_collision_ignore0), self.L)
                self.writer.add_scalar(f"{metric_tag_prefix}/tuple_collision_rate", float(tuple_collision_all), self.L)
            except Exception:
                pass

        # 记录到 last_fit_info
        self._last_fit_info = {
            "N": int(itemnum),
            "Nv": int(sum(m["Nv"] for m in per_layer_metrics) // max(self.L, 1)),
            "D": int(D), "num_layers": int(self.L),
            "codebook_sizes": [int(cb.shape[0]) for cb in self.codebooks],
            "tolerance": float(self.tol), "streaming_epochs": int(self.epochs),
            "streaming_batch_size": int(self.bs), "streaming_eval_bs": int(self.eval_bs),
            # 兼容旧字段 + 新字段
            "tuple_collision_rate": float(tuple_collision_all),
            "tuple_collision_rate_all": float(tuple_collision_all),
            "tuple_collision_rate_ignore_zero": float(tuple_collision_ignore0),
            "emb84_missing_rate": float(emb84_missing_rate),
            "metrics": per_layer_metrics,
        }

        # 保持返回签名不变：返回 "all" 口径作为第三个返回值
        return self.codebooks, per_layer_metrics, tuple_collision_all


# ====================== BalancedKmeans（保留） ======================
class BalancedKmeans(torch.nn.Module):
    def __init__(self, num_clusters: int, kmeans_iters: int, tolerance: float, device: str, logger=None):
        super().__init__()
        self.num_clusters = num_clusters
        self.kmeans_iters = kmeans_iters
        self.tolerance = tolerance
        self.device = device
        self._codebook = None
        self._logger = logger
    def _log(self, s: str):
        if self._logger is not None:
            self._logger(s)
        else:
            print(s)
    def _compute_distances(self, data):
        return torch.cdist(data, self._codebook)
    def _assign_clusters(self, dist: torch.Tensor) -> torch.Tensor:
        N, K = dist.shape
        labels = torch.full((N,), -1, dtype=torch.long, device=self.device)
        q, r = divmod(N, K)
        caps = torch.full((K,), q, dtype=torch.long, device=self.device)
        if r > 0:
            caps[:r] += 1
        unassigned = torch.ones(N, dtype=torch.bool, device=self.device)
        for c in range(K):
            cap_c = int(caps[c].item())
            if cap_c <= 0:
                continue
            d_c = dist[:, c].clone()
            d_c[~unassigned] = float('inf')
            k_take = min(cap_c, unassigned.sum().item())
            if k_take > 0:
                sel = torch.topk(d_c, k=k_take, largest=False).indices
                sel = sel[torch.isfinite(d_c[sel])]
                labels[sel] = c
                unassigned[sel] = False
        if unassigned.any():
            dmin, cid = dist[unassigned].min(dim=1)
            labels[unassigned] = cid
        return labels
    def _update_codebook(self, data, samples_labels):
        _new_codebook = []
        for i in range(self.num_clusters):
            cluster_data = data[samples_labels == i]
            if len(cluster_data) > 0:
                _new_codebook.append(cluster_data.mean(dim=0))
            else:
                _code = self._codebook[i]
                _new_codebook.append(_code)
        return torch.stack(_new_codebook)
    def fit(self, data, verbose: bool = True):
        num_emb, codebook_emb_dim = data.shape
        data = data.to(self.device)
        indices = torch.randperm(num_emb, device=self.device)[: self.num_clusters]
        self._codebook = data[indices].clone()
        if verbose:
            self._log(f"[BKMeans] start: N={num_emb}, D={codebook_emb_dim}, K={self.num_clusters}, iters={self.kmeans_iters}")
        for it in range(self.kmeans_iters):
            dist = self._compute_distances(data)
            samples_labels = self._assign_clusters(dist)
            inertia = torch.gather(dist, 1, samples_labels.view(-1, 1)).pow(2).mean().item()
            _new_codebook = self._update_codebook(data, samples_labels)
            delta = torch.norm(_new_codebook - self._codebook).item()
            if verbose:
                self._log(f"[BKMeans] iter={it+1:>3}/{self.kmeans_iters}  inertia={inertia:.6f}  delta={delta:.6f}")
            if delta < self.tolerance:
                if verbose:
                    self._log(f"[BKMeans] early-stop at iter={it+1} (delta < {self.tolerance})")
                self._codebook = _new_codebook
                break
            self._codebook = _new_codebook
        return self._codebook, samples_labels
    def predict(self, data):
        data = data.to(self.device)
        dist = self._compute_distances(data)
        samples_labels = self._assign_clusters(dist)
        return samples_labels


# ====================== RQ-VAE（保留以兼容） ======================

class RQEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_channels: list, latent_dim: int):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        in_dim = input_dim
        for out_dim in hidden_channels:
            stage = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU())
            self.stages.append(stage)
            in_dim = out_dim
        self.stages.append(torch.nn.Sequential(torch.nn.Linear(in_dim, latent_dim), torch.nn.ReLU()))
    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

class RQDecoder(torch.nn.Module):
    def __init__(self, latent_dim: int, hidden_channels: list, output_dim: int):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        in_dim = latent_dim
        for out_dim in hidden_channels:
            stage = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU())
            self.stages.append(stage)
            in_dim = out_dim
        self.stages.append(torch.nn.Sequential(torch.nn.Linear(in_dim, output_dim), torch.nn.ReLU()))
    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

class VQEmbedding(torch.nn.Embedding):
    def __init__(self, num_clusters, codebook_emb_dim: int, kmeans_method: str, kmeans_iters: int,
                 distances_method: str, device: str):
        super(VQEmbedding, self).__init__(num_clusters, codebook_emb_dim)
        self.num_clusters = num_clusters
        self.codebook_emb_dim = codebook_emb_dim
        self.kmeans_method = kmeans_method
        self.kmeans_iters = kmeans_iters
        self.distances_method = distances_method
        self.device = device
    def _create_codebook(self, data):
        _codebook = torch.randn(self.num_clusters, self.codebook_emb_dim)
        _codebook = _codebook.to(self.device)
        assert _codebook.shape == (self.num_clusters, self.codebook_emb_dim)
        self.codebook = torch.nn.Parameter(_codebook)
    @torch.no_grad()
    def _compute_distances(self, data):
        _codebook_t = self.codebook.t()
        assert _codebook_t.shape == (self.codebook_emb_dim, self.num_clusters)
        assert data.shape[-1] == self.codebook_emb_dim
        if self.distances_method == 'cosine':
            data_norm = F.normalize(data, p=2, dim=-1)
            _codebook_t_norm = F.normalize(_codebook_t, p=2, dim=0)
            distances = 1 - torch.mm(data_norm, _codebook_t_norm)
        else:
            data_norm_sq = data.pow(2).sum(dim=-1, keepdim=True)
            _codebook_t_norm_sq = _codebook_t.pow(2).sum(dim=0, keepdim=True)
            distances = torch.addmm(data_norm_sq + _codebook_t_norm_sq, data, _codebook_t, beta=1.0, alpha=-2.0)
        return distances
    @torch.no_grad()
    def _create_semantic_id(self, data):
        distances = self._compute_distances(data)
        _semantic_id = torch.argmin(distances, dim=-1)
        return _semantic_id
    def _update_emb(self, _semantic_id):
        update_emb = super().forward(_semantic_id)
        return update_emb
    def forward(self, data):
        self._create_codebook(data)
        _semantic_id = self._create_semantic_id(data)
        update_emb = self._update_emb(_semantic_id)
        return update_emb, _semantic_id

class RQ(torch.nn.Module):
    def __init__(self, num_codebooks: int, codebook_size: list, codebook_emb_dim, shared_codebook: bool,
                 kmeans_method, kmeans_iters, distances_method, loss_beta: float, device: str):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        assert len(self.codebook_size) == self.num_codebooks
        self.codebook_emb_dim = codebook_emb_dim
        self.shared_codebook = shared_codebook
        self.kmeans_method = kmeans_method
        self.kmeans_iters = kmeans_iters
        self.distances_method = distances_method
        self.loss_beta = loss_beta
        self.device = device
        if self.shared_codebook:
            self.vqmodules = torch.nn.ModuleList(
                [VQEmbedding(self.codebook_size[0], self.codebook_emb_dim, self.kmeans_method,
                             self.kmeans_iters, self.distances_method, self.device)
                 for _ in range(self.num_codebooks)]
            )
        else:
            self.vqmodules = torch.nn.ModuleList(
                [VQEmbedding(self.codebook_size[idx], self.codebook_emb_dim, self.kmeans_method,
                             self.kmeans_iters, self.distances_method, self.device)
                 for idx in range(self.num_codebooks)]
            )
    def quantize(self, data):
        res_emb = data.detach().clone()
        vq_emb_list, res_emb_list = [], []
        semantic_id_list = []
        vq_emb_aggre = torch.zeros_like(data)
        for i in range(self.num_codebooks):
            vq_emb, _semantic_id = self.vqmodules[i](res_emb)
            res_emb -= vq_emb
            vq_emb_aggre += vq_emb
            res_emb_list.append(res_emb)
            vq_emb_list.append(vq_emb_aggre)
            semantic_id_list.append(_semantic_id.unsqueeze(dim=-1))
        semantic_id_list = torch.cat(semantic_id_list, dim=-1)
        return vq_emb_list, res_emb_list, semantic_id_list
    def _rqvae_loss(self, vq_emb_list, res_emb_list):
        rqvae_loss_list = []
        for idx, quant in enumerate(vq_emb_list):
            loss1 = (res_emb_list[idx].detach() - quant).pow(2.0).mean()
            loss2 = (res_emb_list[idx] - quant.detach()).pow(2.0).mean()
            partial_loss = loss1 + self.loss_beta * loss2
            rqvae_loss_list.append(partial_loss)
        rqvae_loss = torch.sum(torch.stack(rqvae_loss_list))
        return rqvae_loss
    def forward(self, data):
        vq_emb_list, res_emb_list, semantic_id_list = self.quantize(data)
        rqvae_loss = self._rqvae_loss(vq_emb_list, res_emb_list)
        return vq_emb_list, semantic_id_list, rqvae_loss

class RQVAE(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_channels: list, latent_dim: int, num_codebooks: int,
                 codebook_size: list, shared_codebook: bool, kmeans_method, kmeans_iters,
                 distances_method, loss_beta: float, device: str):
        super().__init__()
        self.encoder = RQEncoder(input_dim, hidden_channels, latent_dim).to(device)
        self.decoder = RQDecoder(latent_dim, hidden_channels[::-1], input_dim).to(device)
        self.rq = RQ(num_codebooks, codebook_size, latent_dim, shared_codebook, kmeans_method,
                     kmeans_iters, distances_method, loss_beta, device).to(device)
    def encode(self, x): return self.encoder(x)
    def decode(self, z_vq):
        if isinstance(z_vq, list): z_vq = z_vq[-1]
        return self.decoder(z_vq)
    def compute_loss(self, x_hat, x_gt, rqvae_loss):
        recon_loss = F.mse_loss(x_hat, x_gt, reduction="mean")
        total_loss = recon_loss + rqvae_loss
        return recon_loss, rqvae_loss, total_loss
    def _get_codebook(self, x_gt):
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        return semantic_id_list
    def forward(self, x_gt):
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        x_hat = self.decode(vq_emb_list)
        recon_loss, rqvae_loss, total_loss = self.compute_loss(x_hat, x_gt, rqvae_loss)
        return x_hat, semantic_id_list, recon_loss, rqvae_loss, total_loss


# ====================== 顶层：构建 sid84（流式 + tqdm ETA） ======================

@torch.no_grad()
def build_semantic_id(
    data_dir: str,
    user_cache_path: str,
    mm_id: Union[str, int, List[Union[str, int]], tuple] = "83",   # 可为 str 或 list
    num_layers: int = 4,
    codebook_size = 256,
    kmeans_iters: int = 30,
    tolerance: float = 1e-4,
    device: str = "cuda",
    feature_ids: List[str] = None,
    writer=None,
    log_file=None,
    streaming: bool = True,
    streaming_epochs: int = 3,
    streaming_batch_size: int = 32768,
    streaming_eval_bs: int = 32768,
    seed: int = 2025,
    count_vectors: bool = True,
    num_workers: int = 8,
    prefetch_factor: int = 8,
    persistent_workers: bool = False,
) -> None:
    """
    当 mm_id 为列表时：按给定顺序把多个多模态向量拼接后再做 RK‑Means 量化。
    """
    os.makedirs(user_cache_path, exist_ok=True)
    mm_ids = _normalize_mm_ids(mm_id)
    for m in mm_ids:
        if m not in SHAPE_DICT:
            raise ValueError(f"unknown mm_id={m}, available={list(SHAPE_DICT.keys())}")

    sid_key   = _mm_ids_to_prefix(mm_ids)      # 例如 '84_82'
    sid_prefix = f"sid{sid_key}"
    rq_prefix  = f"rq{sid_key}"
    meta_path = Path(user_cache_path) / f"{sid_prefix}_meta.json"

    # Ks/L
    if isinstance(codebook_size, (list, tuple, np.ndarray)):
        Ks = [int(k) for k in codebook_size]
    else:
        Ks = [int(codebook_size)] * int(num_layers)
    L = len(Ks)

    # feature_ids
    if feature_ids is None:
        feature_ids = [str(940 + i) for i in range(L)]
    else:
        feature_ids = list(feature_ids)
        if len(feature_ids) < L:
            base_next = 940 + len(feature_ids)
            feature_ids += [str(base_next + i) for i in range(L - len(feature_ids))]
        elif len(feature_ids) > L:
            feature_ids = feature_ids[:L]

    out_paths = [Path(user_cache_path) / f"{sid_prefix}_l{i + 1}.npy" for i in range(L)]

    # 若已有同 Ks 联合 SID 则跳过
    skip = False
    if meta_path.exists() and all(p.exists() for p in out_paths):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_old = json.load(f)
            old_L = int(meta_old.get("num_layers", -1))
            old_Ks = list(meta_old.get("codebook_sizes", []))
            old_mm = meta_old.get("mm_id", [])
            skip = (old_L == L) and (list(map(int, old_Ks)) == Ks) and (list(old_mm) == mm_ids)
        except Exception:
            skip = False
    if skip:
        _emit(f"[build_semantic_id] outputs exist and match Ks={Ks}, mm_ids={mm_ids}. Skip.", log_file)
        if writer is not None:
            try:
                writer.add_text(f"{sid_prefix}/skip", f"outputs exist and match Ks={Ks}, mm_ids={mm_ids}", 0)
            except Exception:
                pass
        return

    # 读取 indexer
    _emit("1) 读取 indexer ...", log_file)
    with open(Path(data_dir) / "indexer.pkl", "rb") as f:
        indexer = pickle.load(f)
    indexer_i = indexer["i"]
    indexer_i_rev = {v: k for k, v in indexer_i.items()}
    itemnum = len(indexer_i)

    # 为每个 mm_id 构建单源 stream 用于计数，选覆盖率最佳的作为 anchor
    _emit(f"2) 评估各 mm_id 覆盖率并选择 anchor ...", log_file)
    mm_root = Path(data_dir) / "creative_emb"
    emb_dims = {m: int(SHAPE_DICT[m]) for m in mm_ids}

    counts = {}
    for m in mm_ids:
        try:
            s = MM84Stream(
                Path(data_dir), indexer_i=indexer_i, itemnum=itemnum,
                mm_id=m, emb_dim=emb_dims[m],
                num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
            )
            cache_to = Path(user_cache_path) / f"{sid_prefix}_stream_count__{m}.json"
            cnt = s.count_vectors(cache_to=cache_to, refresh=False, log_file=log_file) if count_vectors else None
            counts[m] = int(cnt or 0)
        except Exception as e:
            counts[m] = 0
            _emit(f"warn: count_vectors for mm_id={m} failed: {e}", log_file)

    # anchor：覆盖率最高者；若相等则按用户给定顺序优先
    anchor_id = max(mm_ids, key=lambda x: (counts.get(x, 0), -mm_ids.index(x)))
    _emit(f"   选择 anchor mm_id={anchor_id} (count={counts.get(anchor_id, 0)})", log_file)

    # 构建 anchor 流
    _emit(f"3) 构建 anchor emb_{anchor_id} 流式读取器 ...", log_file)
    anchor_stream = MM84Stream(
        Path(data_dir), indexer_i=indexer_i, itemnum=itemnum,
        mm_id=anchor_id, emb_dim=emb_dims[anchor_id],
        num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
    )

    # 其它 mm_id 仅加载 raw 字典
    _emit(f"4) 预加载其余 mm_id 的原始向量字典，用于 batch 内拼接 ...", log_file)
    extra_dicts: Dict[str, Dict[str, np.ndarray]] = {}
    for m in mm_ids:
        if m == anchor_id:
            continue
        try:
            extra_dicts[m] = _load_mm_raw_dict(mm_root, m, emb_dims[m])
            _emit(f"   loaded mm_id={m} entries={len(extra_dicts[m])}", log_file)
        except Exception as e:
            extra_dicts[m] = {}
            _emit(f"warn: load raw dict for mm_id={m} failed: {e}", log_file)

    # 拼接流（严格按用户给定顺序拼接）
    _emit("5) 构建联合多模态拼接流 ...", log_file)
    concat_stream = MMConcatStream(
        anchor_stream=anchor_stream,
        mm_ids_order=mm_ids,
        emb_dims=emb_dims,
        extra_dicts=extra_dicts,
        indexer_i_rev=indexer_i_rev
    )

    # 训练与打标
    _emit("6) 流式训练与打标 RK-Means（含 tqdm ETA）", log_file)
    total_vec_count = counts.get(anchor_id, None)
    rk = ResidualKMeansStream(
        num_layers=L,
        codebook_size=Ks,
        streaming_epochs=streaming_epochs, streaming_batch_size=streaming_batch_size,
        streaming_eval_bs=streaming_eval_bs, device=device, tolerance=tolerance,
        seed=seed, log_file=log_file, writer=writer, verbose=True,
        total_vectors=total_vec_count
    )

    codebooks, per_layer_metrics, tuple_collision = rk.fit(
        stream=concat_stream, itemnum=itemnum, user_cache_path=Path(user_cache_path),
        feature_ids=feature_ids, sid_prefix=sid_prefix, metric_tag_prefix=sid_prefix
    )

    _emit(f"7) 保存各层码本 {rq_prefix}_codebook_l*.npy", log_file)
    codebook_sizes = []
    for i, cb in enumerate(codebooks):
        cb_np = cb.numpy().astype(np.float32)
        np.save(Path(user_cache_path) / f"{rq_prefix}_codebook_l{i + 1}.npy", cb_np)
        codebook_sizes.append(int(cb_np.shape[0]))

    _emit("[build_semantic_id] metrics summary:", log_file)
    for m in per_layer_metrics:
        _emit(f"  - L{m['layer']}: empty={m['empty_rate']:.4f}, util={m['utilization']:.4f}, "
              f"ppx={m['perplexity']:.1f}, mse={m['mse_per_dim']:.6f}, res↓={m['explained_ratio']:.4f}", log_file)

    tca = float(rk._last_fit_info.get("tuple_collision_rate_all", tuple_collision))
    tci = float(rk._last_fit_info.get("tuple_collision_rate_ignore_zero", float("nan")))
    miss = float(rk._last_fit_info.get("emb84_missing_rate", float("nan")))  # 字段名沿用

    _emit(f"  - tuple_collision_rate(all) ≈ {tca:.6f}", log_file)
    _emit(f"  - tuple_collision_rate(ignore_zero) ≈ {tci:.6f}", log_file)

    meta = {
        "mm_id": mm_ids,                     # 现在存列表
        "num_layers": L,
        "codebook_sizes": codebook_sizes,
        "feature_ids": feature_ids,
        "metrics_summary": {
            "N": int(rk._last_fit_info.get("N", itemnum)),
            "Nv": int(rk._last_fit_info.get("Nv", 0)),
            "D": int(rk._last_fit_info.get("D", 0)),
            "tolerance": float(tolerance),
            "streaming_epochs": int(streaming_epochs),
            "streaming_batch_size": int(streaming_batch_size),
            "streaming_eval_bs": int(streaming_eval_bs),
            "tuple_collision_rate": float(tca),
            "tuple_collision_rate_all": float(tca),
            "tuple_collision_rate_ignore_zero": float(tci),
            "emb_missing_rate": float(miss),
            "per_layer": per_layer_metrics,
            "total_vectors": int(total_vec_count) if total_vec_count is not None else None,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(Path(user_cache_path) / f"{sid_prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta["metrics_summary"], f, ensure_ascii=False, indent=2)

    _emit(f"[build_semantic_id] saved {sid_prefix}_* , {rq_prefix}_codebook_l* , meta & metrics under {user_cache_path}", log_file)
