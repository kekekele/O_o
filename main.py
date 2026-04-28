import argparse
import json
import os
import math
import time
import random
import traceback
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

# [2026-04-24] 设备兼容改造：可选导入 torch_npu；未安装时保持原行为。
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

from dataset import MyDataset
from model import BaselineModel

# os.environ.setdefault("TRAIN_LOG_PATH", "./logs")
# os.environ.setdefault("TRAIN_TF_EVENTS_PATH", "./logs/tf_events")
# os.environ.setdefault("TRAIN_DATA_PATH", "./data/TencentGR_1k")
# os.environ.setdefault("TRAIN_CKPT_PATH", "./result")


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _stage_log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _to_fp32_feat_dict(feat_dict):
    if not isinstance(feat_dict, dict):
        return feat_dict
    out = {}
    for k, v in feat_dict.items():
        if torch.is_tensor(v):
            out[k] = v.float()
        else:
            out[k] = v
    return out


def _tensor_brief_stats(x: torch.Tensor):
    if not torch.is_tensor(x):
        return {"type": str(type(x))}
    t = x.detach()
    stat = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
    }
    try:
        if t.numel() > 0:
            if t.dtype.is_floating_point:
                tf = t.float()
                stat.update({
                    "min": float(tf.min().item()),
                    "max": float(tf.max().item()),
                    "nan": int(torch.isnan(tf).sum().item()),
                    "inf": int(torch.isinf(tf).sum().item()),
                })
            else:
                stat.update({
                    "min": int(t.min().item()),
                    "max": int(t.max().item()),
                })
    except Exception as e:
        stat["stat_error"] = str(e)
    return stat


def _in_debug_window(global_step: int, start_step: int, end_step: int) -> bool:
    if start_step < 0:
        return False
    if global_step < start_step:
        return False
    if end_step >= 0 and global_step > end_step:
        return False
    return True


def _dump_step_debug(log_dir: Path, payload: dict) -> None:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        p = log_dir / f"step_debug_{payload.get('global_step', -1)}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _stage_log(f"已写入调试快照: {p}")
    except Exception as e:
        _stage_log(f"写调试快照失败: {e}")


def _safe_scalar(x) -> float:
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().cpu().item())
        return float(x)
    except Exception:
        return float('nan')

def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--seed', default=20252026, type=int)
    # Baseline Model construction
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_units', default=512, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)

    parser.add_argument('--temperature', default=0.025, type=float)
    parser.add_argument('--neg_pop_alpha', default=0, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--feature_crosses', nargs='*', default=["118+120", "116+118"],
                        help='例如: ["118+120"]；缺省(None)表示不使用交叉特征')

    # AMP: 混合精度训练
    parser.add_argument('--amp', default='auto',
        choices=['off', 'fp16', 'bf16', 'auto'],
        help='混合精度模式：off 关闭；fp16 半精度；bf16 bfloat16；auto 优先 bf16，不支持则回退 fp16'
    )
    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # ==================== 自监督（由 dataset 进行增广，这里只做损失） ====================
    parser.add_argument('--ssl', default='rfm_no_compl', choices=['none', 'rfm_no_compl'], help='自监督增广方式（dataset 内实现）；none 关闭')
    parser.add_argument('--ssl_alpha', default=0.5, type=float, help='SSL 损失权重alpha')
    parser.add_argument('--ssl_mask_ratio', default=0.6, type=float, help='RFM 域级掩蔽比例（传入 dataset）')
    parser.add_argument('--ssl_value_dropout', default=0.3, type=float, help='多值特征值级 dropout 概率（传入 dataset）')
    parser.add_argument('--log_interval', default=50, type=int, help='训练心跳日志间隔（step）')
    parser.add_argument('--debug_start_step', default=-1, type=int, help='开启精细调试的起始 global_step（<0 关闭）')
    parser.add_argument('--debug_end_step', default=-1, type=int, help='开启精细调试的结束 global_step（<0 表示不设上限）')
    parser.add_argument('--debug_sync_every', default=1, type=int, help='调试窗口内每多少步做一次 npu synchronize（仅NPU）')
    parser.add_argument('--debug_check_indices', action='store_true', help='调试窗口内检查 item/user 索引范围并打印异常')
    parser.add_argument('--debug_check_model_finite', action='store_true', help='调试时检查模型前向各层是否出现非有限值')
    parser.add_argument('--npu_safe_copy', action='store_true', help='NPU 排障模式：关闭 pin_memory，并使用阻塞式 .to() 拷贝')

    args = parser.parse_args()
    return args


# [2026-04-24] 设备兼容改造：统一设备检测与解析，不改变 CUDA/CPU 既有逻辑。
def _npu_available() -> bool:
    return hasattr(torch, 'npu') and torch.npu.is_available()


def _resolve_runtime_device(device_str: str) -> torch.device:
    d = str(device_str).lower()
    if d == 'cpu':
        return torch.device('cpu')
    if d.startswith('cuda'):
        return torch.device(device_str) if torch.cuda.is_available() else torch.device('cpu')
    if d.startswith('npu'):
        return torch.device(device_str) if _npu_available() else torch.device('cpu')
    return torch.device('cpu')


def _pin_memory_for_device(device: torch.device, npu_safe_copy: bool = False) -> bool:
    # NPU 排障模式下关闭 pin_memory，规避 copy_stream 异步拷贝链路不稳定。
    if device.type == 'npu' and npu_safe_copy:
        return False
    return device.type == 'cuda'


def _autocast_ctx(device_type: str, dtype, enabled: bool):
    if not enabled:
        return torch.autocast(device_type='cpu', enabled=False)
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)


def _build_grad_scaler(device_type: str, enabled: bool):
    # 兼容不同 PyTorch 版本：优先 torch.amp.GradScaler，失败则回退 CUDA 旧接口。
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _bf16_supported(device_type: str) -> bool:
    if device_type == 'cuda':
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return False
    if device_type == 'npu':
        # Ascend 侧优先尝试官方查询接口；若不可用，默认允许 bf16（与 NPU 优先策略一致）。
        try:
            if hasattr(torch, 'npu'):
                fn = getattr(torch.npu, 'is_bf16_supported', None)
                if callable(fn):
                    return bool(fn())
        except Exception:
            pass
        return True
    return False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # [2026-04-24] 设备兼容改造：NPU 可用时补充种子设置。
    if _npu_available():
        try:
            torch.npu.manual_seed(seed)
            torch.npu.manual_seed_all(seed)
        except Exception:
            pass


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_weights(m: torch.nn.Module):
    if isinstance(m, torch.nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.padding_idx is not None:
            with torch.no_grad():
                m.weight[m.padding_idx].fill_(0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, (torch.nn.LayerNorm, getattr(torch.nn, 'RMSNorm', ()))):
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def InfoNCE(
    pos_embs: torch.Tensor,           # [B, L, D]
    neg_embs: torch.Tensor,           # [B, L, D]
    log_feats: torch.Tensor,          # [B, L, D]
    temperature: float,               # > 0
    next_token_type: torch.Tensor,    # [B, L]，1 表示 item
    next_action_type: torch.Tensor,   # [B, L]，下一个token动作类型，0表示曝光，1表示点击
    click_scale: float = 1.0,
    exp_scale: float = 0.1,
    pos_ids: torch.Tensor = None,     # [B, L] 正样本的 item_id（与 pos_embs 对齐，用于 -logQ 修正）
    item_logQ: torch.Tensor = None,   # [itemnum+1] 的 logQ
    filter_current_only: bool = True,
    neg_ids: torch.Tensor = None,     # [B, L] 与 neg_embs 对齐
):
    """
    Batch-all negatives InfoNCE + −logQ 修正。
    变更：不再对子采样难负样本；难负样本=“全 batch 内除本序列以外的所有正样本”。
    要求：pos_ids、neg_ids、item_logQ 均有效；item_logQ 建议为 α*logQ 以匹配 Q^α 采样。
    """
    assert temperature > 0.0, "temperature must be > 0"
    assert item_logQ is not None, "item_logQ is required for -logQ correction"
    assert pos_ids is not None and neg_ids is not None, "pos_ids and neg_ids are required for -logQ correction"

    # 仅对 item 位置计算
    mask = (next_token_type == 1)
    if mask.dtype is not torch.bool:
        mask = mask.bool()

    Qn    = log_feats[mask]          # [M, D]
    Kpn   = pos_embs[mask]           # [M, D]
    Knegn = neg_embs[mask]           # [M, D] 共享负样本池（来自 batch 内）
    act   = next_action_type[mask]   # [M]

    device = Qn.device
    M = Qn.size(0)
    if M == 0:
        return torch.zeros((), device=device), {
            'mean_pos_sim': 0.0, 'mean_neg_sim': 0.0, 'mean_hard_neg_sim': 0.0,
            'mask_rate_easy': 0.0, 'mask_rate_hard': 0.0,
        }

    # 对应的 item ids
    pos_ids_flat = pos_ids[mask].long()   # [M]
    neg_ids_pool = neg_ids[mask].long()   # [M]

    # 正样本 logits：sim/T - logQ[pos]
    pos_sim    = (Qn * Kpn).sum(dim=-1, keepdim=True)                  # [M,1]
    logQ_pos   = item_logQ[pos_ids_flat].to(device=device, dtype=pos_sim.dtype).unsqueeze(1)  # [M,1]
    pos_logits = (pos_sim / temperature) - logQ_pos                    # [M,1]

    # 批内“易负样本”池 logits：sim/T - logQ[neg_column]
    neg_sim    = Qn @ Knegn.t()                                        # [M,M]
    logQ_cols  = item_logQ[neg_ids_pool].to(device=device, dtype=neg_sim.dtype)               # [M]
    neg_logits = (neg_sim / temperature) - logQ_cols.unsqueeze(0)      # [M,M]

    # 点击/曝光样本级别权重
    act_f = act.to(Qn.dtype)
    sample_weight = torch.where(
        act_f > 0.5, torch.full_like(act_f, click_scale), torch.full_like(act_f, exp_scale)
    )  # [M]
    sample_weight = sample_weight / sample_weight.sum().clamp_min(1e-12)

    # 过滤“易负样本”：同 id 屏蔽
    neg_large = torch.tensor(-1e9, device=device, dtype=pos_logits.dtype)
    invalid_easy = torch.zeros((M, M), dtype=torch.bool, device=device)
    if filter_current_only:
        invalid_easy = pos_ids_flat.view(M, 1).eq(neg_ids_pool.view(1, M))
        if invalid_easy.any():
            neg_logits = neg_logits.masked_fill(invalid_easy, neg_large)

    # ==================== 全量“难负样本”= 其他序列的正样本 ====================
    # 找到每个被计算位置对应的 batch 序列索引（第 0 维）
    idxs = mask.nonzero(as_tuple=False)    # [M, 2], 每行=(b, l)
    seq_idx = idxs[:, 0]                   # [M]

    # 构造全量难负样本矩阵，并做 −logQ 修正（按列对应正样本 id）
    hard_neg_sim_full = Qn @ Kpn.t()       # [M,M]
    logQ_hard_cols = item_logQ[pos_ids_flat].to(device=device, dtype=hard_neg_sim_full.dtype)  # [M]
    hard_logits = (hard_neg_sim_full / temperature) - logQ_hard_cols.unsqueeze(0)              # [M,M]

    # 屏蔽“同一序列”的列（包含自身），仅保留其他序列的正样本为难负
    invalid_hard = seq_idx.view(M, 1).eq(seq_idx.view(1, M))  # True 表示需屏蔽
    if invalid_hard.any():
        hard_logits = hard_logits.masked_fill(invalid_hard, neg_large)
    mask_rate_hard = float(invalid_hard.float().mean().item())

    # 拼接分母：正样本 + 易负样本 + 难负样本（全量）
    logits = torch.cat([pos_logits, neg_logits, hard_logits], dim=1)  # [M, 1+M+M]
    labels = torch.zeros(M, dtype=torch.long, device=device)

    # 损失
    per_example_loss = F.cross_entropy(logits, labels, reduction='none')  # [M]
    loss = (per_example_loss * sample_weight).sum()

    # 统计：难负样本仅统计未屏蔽对
    if (~invalid_hard).any():
        mean_hard_neg_sim = float(hard_neg_sim_full[~invalid_hard].mean().item())
    else:
        mean_hard_neg_sim = 0.0

    stats = {
        'mean_pos_sim': float(pos_sim.mean().item()),
        'mean_neg_sim': float(neg_sim.mean().item()),
        'mean_hard_neg_sim': mean_hard_neg_sim,
        'mask_rate_easy': float(invalid_easy.float().mean().item()),
        'mask_rate_hard': mask_rate_hard,
    }
    return loss, stats


def ssl_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    assert temperature > 0.0
    if z1.numel() == 0 or z2.numel() == 0:
        return torch.zeros((), device=z1.device)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    M = z1.size(0)
    logits12 = (z1 @ z2.t()) / float(temperature)
    labels = torch.arange(M, device=z1.device)
    loss12 = F.cross_entropy(logits12.float(), labels, reduction='mean')
    logits21 = (z2 @ z1.t()) / float(temperature)
    loss21 = F.cross_entropy(logits21.float(), labels, reduction='mean')
    return 0.5 * (loss12 + loss21)


@torch.no_grad()
def evaluate_hr_ndcg10_and_score(model, valid_loader, device, amp_enabled: bool, amp_dtype):
    """
    更改为评估 HR@10、NDCG@10 以及综合分数：
        score = 0.31 * HR@10 + 0.69 * NDCG@10
    候选集与原 evaluate_acc1 保持一致：每个查询的候选集=其正样本 + 批内 item 位置的负样本池。
    仅在点击位置(next_action_type==1)计算。
    """
    model.eval()
    all_ranks = []

    for batch in valid_loader:
        # 兼容包含 SSL 视图的 batch（训练集 collate_fn 返回 12 个元素）
        if isinstance(batch, (list, tuple)) and len(batch) >= 10:
            (seq, pos, neg, token_type, next_token_type, next_action_type,
             seq_feat, pos_feat, neg_feat, seq_ts) = batch[:10]
        else:
            continue

        seq = seq.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        token_type = token_type.to(device)
        next_token_type = next_token_type.to(device)
        next_action_type = next_action_type.to(device)

        with _autocast_ctx(device.type, amp_dtype, amp_enabled):
            pos_embs, neg_embs, log_feats = model(
                seq, pos, neg, token_type, next_token_type, next_action_type,
                seq_feat, pos_feat, neg_feat, seq_ts
            )  # [B,L,D] x3

            # 只评估点击的 item 位置
            mask = (next_token_type == 1) & (next_action_type == 1)
            if mask.sum().item() == 0:
                continue

            Q = log_feats[mask]                       # [M,D]
            P = pos_embs[mask]                        # [M,D]
            N = neg_embs[next_token_type == 1]        # [N,D] 批内负样本池

            pos_logits = (Q * P).sum(dim=-1, keepdim=True)  # [M,1]
            if N.numel() == 0:
                ranks = torch.ones_like(pos_logits, dtype=torch.long).squeeze(1)
            else:
                neg_logits = Q @ N.t()                        # [M,N]
                # rank = 1 + 负样本中严格大于正样本得分的个数（降序排名）
                # 如需平分处理，可将 (>=) 分摊，这里保持严格大于以避免乐观偏置。
                greater_cnt = (neg_logits > pos_logits).sum(dim=1)
                ranks = 1 + greater_cnt  # [M]

            all_ranks.append(ranks)

    if len(all_ranks) == 0:
        return {'hr10': 0.0, 'ndcg10': 0.0, 'score': 0.0}

    ranks = torch.cat(all_ranks, dim=0).to(torch.float32)  # [T]
    hits10 = (ranks <= 10).to(torch.float32)
    hr10 = hits10.mean().item()

    # 单一正样本的 NDCG@10：在命中前10名时为 1/log2(rank+1)，否则为 0
    ndcg10 = (hits10 * (1.0 / torch.log2(ranks + 1.0))).mean().item()

    score = 0.31 * hr10 + 0.69 * ndcg10
    return {'hr10': hr10, 'ndcg10': ndcg10, 'score': score}


if __name__ == '__main__':
    _stage_log("训练进程启动")
    # 路径与日志
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    _stage_log(f"参数解析完成: device={args.device}, batch_size={args.batch_size}, epochs={args.num_epochs}")
    runtime_device = _resolve_runtime_device(args.device)
    args.device = str(runtime_device)
    _stage_log(f"运行设备解析完成: {runtime_device}")
    if args.debug_start_step >= 0:
        _stage_log(
            f"调试窗口开启: start={args.debug_start_step}, end={args.debug_end_step}, "
            f"sync_every={args.debug_sync_every}, check_indices={args.debug_check_indices}"
        )

    # CUDA/TF32 设置（在支持的 NVIDIA GPU 上进一步加速）
    if runtime_device.type == 'cuda' and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # 解析 AMP 配置（CUDA/NPU 都可生效）
    amp_available = runtime_device.type in ('cuda', 'npu')
    if args.amp == 'off' or not amp_available:
        amp_enabled = False
        amp_dtype = None
    else:
        bf16_ok = _bf16_supported(runtime_device.type)
        if args.amp == 'bf16':
            amp_enabled = True
            amp_dtype = torch.bfloat16
        elif args.amp == 'fp16':
            amp_enabled = True
            amp_dtype = torch.float16
        else:
            # auto: 优先 bf16，不支持时回退 fp16
            amp_enabled = True
            amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    _stage_log(
        f"AMP 解析结果: request={args.amp}, enabled={amp_enabled}, "
        f"dtype={str(amp_dtype) if amp_dtype is not None else 'fp32'}, device={runtime_device.type}"
    )
    if runtime_device.type == 'npu':
        _stage_log(f"NPU safe copy={'on' if args.npu_safe_copy else 'off'}")

    # 固定随机种子
    set_seed(args.seed)
    _stage_log(f"随机种子设置完成: seed={args.seed}")

    # 数据集
    _stage_log(f"开始加载数据集: {data_path}")
    t_dataset = time.time()
    dataset = MyDataset(data_path, args)
    _stage_log(f"数据集加载完成: users={dataset.usernum}, items={dataset.itemnum}, samples={len(dataset)}, cost={time.time() - t_dataset:.2f}s")

    # 按 1% 划分验证集（可复现）
    N = len(dataset)
    val_count = max(1, int(round(N * 0.01)))
    rng = np.random.RandomState(args.seed)
    all_ids = np.arange(N)
    val_indices = rng.choice(all_ids, size=val_count, replace=False)
    train_mask = np.ones(N, dtype=bool)
    train_mask[val_indices] = False
    train_indices = all_ids[train_mask]

    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())
    print(f"Data split: total={N}, train={len(train_dataset)}, val={len(val_dataset)}")
    _stage_log(f"数据集划分完成: train={len(train_dataset)}, val={len(val_dataset)}")

    # 在训练开始前，导出 item 点击分桶映射供推理使用（若已存在则复用，不重复生成）
    try:
        cache_root = os.environ.get("USER_CACHE_PATH", None)
        ckpt_root = os.environ.get("TRAIN_CKPT_PATH", None)
        candidates = []
        if cache_root:
            candidates.append(Path(cache_root) / "item_click_bucket.json")
        if ckpt_root:
            candidates.append(Path(ckpt_root) / "item_click_bucket.json")
        if not candidates:
            candidates.append(Path("./item_click_bucket.json"))

        exist_path = next((p for p in candidates if p.exists()), None)
        if exist_path is not None:
            print(f"item_click_bucket.json already exists at {exist_path}. Reusing and skip saving.")
        else:
            bucket = {}
            for iid in range(1, dataset.itemnum + 1):
                cnt = int(getattr(dataset, 'item_click_counts', np.zeros(1))[iid])
                b = int(dataset._click_count_to_bucket(cnt)) if hasattr(dataset, '_click_count_to_bucket') else 1
                bucket[iid] = b

            out_paths = []
            if cache_root:
                Path(cache_root).mkdir(parents=True, exist_ok=True)
                out_paths.append(Path(cache_root) / "item_click_bucket.json")
            if ckpt_root:
                Path(ckpt_root).mkdir(parents=True, exist_ok=True)
                out_paths.append(Path(ckpt_root) / "item_click_bucket.json")
            if not out_paths:
                out_paths.append(Path("./item_click_bucket.json"))

            for p in out_paths:
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(bucket, f)
                print(f"Saved item_click_bucket.json -> {p}")
    except Exception as e:
        print(f'warn: failed to dump item_click_bucket.json: {e}')

    # DataLoaders
    num_workers = 12
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        collate_fn=dataset.collate_fn,  # 使用原始 dataset 的 collate_fn
        pin_memory=_pin_memory_for_device(runtime_device, args.npu_safe_copy),
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(8, num_workers),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        collate_fn=dataset.collate_fn,
        pin_memory=_pin_memory_for_device(runtime_device, args.npu_safe_copy),
        prefetch_factor=2
    )
    _stage_log(f"DataLoader 构建完成: train_steps={len(train_loader)}, val_steps={len(val_loader)}, workers={num_workers}")

    # 模型
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(runtime_device)
    _stage_log("模型构建并搬运到设备完成")
    try:
        _stage_log(
            f"模型来源: class={model.__class__.__module__}.{model.__class__.__name__}, "
            f"file={inspect.getfile(model.__class__)}"
        )
    except Exception as e:
        _stage_log(f"模型来源打印失败: {e}")

    # 加载全局 logQ（供难样本修正）
    device_t = runtime_device
    def _load_item_logQ_for_train(dataset, device: torch.device):
        """
        优先从 USER_CACHE_PATH / TRAIN_CKPT_PATH / 数据目录 读取 item_logQ.npy；
        若不存在则基于 dataset.item_appear_counts（曝光+点击）即时计算并尝试保存。
        返回：torch.float32[ itemnum+1 ]（在 device 上）
        """
        candidates = []
        for k in ["USER_CACHE_PATH", "TRAIN_CKPT_PATH", "TRAIN_DATA_PATH"]:
            v = os.environ.get(k, None)
            if v:
                candidates.append(Path(v) / "item_logQ.npy")
        # 数据目录兜底
        data_dir = getattr(dataset, "data_dir", None)
        if data_dir is not None:
            candidates.append(Path(data_dir) / "item_logQ.npy")

        logq_np = None
        for p in candidates:
            try:
                if p.exists():
                    logq_np = np.load(p)
                    print(f"Loaded item_logQ.npy from {p}")
                    break
            except Exception as e:
                print(f"warn: failed loading item_logQ.npy from {p}: {e}")

        if logq_np is None:
            # 即时计算：使用“出现次数”
            counts = getattr(dataset, "item_appear_counts", None)
            itemnum = getattr(dataset, "itemnum", 0)
            if counts is None or itemnum <= 0:
                # 退化
                logq_np = np.full(itemnum + 1, -np.log(max(itemnum, 1.0)), dtype=np.float32)
                logq_np[0] = -1e9
            else:
                counts = np.asarray(counts)
                total = float(np.sum(counts[1:]))
                if not np.isfinite(total) or total <= 0:
                    q = np.zeros(itemnum + 1, dtype=np.float32)
                    if itemnum > 0:
                        q[1:] = 1.0 / float(itemnum)
                else:
                    q = counts.astype(np.float64)
                    q[0] = 0.0
                    q = (q / total).astype(np.float32)
                q = np.maximum(q, 1e-12)
                logq_np = np.log(q, dtype=np.float64).astype(np.float32)
            # 尝试写回缓存（不强制）
            try:
                cache_root = os.environ.get("USER_CACHE_PATH", None)
                if cache_root:
                    Path(cache_root).mkdir(parents=True, exist_ok=True)
                    np.save(Path(cache_root) / "item_logQ.npy", logq_np)
                    print(f"Saved item_logQ.npy -> {Path(cache_root) / 'item_logQ.npy'}")
            except Exception as e:
                print(f"warn: failed saving item_logQ.npy: {e}")

        return torch.from_numpy(logq_np).to(device=device, dtype=torch.float32)

    item_logQ_t = _load_item_logQ_for_train(dataset, device=device_t)
    _stage_log("item_logQ 加载完成")

    # 模块初始化
    model.apply(init_weights)
    with torch.no_grad():
        model.item_emb.weight.data[0, :] = 0
        model.user_emb.weight.data[0, :] = 0
        for k in model.sparse_emb:
            model.sparse_emb[k].weight.data[0, :] = 0

    # 可选恢复
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=runtime_device))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except Exception as e:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError(f'failed loading state_dicts: {e}')

    # 优化器与调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    T_total = len(train_loader) * args.num_epochs
    num_warmup_steps = int(T_total * 0.1)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=num_warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_total - num_warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps])

    # AMP GradScaler（bf16 不需要缩放；fp16 需要）
    if amp_enabled and amp_dtype == torch.float16:
        scaler = _build_grad_scaler(runtime_device.type, enabled=True)
    else:
        scaler = _build_grad_scaler(runtime_device.type, enabled=False)

    # 早停相关（基于 score 最大化）
    best_score = float('-inf')
    epochs_no_improve = 0

    global_step = 0
    skipped_nonfinite_steps = 0
    step_debug_dir = Path(os.environ.get('TRAIN_LOG_PATH') or "./logs") / "step_debug"

    if args.inference_only:
        print("Inference only mode enabled. Skip training.")
    else:
        print(f"Start training (AMP: {'on' if amp_enabled else 'off'}, dtype={str(amp_dtype) if amp_enabled else 'fp32'})")
        _stage_log(f"开始训练: amp_enabled={amp_enabled}, amp_dtype={str(amp_dtype) if amp_enabled else 'fp32'}")
        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            model.train()
            epoch_t0 = time.time()
            _stage_log(f"Epoch {epoch} 开始")
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Train epoch {epoch}",
                dynamic_ncols=True,
                leave=True,
            )

            # 训练阶段
            for step, batch in pbar:
                step_t0 = time.time()
                # 兼容包含两条负样本 SSL 视图
                if isinstance(batch, (list, tuple)) and len(batch) >= 14:
                    (seq, pos, neg, token_type, next_token_type, next_action_type,
                     seq_feat, pos_feat, neg_feat, seq_ts, neg_feat_ssl1, neg_feat_ssl2,
                     neg_ssl1, neg_ssl2) = batch
                else:
                    (seq, pos, neg, token_type, next_token_type, next_action_type,
                     seq_feat, pos_feat, neg_feat, seq_ts) = batch
                    neg_feat_ssl1, neg_feat_ssl2 = neg_feat, neg_feat
                    neg_ssl1, neg_ssl2 = neg, neg

                device_t = runtime_device
                npu_blocking = (device_t.type == 'npu' and args.npu_safe_copy)
                seq = seq.to(device_t, non_blocking=(not npu_blocking))
                pos = pos.to(device_t, non_blocking=(not npu_blocking))
                neg = neg.to(device_t, non_blocking=(not npu_blocking))
                neg_ssl1 = neg_ssl1.to(device_t, non_blocking=(not npu_blocking))
                neg_ssl2 = neg_ssl2.to(device_t, non_blocking=(not npu_blocking))
                token_type = token_type.to(device_t, non_blocking=(not npu_blocking))
                next_token_type = next_token_type.to(device_t, non_blocking=(not npu_blocking))
                next_action_type = next_action_type.to(device_t, non_blocking=(not npu_blocking))

                debug_active = _in_debug_window(global_step, args.debug_start_step, args.debug_end_step)

                if debug_active and args.debug_check_indices:
                    idx_errs = []
                    try:
                        seq_min, seq_max = int(seq.min().item()), int(seq.max().item())
                        pos_min, pos_max = int(pos.min().item()), int(pos.max().item())
                        neg_min, neg_max = int(neg.min().item()), int(neg.max().item())
                        if seq_min < 0 or seq_max > model.item_num:
                            idx_errs.append(f"seq index out of range: min={seq_min}, max={seq_max}, item_num={model.item_num}")
                        if pos_min < 0 or pos_max > model.item_num:
                            idx_errs.append(f"pos index out of range: min={pos_min}, max={pos_max}, item_num={model.item_num}")
                        if neg_min < 0 or neg_max > model.item_num:
                            idx_errs.append(f"neg index out of range: min={neg_min}, max={neg_max}, item_num={model.item_num}")
                    except Exception as e:
                        idx_errs.append(f"index check failed: {e}")

                    if idx_errs:
                        for em in idx_errs:
                            _stage_log(f"[IndexCheck] {em}")
                        _dump_step_debug(step_debug_dir, {
                            "global_step": global_step,
                            "epoch": epoch,
                            "step": step,
                            "index_errors": idx_errs,
                            "seq": _tensor_brief_stats(seq),
                            "pos": _tensor_brief_stats(pos),
                            "neg": _tensor_brief_stats(neg),
                        })

                optimizer.zero_grad(set_to_none=True)

                # AMP 前向
                try:
                    with _autocast_ctx(runtime_device.type, amp_dtype, amp_enabled):
                        pos_embs, neg_embs, log_feats = model(
                            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts
                        )
                except Exception as e:
                    _stage_log(f"[Forward] model forward failed at step={global_step}: {e}")
                    _dump_step_debug(step_debug_dir, {
                        "global_step": global_step,
                        "epoch": epoch,
                        "step": step,
                        "phase": "forward_exception",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "lr": float(optimizer.param_groups[0]['lr']),
                        "temperature": float(args.temperature),
                        "ssl_alpha": float(args.ssl_alpha),
                        "seq": _tensor_brief_stats(seq),
                        "pos": _tensor_brief_stats(pos),
                        "neg": _tensor_brief_stats(neg),
                        "seq_ts": _tensor_brief_stats(seq_ts),
                        "token_type": _tensor_brief_stats(token_type),
                        "next_token_type": _tensor_brief_stats(next_token_type),
                        "next_action_type": _tensor_brief_stats(next_action_type),
                    })
                    raise

                if not torch.isfinite(log_feats).all().item():
                    _stage_log(f"[Forward] 检测到非有限 log_feats，step={global_step}")
                    _dump_step_debug(step_debug_dir, {
                        "global_step": global_step,
                        "epoch": epoch,
                        "step": step,
                        "phase": "non_finite_log_feats",
                        "lr": float(optimizer.param_groups[0]['lr']),
                        "temperature": float(args.temperature),
                        "ssl_alpha": float(args.ssl_alpha),
                        "seq": _tensor_brief_stats(seq),
                        "pos": _tensor_brief_stats(pos),
                        "neg": _tensor_brief_stats(neg),
                        "seq_ts": _tensor_brief_stats(seq_ts),
                        "token_type": _tensor_brief_stats(token_type),
                        "next_token_type": _tensor_brief_stats(next_token_type),
                        "next_action_type": _tensor_brief_stats(next_action_type),
                        "pos_embs": _tensor_brief_stats(pos_embs),
                        "neg_embs": _tensor_brief_stats(neg_embs),
                        "log_feats": _tensor_brief_stats(log_feats),
                    })

                    # 自动二次定位：在同一 batch 上开启模型层级有限性检查，抓到首个出错阶段。
                    try:
                        _stage_log(
                            f"[ForwardDiag] precheck: has_attr={hasattr(model, 'debug_check_model_finite')}, "
                            f"flag={getattr(model, 'debug_check_model_finite', 'NA')}"
                        )
                        if hasattr(model, 'debug_check_model_finite'):
                            prev_flag = bool(model.debug_check_model_finite)
                            prev_training = bool(model.training)
                            model.debug_check_model_finite = True
                            diag_payload = {
                                "global_step": global_step,
                                "epoch": epoch,
                                "step": step,
                                "phase": "non_finite_log_feats_diag",
                                "lr": float(optimizer.param_groups[0]['lr']),
                                "temperature": float(args.temperature),
                                "ssl_alpha": float(args.ssl_alpha),
                                "diag_rerun_exception": None,
                            }
                            try:
                                with torch.no_grad():
                                    with _autocast_ctx(runtime_device.type, amp_dtype, amp_enabled):
                                        pos_embs2, neg_embs2, log_feats2 = model(
                                            seq, pos, neg, token_type, next_token_type, next_action_type,
                                            seq_feat, pos_feat, neg_feat, seq_ts
                                        )
                                diag_payload.update({
                                    "diag_rerun_pos_embs": _tensor_brief_stats(pos_embs2),
                                    "diag_rerun_neg_embs": _tensor_brief_stats(neg_embs2),
                                    "diag_rerun_log_feats": _tensor_brief_stats(log_feats2),
                                    "diag_rerun_log_feats_isfinite": bool(torch.isfinite(log_feats2).all().item()),
                                })
                                _stage_log(
                                    f"[ForwardDiag] step={global_step} 二次前向完成: "
                                    f"log_feats_isfinite={diag_payload['diag_rerun_log_feats_isfinite']}"
                                )
                            except Exception as e:
                                _stage_log(f"[ForwardDiag] step={global_step} 层级有限性定位命中: {e}")
                                diag_payload.update({
                                    "diag_rerun_exception": str(e),
                                    "traceback": traceback.format_exc(),
                                })
                            finally:
                                model.debug_check_model_finite = prev_flag
                                model.train(prev_training)
                            _dump_step_debug(step_debug_dir, diag_payload)
                    except Exception as e:
                        _stage_log(f"[ForwardDiag] 二次定位流程失败: {e}")

                    skipped_nonfinite_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # 损失统一切回 fp32，降低 bf16 下对比学习 logits 溢出/失稳风险。
                loss_main, stats = InfoNCE(
                    pos_embs.float(), neg_embs.float(), log_feats.float(), temperature=args.temperature,
                    next_token_type=next_token_type, next_action_type=next_action_type,
                    pos_ids=pos, item_logQ=item_logQ_t.float(),
                    filter_current_only=True, neg_ids=neg
                )

                # ====== SSL：对负样本特征两视图做对比 ======
                if args.ssl != 'none' and args.ssl_alpha > 0.0:
                    with _autocast_ctx(runtime_device.type, amp_dtype, amp_enabled):
                        z1 = model.feat2emb(neg_ssl1, neg_feat_ssl1, include_user=False)  # [B,L,D]
                        z2 = model.feat2emb(neg_ssl2, neg_feat_ssl2, include_user=False)  # [B,L,D]
                    mask_ssl = (token_type == 1)
                    if mask_ssl.dtype is not torch.bool:
                        mask_ssl = mask_ssl.bool()
                    loss_ssl = ssl_loss(z1.float()[mask_ssl], z2.float()[mask_ssl], temperature=args.temperature)
                else:
                    loss_ssl = torch.zeros((), device=log_feats.device)

                loss = loss_main + float(args.ssl_alpha) * loss_ssl

                if not torch.isfinite(loss).item():
                    skipped_nonfinite_steps += 1
                    loss_v = _safe_scalar(loss)
                    loss_main_v = _safe_scalar(loss_main)
                    loss_ssl_v = _safe_scalar(loss_ssl)
                    _stage_log(
                        f"Epoch {epoch} step {step + 1}/{len(train_loader)} 检测到非有限 loss，跳过更新: "
                        f"loss={loss_v}, "
                        f"main={loss_main_v}, "
                        f"ssl={loss_ssl_v}, "
                        f"skipped={skipped_nonfinite_steps}"
                    )
                    _dump_step_debug(step_debug_dir, {
                        "global_step": global_step,
                        "epoch": epoch,
                        "step": step,
                        "phase": "non_finite_loss",
                        "loss": loss_v,
                        "loss_main": loss_main_v,
                        "loss_ssl": loss_ssl_v,
                        "loss_isfinite": bool(torch.isfinite(loss).item()),
                        "loss_main_isfinite": bool(torch.isfinite(loss_main).item()),
                        "loss_ssl_isfinite": bool(torch.isfinite(loss_ssl).item()),
                        "lr": float(optimizer.param_groups[0]['lr']),
                        "temperature": float(args.temperature),
                        "ssl_alpha": float(args.ssl_alpha),
                        "seq": _tensor_brief_stats(seq),
                        "pos": _tensor_brief_stats(pos),
                        "neg": _tensor_brief_stats(neg),
                        "seq_ts": _tensor_brief_stats(seq_ts),
                        "token_type": _tensor_brief_stats(token_type),
                        "next_token_type": _tensor_brief_stats(next_token_type),
                        "next_action_type": _tensor_brief_stats(next_action_type),
                        "pos_embs": _tensor_brief_stats(pos_embs),
                        "neg_embs": _tensor_brief_stats(neg_embs),
                        "log_feats": _tensor_brief_stats(log_feats),
                        "diag_stats": {
                            "mean_pos_sim": float(stats.get('mean_pos_sim', 0.0)),
                            "mean_neg_sim": float(stats.get('mean_neg_sim', 0.0)),
                            "mean_hard_neg_sim": float(stats.get('mean_hard_neg_sim', 0.0)),
                            "mask_rate_easy": float(stats.get('mask_rate_easy', 0.0)),
                            "mask_rate_hard": float(stats.get('mask_rate_hard', 0.0)),
                        },
                        "item_pos_count": int((next_token_type == 1).sum().item()),
                        "click_count": int((next_action_type == 1).sum().item()),
                    })
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if debug_active and runtime_device.type == 'npu' and args.debug_sync_every > 0:
                    if (global_step + 1) % args.debug_sync_every == 0:
                        try:
                            torch.npu.synchronize()
                        except Exception as e:
                            _stage_log(f"[DebugSync] npu synchronize failed at step={global_step}: {e}")
                            _dump_step_debug(step_debug_dir, {
                                "global_step": global_step,
                                "epoch": epoch,
                                "step": step,
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                                "seq": _tensor_brief_stats(seq),
                                "pos": _tensor_brief_stats(pos),
                                "neg": _tensor_brief_stats(neg),
                                "token_type": _tensor_brief_stats(token_type),
                                "next_token_type": _tensor_brief_stats(next_token_type),
                                "next_action_type": _tensor_brief_stats(next_action_type),
                                "loss_main": float(loss_main.detach().float().cpu().item()),
                                "loss_ssl": float(loss_ssl.detach().float().cpu().item()),
                                "loss": float(loss.detach().float().cpu().item()),
                            })
                            raise

                # 反向与优化
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()

                # 再做一次窗口内同步，覆盖反向/优化阶段的异步报错定位。
                if debug_active and runtime_device.type == 'npu' and args.debug_sync_every > 0:
                    if (global_step + 1) % args.debug_sync_every == 0:
                        try:
                            torch.npu.synchronize()
                        except Exception as e:
                            _stage_log(f"[DebugSync-PostStep] npu synchronize failed at step={global_step}: {e}")
                            _dump_step_debug(step_debug_dir, {
                                "global_step": global_step,
                                "epoch": epoch,
                                "step": step,
                                "phase": "post_step_sync",
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                                "seq": _tensor_brief_stats(seq),
                                "pos": _tensor_brief_stats(pos),
                                "neg": _tensor_brief_stats(neg),
                                "token_type": _tensor_brief_stats(token_type),
                                "next_token_type": _tensor_brief_stats(next_token_type),
                                "next_action_type": _tensor_brief_stats(next_action_type),
                            })
                            raise

                # 日志
                try:
                    log_json = json.dumps(
                        {'global_step': global_step, 'loss_main': float(loss_main.item()),
                         'loss_ssl': float(loss_ssl.item()), 'loss_total': float(loss.item()),
                         'epoch': epoch, 'LR': optimizer.param_groups[0]['lr'], 'time': time.time()}
                    )
                except Exception as e:
                    _stage_log(f"[DebugLog] loss.item() failed at step={global_step}: {e}")
                    _dump_step_debug(step_debug_dir, {
                        "global_step": global_step,
                        "epoch": epoch,
                        "step": step,
                        "phase": "log_item",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "seq": _tensor_brief_stats(seq),
                        "pos": _tensor_brief_stats(pos),
                        "neg": _tensor_brief_stats(neg),
                        "token_type": _tensor_brief_stats(token_type),
                        "next_token_type": _tensor_brief_stats(next_token_type),
                        "next_action_type": _tensor_brief_stats(next_action_type),
                    })
                    raise
                log_file.write(log_json + '\n')
                log_file.flush()
                writer.add_scalar('Loss/main', loss_main.item(), global_step)
                writer.add_scalar('Loss/ssl', loss_ssl.item(), global_step)
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Diag/mean_pos_sim', stats['mean_pos_sim'], global_step)
                writer.add_scalar('Diag/mean_neg_sim', stats['mean_neg_sim'], global_step)
                writer.add_scalar('Diag/mean_hard_neg_sim', stats['mean_hard_neg_sim'], global_step)
                writer.add_scalar('Mask/mask_rate_easy', stats['mask_rate_easy'], global_step)
                writer.add_scalar('Mask/mask_rate_hard', stats['mask_rate_hard'], global_step)

                if step == 0:
                    _stage_log(f"Epoch {epoch} 首个 batch 完成, first_step_cost={time.time() - step_t0:.2f}s")
                if args.log_interval > 0 and ((step + 1) % args.log_interval == 0 or (step + 1) == len(train_loader)):
                    _stage_log(
                        f"Epoch {epoch} step {step + 1}/{len(train_loader)} "
                        f"loss={float(loss.item()):.6f} main={float(loss_main.item()):.6f} ssl={float(loss_ssl.item()):.6f} "
                        f"lr={optimizer.param_groups[0]['lr']:.6e} step_cost={time.time() - step_t0:.2f}s"
                    )

                global_step += 1

            # ===== 验证：HR@10、NDCG@10、score =====
            val_dict = evaluate_hr_ndcg10_and_score(
                model, val_loader, device=runtime_device,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype
            )
            _stage_log(
                f"Epoch {epoch} 验证完成: hr10={val_dict['hr10']:.6f}, "
                f"ndcg10={val_dict['ndcg10']:.6f}, score={val_dict['score']:.6f}, "
                f"epoch_cost={time.time() - epoch_t0:.2f}s"
            )
            writer.add_scalar('Val/HR@10', val_dict['hr10'], epoch)
            writer.add_scalar('Val/NDCG@10', val_dict['ndcg10'], epoch)
            writer.add_scalar('Val/score', val_dict['score'], epoch)

            # 保存 checkpoint（每个 epoch）
            save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'),
                            f"global_step{global_step}.epoch={epoch}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")

            # 早停逻辑（最大化 score）
            current_score = val_dict['score']
            if current_score > best_score + 1e-12:
                best_score = current_score
                epochs_no_improve = 0
                # 可选：额外保存最佳模型
                best_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), "best")
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_dir / "model.pt")
                with open(best_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(val_dict, f, ensure_ascii=False, indent=2)
                print(f"[Epoch {epoch}] New best score: {best_score:.6f} (HR@10={val_dict['hr10']:.6f}, NDCG@10={val_dict['ndcg10']:.6f})")
            else:
                epochs_no_improve += 1
                print(f"[Epoch {epoch}] No improvement. best={best_score:.6f}, curr={current_score:.6f}. patience={epochs_no_improve}/2")
                if epochs_no_improve >= 2:
                    print(f"Early stopping triggered after {epoch} epochs. Best score={best_score:.6f}")
                    break

    print("Done")
    writer.close()
    log_file.close()