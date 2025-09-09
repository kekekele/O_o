import argparse
import json
import os
import math
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel

# os.environ.setdefault("TRAIN_LOG_PATH", "./logs")
# os.environ.setdefault("TRAIN_TF_EVENTS_PATH", "./logs/tf_events")
# os.environ.setdefault("TRAIN_DATA_PATH", "./data/TencentGR_1k")
# os.environ.setdefault("TRAIN_CKPT_PATH", "./result")

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

    args = parser.parse_args()
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
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
    # 路径与日志
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()

    # CUDA/TF32 设置（在支持的 NVIDIA GPU 上进一步加速）
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # 解析 AMP 配置
    use_cuda = torch.cuda.is_available() and str(args.device).startswith('cuda')
    if args.amp == 'off' or not use_cuda:
        amp_enabled = False
        amp_dtype = None
    else:
        # auto: 优先 bfloat16，不支持则回退到 float16
        bf16_ok = torch.cuda.is_bf16_supported()
        if args.amp == 'bf16' or (args.amp == 'auto' and bf16_ok):
            amp_enabled = True
            amp_dtype = torch.bfloat16
        else:
            amp_enabled = True
            amp_dtype = torch.float16

    # 固定随机种子
    set_seed(args.seed)

    # 数据集
    dataset = MyDataset(data_path, args)

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
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(8, num_workers),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2
    )

    # 模型
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # 加载全局 logQ（供难样本修正）
    device_t = torch.device(args.device)
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
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
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
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    # 早停相关（基于 score 最大化）
    best_score = float('-inf')
    epochs_no_improve = 0

    global_step = 0

    if args.inference_only:
        print("Inference only mode enabled. Skip training.")
    else:
        print(f"Start training (AMP: {'on' if amp_enabled else 'off'}, dtype={str(amp_dtype) if amp_enabled else 'fp32'})")
        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            model.train()
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Train epoch {epoch}",
                dynamic_ncols=True,
                leave=False,
                bar_format='{l_bar}{bar}{r_bar}\n',
            )

            # 训练阶段
            for step, batch in pbar:
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

                device_t = args.device
                seq = seq.to(device_t, non_blocking=True)
                pos = pos.to(device_t, non_blocking=True)
                neg = neg.to(device_t, non_blocking=True)
                neg_ssl1 = neg_ssl1.to(device_t, non_blocking=True)
                neg_ssl2 = neg_ssl2.to(device_t, non_blocking=True)
                token_type = token_type.to(device_t, non_blocking=True)
                next_token_type = next_token_type.to(device_t, non_blocking=True)
                next_action_type = next_action_type.to(device_t, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # AMP 前向与损失
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    pos_embs, neg_embs, log_feats = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts
                    )
                    loss_main, stats = InfoNCE(
                        pos_embs, neg_embs, log_feats, temperature=args.temperature,
                        next_token_type=next_token_type, next_action_type=next_action_type,
                        pos_ids=pos, item_logQ=item_logQ_t,
                        filter_current_only=True, neg_ids=neg
                    )

                    # ====== SSL：对负样本特征两视图做对比 ======
                    if args.ssl != 'none' and args.ssl_alpha > 0.0:
                        z1 = model.feat2emb(neg_ssl1, neg_feat_ssl1, include_user=False)  # [B,L,D]
                        z2 = model.feat2emb(neg_ssl2, neg_feat_ssl2, include_user=False)  # [B,L,D]
                        mask_ssl = (token_type == 1)
                        if mask_ssl.dtype is not torch.bool:
                            mask_ssl = mask_ssl.bool()
                        loss_ssl = ssl_loss(z1[mask_ssl], z2[mask_ssl], temperature=args.temperature)
                    else:
                        loss_ssl = torch.zeros((), device=log_feats.device)

                    loss = loss_main + float(args.ssl_alpha) * loss_ssl

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

                # 日志
                log_json = json.dumps(
                    {'global_step': global_step, 'loss_main': float(loss_main.item()),
                     'loss_ssl': float(loss_ssl.item()), 'loss_total': float(loss.item()),
                     'epoch': epoch, 'LR': optimizer.param_groups[0]['lr'], 'time': time.time()}
                )
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

                global_step += 1

            # ===== 验证：HR@10、NDCG@10、score =====
            val_dict = evaluate_hr_ndcg10_and_score(
                model, val_loader, device=torch.device(args.device),
                amp_enabled=amp_enabled, amp_dtype=amp_dtype
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