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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel


# 可根据需要修改，或在外部设置
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
    parser.add_argument('--hidden_units', default=256, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)

    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--neg_pop_alpha', default=0.15, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--feature_crosses', nargs='*', default=["118+120", "116+118"],
                        help='例如: ["118+120"]；缺省(None)表示不使用交叉特征')

    # AMP: 混合精度训练
    parser.add_argument(
        '--amp',
        default='auto',
        choices=['off', 'fp16', 'bf16', 'auto'],
        help='混合精度模式：off 关闭；fp16 半精度；bf16 bfloat16；auto 优先 bf16，不支持则回退 fp16'
    )

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

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
    click_scale: float = 1.0,  # 点击样本损失权重
    exp_scale: float = 0.1,  # 曝光样本损失权重
):
    """
    Batch-all negatives InfoNCE（余弦相似度，负样本分母+1，负样本朝 -1 优化）
    - 点击/曝光通过“损失外权重”实现，不再改动正样本 logit 的几何关系
    返回：loss, stats
    """
    assert temperature > 0.0, "temperature must be > 0"

    # 仅对 item 位置计算
    mask = (next_token_type == 1)
    if mask.dtype is not torch.bool:
        mask = mask.bool()

    Qn    = log_feats[mask]          # [M, D]
    Kpn   = pos_embs[mask]           # [M, D]
    Knegn = neg_embs[mask]           # [M, D] 共享负样本池
    act   = next_action_type[mask]   # [M]

    device = Qn.device
    M = Qn.size(0)
    if M == 0:
        return torch.zeros((), device=device), {
            'mean_pos_sim': 0.0, 'mean_neg_sim': 0.0, 'click_ratio': 0.0
        }

    # 正样本相似度与 logit（不做动作缩放）
    pos_sim    = (Qn * Kpn).sum(dim=-1, keepdim=True)           # [M,1]
    pos_logits = pos_sim / temperature                          # [M,1]

    # 负样本相似度与“分母+1”的变换（保持你的原设计）
    neg_sim    = Qn @ Knegn.t()                                 # [M,M]
    neg_logits = neg_sim / temperature                          # [M,M]

    logits = torch.cat([pos_logits, neg_logits], dim=1)         # [M, 1+M]
    labels = torch.zeros(M, dtype=torch.long, device=device)

    # 样本级别权重：点击=click_scale，曝光=exp_scale
    act_f = act.to(Qn.dtype)
    sample_weight = torch.where(
        act_f > 0.5,
        torch.full_like(act_f, click_scale),
        torch.full_like(act_f, exp_scale)
    )  # [M]
    sample_weight = sample_weight / sample_weight.sum()

    per_example_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')  # [M]
    loss = (per_example_loss * sample_weight).sum()

    stats = {
        'mean_pos_sim': float(pos_sim.mean().item()),
        'mean_neg_sim': float(neg_sim.mean().item()),
    }

    return loss, stats


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
            # PyTorch 2.x：控制 FP32 matmul 的 TF32 精度
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

    # 数据集（不再划分验证集，直接全量用于训练）
    dataset = MyDataset(data_path, args)
    train_dataset = dataset  # 全量训练

    # 在训练开始前，导出 item 点击分桶映射供推理使用（若已存在则复用，不重复生成）
    try:
        # 构造候选保存路径（优先 USER_CACHE_PATH -> TRAIN_CKPT_PATH -> 当前目录）
        cache_root = os.environ.get("USER_CACHE_PATH", None)
        ckpt_root = os.environ.get("TRAIN_CKPT_PATH", None)
        candidates = []
        if cache_root:
            candidates.append(Path(cache_root) / "item_click_bucket.json")
        if ckpt_root:
            candidates.append(Path(ckpt_root) / "item_click_bucket.json")
        if not candidates:
            candidates.append(Path("./item_click_bucket.json"))

        # 若任一位置已存在，则跳过生成与保存
        exist_path = next((p for p in candidates if p.exists()), None)
        if exist_path is not None:
            print(f"item_click_bucket.json already exists at {exist_path}. Reusing and skip saving.")
        else:
            # 生成分桶映射
            bucket = {}
            for iid in range(1, dataset.itemnum + 1):
                cnt = int(getattr(dataset, 'item_click_counts', np.zeros(1))[iid])
                b = int(dataset._click_count_to_bucket(cnt)) if hasattr(dataset, '_click_count_to_bucket') else 1
                bucket[iid] = b

            # 保存到 USER_CACHE_PATH（优先），回退到 TRAIN_CKPT_PATH / 当前目录
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

    # DataLoader（仅训练集）
    num_workers = 12
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4
    )

    # 模型
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

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

    best_val_loss = float('inf')
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

            # 训练阶段（全量数据）
            for step, batch in pbar:
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts = batch
                device = args.device
                seq = seq.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                neg = neg.to(device, non_blocking=True)
                token_type = token_type.to(device, non_blocking=True)
                next_token_type = next_token_type.to(device, non_blocking=True)
                next_action_type = next_action_type.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # AMP 前向与损失
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    pos_embs, neg_embs, log_feats = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts
                    )
                    loss, stats = InfoNCE(
                        pos_embs, neg_embs, log_feats, temperature=args.temperature, next_token_type=next_token_type,
                        next_action_type=next_action_type
                    )

                # 反向与优化（fp16 用 scaler，bf16/FP32 直接）
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    # 需先 unscale 再做梯度裁剪
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
                    {'global_step': global_step, 'loss': float(loss.item()), 'epoch': epoch,
                     'LR': optimizer.param_groups[0]['lr'], 'time': time.time()}
                )
                log_file.write(log_json + '\n')
                log_file.flush()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Diag/mean_pos_sim', stats['mean_pos_sim'], global_step)
                writer.add_scalar('Diag/mean_neg_sim', stats['mean_neg_sim'], global_step)

                global_step += 1

            # 每个 epoch 结束后保存 checkpoint（不含验证指标）
            save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'),
                            f"global_step{global_step}.epoch={epoch}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()