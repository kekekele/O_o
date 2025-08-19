import argparse
import json
import os
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
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--seed', default=20252026, type=int)

    # Baseline Model construction
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--hidden_units', default=256, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', default=False, action='store_true')

    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)

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
    click_scale: float = 2.0,  # 点击样本损失权重
    exp_scale: float = 0.5,  # 曝光样本损失权重
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
    # 若需要归一化向量获得真正余弦，请确保上游已做 normalize 或在此加 F.normalize
    pos_sim    = (Qn * Kpn).sum(dim=-1, keepdim=True)           # [M,1] ∈[-1,1]
    pos_logits = pos_sim / temperature                          # [M,1]

    # 负样本相似度与“分母+1”的变换（保持你的原设计）
    neg_sim    = Qn @ Knegn.t()                                 # [M,M] ∈[-1,1]
    neg_logits = neg_sim / temperature   # [M,M]

    # 拼接 logits
    logits = torch.cat([pos_logits, neg_logits], dim=1)         # [M, 1+M]
    labels = torch.zeros(M, dtype=torch.long, device=device)

    # 样本级别权重：点击=click_scale，曝光=exp_scale
    act_f = act.to(Qn.dtype)
    sample_weight = torch.where(
        act_f > 0.5,
        torch.full_like(act_f, click_scale),
        torch.full_like(act_f, exp_scale)
    )  # [M]

    # 逐样本 CE，再按权重归一化
    per_example_loss = F.cross_entropy(logits, labels, reduction='none')  # [M]
    loss = (per_example_loss * sample_weight).mean()

    stats = {
        'mean_pos_sim': float(pos_sim.mean().item()),
        'mean_neg_sim': float(neg_sim.mean().item()),
    }

    return loss, stats

@torch.no_grad()
def evaluate_valid_score(model, valid_loader, device, temperature=0.05):
    """
    在完整验证集上计算 score = 0.31*HR@10 + 0.69*NDCG@10
    使用近似候选集：对每个查询，候选集=该样本的正样本 + 本批所有负样本池（与 InfoNCE 一致）。
    """
    model.eval()
    total_hits = 0.0
    total_dcg = 0.0
    total_queries = 0

    for batch in valid_loader:
        # 解包
        (seq, pos, neg, token_type, next_token_type, next_action_type,
         seq_feat, pos_feat, neg_feat, seq_ts) = batch

        seq = seq.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        token_type = token_type.to(device)
        next_token_type = next_token_type.to(device)

        # 前向
        pos_embs, neg_embs, log_feats = model(
            seq, pos, neg, token_type, next_token_type, next_action_type,
            seq_feat, pos_feat, neg_feat, seq_ts
        )  # [B,L,D] x3

        # 只评 item 位置
        mask = (next_token_type == 1)
        if mask.sum().item() == 0:
            continue

        Q = log_feats[mask]   # [M,D]
        P = pos_embs[mask]    # [M,D]
        N = neg_embs[mask]    # [M,D] 批内负样本池

        # 构造 logits（与 InfoNCE 一致）
        pos_logits = (Q * P).sum(dim=-1, keepdim=True)  # [M,1]
        neg_logits = Q @ N.t()                          # [M,M]
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [M, 1+M]

        # Top-k 命中与 NDCG
        k = min(10, logits.size(1))
        topk = torch.topk(logits, k=k, dim=1)
        hits_bool = (topk.indices == 0)                 # [M,k]
        hits_any = hits_bool.any(dim=1)                 # [M]
        total_hits += hits_any.float().sum().item()

        first_pos = torch.argmax(hits_bool.int(), dim=1)               # [M]
        dcg = hits_any.float() * (1.0 / torch.log2(first_pos.float() + 2.0))
        total_dcg += dcg.sum().item()

        total_queries += logits.size(0)

    if total_queries == 0:
        return 0.0

    hr10 = total_hits / total_queries
    ndcg10 = total_dcg / total_queries
    score = 0.31 * hr10 + 0.69 * ndcg10
    return score

if __name__ == '__main__':
    # 路径与日志
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()

    # 固定随机种子
    set_seed(args.seed)

    # 数据集与可复现划分
    dataset = MyDataset(data_path, args)
    n_total = len(dataset)
    n_valid = max(1, int(n_total * 0.1))
    n_train = n_total - n_valid
    split_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [n_train, n_valid], generator=split_gen)

    # DataLoader
    num_workers = 9
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=6
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=6
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
    scheduler = CosineAnnealingLR(optimizer, T_max=T_total, eta_min=1e-6)

    best_val_loss = float('inf')
    global_step = 0

    if args.inference_only:
        print("Inference only mode enabled. Skip training.")
    else:
        print("Start training")
        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            model.train()

            # 训练阶段（训练集）
            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts = batch
                device = args.device
                seq = seq.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                token_type = token_type.to(device)
                next_token_type = next_token_type.to(device)
                next_action_type = next_action_type.to(device)

                pos_embs, neg_embs, log_feats = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts
                )
                loss, stats = InfoNCE(
                    pos_embs, neg_embs, log_feats, temperature=args.temperature, next_token_type=next_token_type,
                    next_action_type=next_action_type
                )

                log_json = json.dumps(
                    {'global_step': global_step, 'loss': float(loss.item()), 'epoch': epoch,
                     'LR': optimizer.param_groups[0]['lr'], 'time': time.time()}
                )
                print(log_json)
                log_file.write(log_json + '\n')
                log_file.flush()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Diag/mean_pos_sim', stats['mean_pos_sim'], global_step)
                writer.add_scalar('Diag/mean_neg_sim', stats['mean_neg_sim'], global_step)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                global_step += 1

            # 验证阶段（评估）
            model.eval()
            valid_loss_sum = 0.0
            with torch.no_grad():
                for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts = batch
                    device = args.device
                    seq = seq.to(device)
                    pos = pos.to(device)
                    neg = neg.to(device)
                    token_type = token_type.to(device)
                    next_token_type = next_token_type.to(device)
                    next_action_type = next_action_type.to(device)

                    pos_embs, neg_embs, log_feats = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts
                    )
                    loss, _ = InfoNCE(
                        pos_embs, neg_embs, log_feats, temperature=args.temperature, next_token_type=next_token_type,
                        next_action_type=next_action_type
                    )

                    valid_loss_sum += loss.item()

            valid_loss_avg = valid_loss_sum / max(1, len(valid_loader))
            writer.add_scalar('Loss/valid', valid_loss_avg, global_step)

            # 只在每个 epoch 结束后，在完整验证集上计算 score
            score = evaluate_valid_score(model, valid_loader, device=args.device, temperature=args.temperature)
            writer.add_scalar('Score/valid', score, global_step)
            log_json = json.dumps(
                {'global_step': global_step, 'epoch': epoch, 'valid_score': float(score), 'time': time.time()}
            )
            print(log_json)
            log_file.write(log_json + '\n')
            log_file.flush()

            # 保存 checkpoint
            save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_avg:.4f}.score={score:.4f}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")


    print("Done")
    writer.close()
    log_file.close()