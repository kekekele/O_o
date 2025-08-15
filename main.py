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


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--seed', default=20252026, type=int)

    # Baseline Model construction
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--hidden_units', default=512, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
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
) -> torch.Tensor:
    """
    Batch-all negatives InfoNCE:
    对每个 query 使用本批次 mask 后的全部负样本作为负例（不降采样）。

    返回：
        平均交叉熵损失（标注：正样本为第 0 列）
    """
    assert temperature > 0.0, "temperature must be > 0"

    mask = (next_token_type == 1)
    if mask.dtype is not torch.bool:
        mask = mask.bool()

    # 展平后按 mask 取出有效位置
    Q   = log_feats[mask]   # [M, D]
    Kp  = pos_embs[mask]    # [M, D] 每行对应自己的正样本 key
    Kneg = neg_embs[mask]   # [M, D] 本批全部负样本池（供所有行共享）
    device = Q.device

    M = Q.size(0)
    if M == 0:
        return torch.zeros((), device=device)

    # 正样本打分：[M, 1]
    pos_logits = (Q * Kp).sum(dim=-1, keepdim=True) / temperature

    # 全批负样本打分：[M, M]
    # 每个 query 与整批负样本池逐一打分
    neg_logits = (Q @ Kneg.t()) / temperature

    # 拼接：[M, 1 + M]，labels 全为 0（正样本在第 0 列）
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    labels = torch.zeros(M, dtype=torch.long, device=device)

    loss = F.cross_entropy(logits, labels, reduction='mean')
    return loss

@torch.no_grad()
def evaluate_valid_subset_score(model, valid_subset_loader, device, temperature=0.05):
    """
    在验证子集上计算 score = 0.31*HR@10 + 0.69*NDCG@10
    使用近似候选集：对每个查询，候选集=该样本的正样本 + 本批所有负样本池（与 InfoNCE 一致）。
    """
    model.eval()
    total_hits = 0.0
    total_dcg = 0.0
    total_queries = 0

    for batch in valid_subset_loader:
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

        # 构造 logits（与 InfoNCE 一致的打分方式，不加温度缩放也可以）
        pos_logits = (Q * P).sum(dim=-1, keepdim=True)  # [M,1]
        neg_logits = Q @ N.t()                          # [M,M]
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [M, 1+M]

        # Top-k 命中与 NDCG
        k = min(10, logits.size(1))
        topk = torch.topk(logits, k=k, dim=1)
        hits_bool = (topk.indices == 0)                 # [M,k] 是否包含正样本
        hits_any = hits_bool.any(dim=1)                 # [M]
        total_hits += hits_any.float().sum().item()

        # 正样本在 top-k 的位置 -> NDCG
        # 若未命中 top-k，贡献为 0；若命中，位置 p 的 DCG=1/log2(p+2)
        # 使用 argmax 找到第一个 True 的位置（未命中时值无效，但会被 hits_any 掩蔽）
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
    num_workers = 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        collate_fn=dataset.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        collate_fn=dataset.collate_fn,
    )
    # 验证子集：随机采样 5% 用户
    n_valid_total = len(valid_dataset)
    n_valid_sub = max(1, int(n_valid_total * 0.05))
    g_sub = torch.Generator().manual_seed(args.seed)  # 固定种子，保证可复现
    perm = torch.randperm(n_valid_total, generator=g_sub)
    valid_subset_idx = perm[:n_valid_sub].tolist()
    valid_subset = torch.utils.data.Subset(valid_dataset, valid_subset_idx)
    valid_subset_loader = DataLoader(
        valid_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        collate_fn=dataset.collate_fn,
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

                pos_embs, neg_embs, log_feats = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts
                )
                loss = InfoNCE(
                    pos_embs, neg_embs, log_feats, temperature=args.temperature, next_token_type=next_token_type,
                )

                log_json = json.dumps(
                    {'global_step': global_step, 'loss': float(loss.item()), 'epoch': epoch,
                     'LR': optimizer.param_groups[0]['lr'], 'time': time.time()}
                )
                print(log_json)
                log_file.write(log_json + '\n')
                log_file.flush()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scheduler.step()
                optimizer.step()

                global_step += 1

                # 每 5000 步在 5% 验证子集上评估 score
                if global_step % 5000 == 0:
                    prev_mode = model.training
                    model.eval()
                    with torch.no_grad():
                        score = evaluate_valid_subset_score(
                            model, valid_subset_loader, device=args.device, temperature=args.temperature
                        )
                    # 恢复训练/评估模式
                    if prev_mode:
                        model.train()
                    writer.add_scalar('Score/valid_subset', score, global_step)
                    log_json = json.dumps(
                        {'global_step': global_step, 'valid_subset_score': float(score), 'time': time.time()}
                    )
                    print(log_json)
                    log_file.write(log_json + '\n')
                    log_file.flush()

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

                    pos_embs, neg_embs, log_feats = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_ts
                    )
                    loss = InfoNCE(
                        pos_embs, neg_embs, log_feats, temperature=args.temperature, next_token_type=next_token_type,
                    )

                    valid_loss_sum += loss.item()

            valid_loss_avg = valid_loss_sum / max(1, len(valid_loader))
            writer.add_scalar('Loss/valid', valid_loss_avg, global_step)

            # 保存 checkpoint
            save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_avg:.4f}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")


    print("Done")
    writer.close()
    log_file.close()