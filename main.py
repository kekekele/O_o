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
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--seed', default=20252026, type=int)

    # Baseline Model construction
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--hidden_units', default=512, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', default=False, action='store_true')

    parser.add_argument('--num_negatives', default=1024, action='store_true')
    parser.add_argument('--loss_type', default='infonce_neg', choices=['infonce_pos', 'bce', 'infonce_neg'])
    parser.add_argument('--temperature', default=0.07, type=float)

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

def InfoNCE_pos(pos_embs, log_feats, temperature: float, next_token_type: torch.Tensor, num_negatives=128):
    """
    In-batch InfoNCE with vectorized random negative sampling (no Python loops).

    Args:
        pos_embs       : [B, L, D] 正样本 key embeddings
        log_feats      : [B, L, D] 查询 query embeddings
        temperature    : float     温度系数
        next_token_type: [B, L]    token类型 (1 表示是 item)
        num_negatives  : int       每个 query 随机采样的负样本数（不含正样本）

    Returns:
        Scalar Tensor: 平均 InfoNCE loss
    """
    # 选出需要参与训练的位置
    mask = (next_token_type == 1)
    if mask.dtype != torch.bool:
        mask = mask.bool()

    pos_embs = pos_embs[mask]   # [M, D]
    log_feats = log_feats[mask] # [M, D]
    M = pos_embs.size(0)
    device = pos_embs.device

    if M <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 全集索引矩阵: [M, M]，每行是所有 key 的编号
    all_indices = torch.arange(M, device=device)
    all_indices = all_indices.unsqueeze(0).expand(M, M)  # [M, M]

    # 去掉自身正样本索引
    mask_self = all_indices != torch.arange(M, device=device).unsqueeze(1)
    filtered_indices = all_indices[mask_self].view(M, M-1)  # [M, M-1]

    # 对过滤后的索引随机打乱并取前num_negatives
    perm = torch.argsort(torch.rand(M, M-1, device=device), dim=1)
    neg_indices = filtered_indices.gather(1, perm[:, :min(num_negatives, M-1)])  # [M, num_negatives]

    # 拼接正样本索引到最前列
    pos_indices = torch.arange(M, device=device).unsqueeze(1)  # [M, 1]
    sampled_indices = torch.cat([pos_indices, neg_indices], dim=1)  # [M, 1+num_negatives]

    # 计算相似度：[M, 1+num_negatives]
    sampled_keys = pos_embs[sampled_indices]  # [M, 1+num_negatives, D]
    logits = torch.bmm(
        log_feats.unsqueeze(1),            # [M, 1, D]
        sampled_keys.transpose(1, 2)       # [M, D, 1+num_negatives]
    ).squeeze(1) / temperature             # [M, 1+num_negatives]

    labels = torch.zeros(M, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels)
    return loss

def InfoNCE_neg(
    pos_embs: torch.Tensor,           # [B, L, D]
    neg_embs: torch.Tensor,           # [B, L, D]
    log_feats: torch.Tensor,          # [B, L, D]
    temperature: float,               # > 0
    next_token_type: torch.Tensor,    # [B, L]，1 表示 item
    num_negatives: int = 1024,
) -> torch.Tensor:
    assert temperature > 0.0, "temperature must be > 0"
    mask = (next_token_type == 1)
    if mask.dtype is not torch.bool:
        mask = mask.bool()

    Q   = log_feats[mask]   # [M, D]
    Kp  = pos_embs[mask]    # [M, D]
    Knp = neg_embs[mask]    # [M, D] 作为负样本池
    device = Q.device

    M = Q.size(0)
    if M == 0:
        return torch.zeros((), device=device)

    # 从负样本池中采样 K 个负样本（共享池，显存友好）
    pool_size = Knp.size(0)
    K = num_negatives
    if pool_size == 0:
        # 没有负样本可用，退化为仅正样本（loss 会为 0）
        pos_logits = (Q * Kp).sum(dim=-1, keepdim=True) / temperature
        labels = torch.zeros(M, dtype=torch.long, device=device)
        return F.cross_entropy(pos_logits, labels, reduction='mean')

    if K <= pool_size:
        neg_idx = torch.randperm(pool_size, device=device)[:K]            # 无放回
    else:
        # 池子不够大，允许有放回采样
        neg_idx = torch.randint(0, pool_size, (K,), device=device)        # 有放回
    Kn = Knp[neg_idx]                                                     # [K, D]

    # 计算 logits
    pos_logits = (Q * Kp).sum(dim=-1, keepdim=True) / temperature         # [M, 1]
    neg_logits = (Q @ Kn.t()) / temperature                               # [M, K]

    logits = torch.cat([pos_logits, neg_logits], dim=1)                   # [M, 1+K]
    labels = torch.zeros(M, dtype=torch.long, device=device)              # 正例在第0列
    loss = F.cross_entropy(logits, labels, reduction='mean')
    return loss

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

    # 模型
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # 模块初始化
    model.apply(init_weights)
    with torch.no_grad():
        model.pos_emb.weight.data[0, :] = 0
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    T_total = len(train_loader) * args.num_epochs
    num_warmup_steps = int(T_total * 0.05)
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
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                device = args.device
                seq = seq.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                token_type = token_type.to(device)
                next_token_type = next_token_type.to(device)

                if args.loss_type == 'infonce_pos':
                    pos_embs, neg_embs, log_feats = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                    )
                    loss = InfoNCE_pos(
                        pos_embs, log_feats, temperature=args.temperature, next_token_type=next_token_type, num_negatives=args.num_negatives
                    )
                elif args.loss_type == 'infonce_neg':
                    pos_embs, neg_embs, log_feats = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                    )
                    loss = InfoNCE_neg(
                        pos_embs, neg_embs, log_feats, temperature=args.temperature, next_token_type=next_token_type,
                        num_negatives=args.num_negatives
                    )
                    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
                    mask = (next_token_type == 1)
                    pos_logits = (log_feats * pos_embs).sum(dim=-1)
                    neg_logits = (log_feats * neg_embs).sum(dim=-1)
                    pos_labels = torch.ones_like(pos_logits, device=device)
                    neg_labels = torch.zeros_like(neg_logits, device=device)
                    loss += bce_criterion(pos_logits[mask], pos_labels[mask])
                    loss += bce_criterion(neg_logits[mask], neg_labels[mask])
                elif args.loss_type == 'bce':
                    pos_logits, neg_logits = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                    )
                    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
                    mask = (next_token_type == 1)
                    pos_labels = torch.ones_like(pos_logits, device=device)
                    neg_labels = torch.zeros_like(neg_logits, device=device)
                    loss = bce_criterion(pos_logits[mask], pos_labels[mask])
                    loss += bce_criterion(neg_logits[mask], neg_labels[mask])

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
                optimizer.step()
                scheduler.step()

                global_step += 1

            # 验证阶段（评估）
            model.eval()
            valid_loss_sum = 0.0
            with torch.no_grad():
                for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                    device = args.device
                    seq = seq.to(device)
                    pos = pos.to(device)
                    neg = neg.to(device)
                    token_type = token_type.to(device)
                    next_token_type = next_token_type.to(device)

                    if args.loss_type == 'infonce_pos':
                        pos_embs, neg_embs, log_feats = model(
                            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                        )
                        loss = InfoNCE_pos(
                            pos_embs, log_feats, temperature=args.temperature, next_token_type=next_token_type,
                            num_negatives=args.num_negatives
                        )
                    elif args.loss_type == 'infonce_neg':
                        pos_embs, neg_embs, log_feats = model(
                            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                        )
                        loss = InfoNCE_neg(
                            pos_embs, neg_embs, log_feats, temperature=args.temperature,
                            next_token_type=next_token_type,
                            num_negatives=args.num_negatives
                        )
                        bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
                        mask = (next_token_type == 1)
                        pos_logits = (log_feats * pos_embs).sum(dim=-1)
                        neg_logits = (log_feats * neg_embs).sum(dim=-1)
                        pos_labels = torch.ones_like(pos_logits, device=device)
                        neg_labels = torch.zeros_like(neg_logits, device=device)
                        loss += bce_criterion(pos_logits[mask], pos_labels[mask])
                        loss += bce_criterion(neg_logits[mask], neg_labels[mask])
                    elif args.loss_type == 'bce':
                        pos_logits, neg_logits = model(
                            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                        )
                        bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
                        mask = (next_token_type == 1)
                        pos_labels = torch.ones_like(pos_logits, device=device)
                        neg_labels = torch.zeros_like(neg_logits, device=device)
                        loss = bce_criterion(pos_logits[mask], pos_labels[mask])
                        loss += bce_criterion(neg_logits[mask], neg_labels[mask])

                    valid_loss_sum += loss.item()

            valid_loss_avg = valid_loss_sum / max(1, len(valid_loader))
            writer.add_scalar('Loss/valid', valid_loss_avg, global_step)

            # 保存 checkpoint
            save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_avg:.4f}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")

        # # ===== 额外步骤：用验证集训练一轮（fine-tune）=====
        # # 可复现的乱序 DataLoader（验证集）
        # valid_train_loader = DataLoader(
        #     valid_dataset,
        #     batch_size=args.batch_size,
        #     shuffle=True,  # 训练用 shuffle
        #     num_workers=num_workers,
        #     worker_init_fn=worker_init_fn if num_workers > 0 else None,
        #     collate_fn=dataset.collate_fn,
        #     generator=torch.Generator().manual_seed(args.seed + 1),  # 可复现
        # )
        #
        # model.train()
        # finetune_step = 0
        #
        # # 如需单独的微调学习率（例如降低10倍），可解除注释：
        # # for pg in optimizer.param_groups:
        # #     pg['lr'] = 1e-5
        #
        # print("Start fine-tuning on validation set for 1 epoch")
        # for step, batch in tqdm(enumerate(valid_train_loader), total=len(valid_train_loader)):
        #     seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
        #     device = args.device
        #     seq = seq.to(device)
        #     pos = pos.to(device)
        #     neg = neg.to(device)
        #     token_type = token_type.to(device)
        #     next_token_type = next_token_type.to(device)
        #
        #     pos_logits, neg_logits = model(
        #         seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
        #     )
        #
        #     if args.loss_type == 'listwise':
        #         loss = listwise_loss_from_logits(
        #             pos_logits, neg_logits, next_token_type, temperature=args.temperature
        #         )
        #     else:
        #         bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        #         mask = (next_token_type == 1)
        #         pos_labels = torch.ones_like(pos_logits, device=device)
        #         neg_labels = torch.zeros_like(neg_logits, device=device)
        #         loss = bce_criterion(pos_logits[mask], pos_labels[mask])
        #         loss += bce_criterion(neg_logits[mask], neg_labels[mask])
        #
        #     log_json = json.dumps(
        #         {'global_step': global_step, 'fine_tune_step': finetune_step, 'loss': float(loss.item()),
        #          'phase': 'finetune_valid', 'LR': optimizer.param_groups[0]['lr'], 'time': time.time()}
        #     )
        #     print(log_json)
        #     log_file.write(log_json + '\n')
        #     log_file.flush()
        #     writer.add_scalar('Loss/finetune_valid', loss.item(), global_step)
        #
        #     optimizer.zero_grad(set_to_none=True)
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #     optimizer.step()
        #
        #     finetune_step += 1
        #     global_step += 1

        # 保存最终模型（fine-tune 后）
        # final_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"final_after_finetune_step{global_step}")
        # final_dir.mkdir(parents=True, exist_ok=True)
        # torch.save(model.state_dict(), final_dir / "final_model.pt")
        # print(f"Fine-tuning done. Final model saved to: {final_dir}")

    print("Done")
    writer.close()
    log_file.close()