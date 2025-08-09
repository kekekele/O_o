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
    parser.add_argument('--lr', default=0.001, type=float)
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
    parser.add_argument('--norm_first', default=True, action='store_true')

    # Loss
    parser.add_argument('--loss_type', default='bce', choices=['batchsoftmax', 'bce', 'infonce'])
    parser.add_argument('--temperature', default=0.2, type=float)

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

def BatchSoftmax(pos_embs, log_feats, temperature: float, next_token_type: torch.Tensor):
    # 选出参与训练的位置
    mask = (next_token_type == 1)
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    # 直接按 mask 筛选出 [M, D]
    pos_embs = pos_embs[mask]
    log_feats = log_feats[mask]

    pos_score = (log_feats * pos_embs).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(pos_embs, log_feats.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score + 10e-6)
    return torch.mean(loss)


def InfoNCE(pos_embs, neg_embs, log_feats, temperature: float, next_token_type: torch.Tensor):
    """
    Args:
        pos_embs: (torch.Tensor - N x L × D)
        neg_embs: (torch.Tensor - N x L × D)
        log_feats: (torch.Tensor - N x L × D)
        temperature: float

    Return: Average InfoNCE Loss
    """
    # 选出参与训练的位置
    mask = (next_token_type == 1)
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    # 直接按 mask 筛选出 [M, D]
    pos_embs = pos_embs[mask]
    log_feats = log_feats[mask]
    neg_embs = neg_embs[mask]

    # 计算正样本得分
    pos_score = (log_feats * pos_embs).sum(dim=-1)  # [M]
    pos_score = torch.exp(pos_score / temperature)

    # 计算负样本得分矩阵
    # log_feats: [M, D]
    # neg_embs: [M, D]
    # ttl_score: [M] (每个正样本与所有负样本的相似度)
    ttl_score = torch.matmul(log_feats, neg_embs.transpose(-1, -2))  # [M, M]
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=-1)  # [M]

    # 计算softmax损失
    loss = -torch.log(pos_score / (ttl_score + pos_score + 1e-6))
    return torch.mean(loss)


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
    num_warmup_steps = int(T_total * 0.1)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_warmup_steps)
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

                if args.loss_type == 'infonce':
                    pos_embs, neg_embs, log_feats = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                    )
                    loss = InfoNCE(
                        pos_embs, neg_embs, log_feats, temperature=args.temperature, next_token_type=next_token_type
                    )
                elif args.loss_type == 'batchsoftmax':
                    pos_embs, neg_embs, log_feats = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                    )
                    loss = BatchSoftmax(
                        pos_embs, log_feats, temperature=args.temperature, next_token_type=next_token_type
                    )
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

                    if args.loss_type == 'infonce':
                        pos_embs, neg_embs, log_feats = model(
                            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                        )
                        loss = InfoNCE(
                            pos_embs, neg_embs, log_feats, temperature=args.temperature, next_token_type=next_token_type
                        )
                    elif args.loss_type == 'batchsoftmax':
                        pos_embs, neg_embs, log_feats = model(
                            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                        )
                        loss = BatchSoftmax(
                            pos_embs, log_feats, temperature=args.temperature, next_token_type=next_token_type
                        )
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