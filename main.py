import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from dataset import MyDataset
from model import BaselineModel


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--temperature', default=0.2, type=int)

    # Baseline Model construction
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--hidden_units', default=512, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', default=True, action='store_true')

    # 新增：早停耐心值
    parser.add_argument('--early_stopping_patience', default=2, type=int)

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()
    return args

# def PSL(pos_logits, neg_logits):
#     sigma = lambda x: torch.log(F.relu(x + 1))
#     d = neg_logits - pos_logits.unsqueeze(1)  # (B, S)
#     # PSL1: softmax-like
#     loss = torch.logsumexp(sigma(d) / 0.2, dim=1).mean()
#     # PSL2: BPR-like
#     # loss = torch.exp(sigma(d) / 0.2).mean()
#     # loss = torch.log(loss)
#     return loss

if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # 早停相关变量
    best_val_loss = float('inf')
    patience_counter = 0

    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break

        # 训练阶段
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            loss = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            # pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
            #     neg_logits.shape, device=args.device
            # )
            optimizer.zero_grad()
            # indices = np.where(next_token_type == 1)
            # loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            # loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            # loss = PSL(pos_logits[indices], neg_logits[indices])

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'LR': optimizer.param_groups[0]['lr'], 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        valid_loss_sum = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                loss = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                # pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                #     neg_logits.shape, device=args.device
                # )
                # indices = np.where(next_token_type == 1)
                # loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                # loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                # loss = PSL(pos_logits[indices], neg_logits[indices])
                valid_loss_sum += loss.item()

        valid_loss_avg = valid_loss_sum / len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_avg, global_step)

        # 🧠 早停判断逻辑
        if valid_loss_avg < best_val_loss:
            best_val_loss = valid_loss_avg
            patience_counter = 0
        else:
            patience_counter += 1

            # 学习率衰减
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = old_lr * 0.1  # 衰减10倍，可调
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Decayed learning rate from {old_lr:.6f} to {new_lr:.6f} at epoch {epoch}")

            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}, best validation loss: {best_val_loss:.4f}")
                break

        # 每个epoch都保存模型 (保留指定的路径格式)
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_avg:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()

