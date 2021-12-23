import argparse
from tqdm import tqdm

import torch

from dataset import get_loader
from model import CifarNet
import utils


parser = argparse.ArgumentParser()

parser.add_argument(
    '--fbs',
    type=utils.str2bool,
    default=False
)
parser.add_argument(
    '--sparsity_ratio',
    type=float,
    default=1.0
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=256
)

parser.add_argument(
    '--num_worker',
    type=int,
    default=4
)
parser.add_argument(
    '--ckpt_path',
    type=str,
    default='checkpoints'
)

parser.add_argument(
    '--target_cls',
    type=int,
    default=0
)

args = parser.parse_args()


train_loader, test_loader = get_loader(args.batch_size, args.num_worker)
model = CifarNet(fbs=args.fbs, sparsity_ratio=args.sparsity_ratio).cuda()

state_dict = torch.load(
    f'{args.ckpt_path}/best_{args.fbs}_{args.sparsity_ratio}.pt')
model.load_state_dict(state_dict)

with torch.no_grad():
    total_num = 0
    correct_num = 0
    cls_active_prob_list = [0]*8

    model.eval()
    for img_batch, lb_batch in tqdm(test_loader, total=len(test_loader)):
        img_batch = img_batch.cuda()
        lb_batch = lb_batch.cuda()

        cls_mask = lb_batch == args.target_cls

        if cls_mask.sum().item() == 0:
            continue

        img_batch = img_batch[cls_mask]
        lb_batch = lb_batch[cls_mask]

        pred_batch, lasso, active_channels_list = model(img_batch, True)

        _, pred_lb_batch = pred_batch.max(dim=1)
        total_num += lb_batch.shape[0]
        correct_num += pred_lb_batch.eq(lb_batch).sum().item()

        for i in range(8):
            cls_active_prob_list[i] += active_channels_list[i].sum(dim=0)
    
    test_acc = 100.*correct_num/total_num

    for i in range(8):
        cls_active_prob_list[i] /= total_num

print(f'Test accuracy: {test_acc}%')

for i in range(8):
    with open(f'fig3b/conv{i}.tsv', 'a') as f:
        f.write('\t'.join([str(prob) for prob in cls_active_prob_list[i].tolist()]))
        f.write('\n')