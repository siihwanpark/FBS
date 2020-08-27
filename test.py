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

args = parser.parse_args()


train_loader, test_loader = get_loader(args.batch_size, args.num_worker)
model = CifarNet(fbs=args.fbs, sparsity_ratio=args.sparsity_ratio).cuda()

state_dict = torch.load(
    f'{args.ckpt_path}/best_{args.fbs}_{args.sparsity_ratio}.pt')
model.load_state_dict(state_dict)

with torch.no_grad():
    total_num = 0
    correct_num = 0
    MACs_total = 0

    model.eval()
    for img_batch, lb_batch in tqdm(test_loader, total=len(test_loader)):
        img_batch = img_batch.cuda()
        lb_batch = lb_batch.cuda()

        #pred_batch, MACs = model(img_batch)
        pred_batch, _, MACs = model(img_batch, True)

        _, pred_lb_batch = pred_batch.max(dim=1)
        total_num += lb_batch.shape[0]
        correct_num += pred_lb_batch.eq(lb_batch).sum().item()

        MACs_total += MACs.sum().item()
    
    test_acc = correct_num/total_num
    MACs_avg = MACs_total/total_num

print(f'Test accuracy: {test_acc}')
print(f'MACs: {MACs_avg}')

with open('fig3a/results.tsv', 'a') as f:
    f.write(f'{args.fbs}\t{args.sparsity_ratio}\t{test_acc}\t{MACs_avg}\n')
