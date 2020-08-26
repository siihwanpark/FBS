import os

#os.system('python main.py')
os.system('python main.py --fbs 1 --sparsity_ratio 1.0 --pretrained checkpoints/best_False_1.0.pt --epochs 500')
os.system('python main.py --fbs 1 --sparsity_ratio 0.9 --pretrained checkpoints/best_True_1.0.pt --epochs 500')
os.system('python main.py --fbs 1 --sparsity_ratio 0.8 --pretrained checkpoints/best_True_0.9.pt --epochs 500')
os.system('python main.py --fbs 1 --sparsity_ratio 0.7 --pretrained checkpoints/best_True_0.8.pt --epochs 500')
os.system('python main.py --fbs 1 --sparsity_ratio 0.6 --pretrained checkpoints/best_True_0.7.pt --epochs 500')
os.system('python main.py --fbs 1 --sparsity_ratio 0.5 --pretrained checkpoints/best_True_0.6.pt --epochs 500')
os.system('python main.py --fbs 1 --sparsity_ratio 0.4 --pretrained checkpoints/best_True_0.5.pt --epochs 500')
os.system('python main.py --fbs 1 --sparsity_ratio 0.3 --pretrained checkpoints/best_True_0.4.pt --epochs 500')
