import torch
import torch.multiprocessing as mp
import os
import signal
import sys
import numpy as np
import random

from trainer import main_worker


def signal_handler(signal, frame):
    sys.exit(0)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train optimized 4-modal model with consistent gradient-based modality dropping (Single GPU)')
    parser.add_argument('--embedding_dir', type=str,
                        default='pretrain_data/embeddings',
                        help='Directory containing precomputed embeddings')
    parser.add_argument('--batch_size', type=int, default=1280,
                        help='Batch size per GPU (can be larger with precomputed embeddings)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs')
    parser.add_argument('--dataset_type', type=str, default='memory',
                        choices=['memory', 'lazy', 'batched', 'original'],
                        help='Dataset loading strategy: '
                             'memory (load all to RAM), '
                             'lazy (load on-demand), '
                             'batched (cache batches), '
                             'original (compatibility)')
    parser.add_argument('--use_multiprocessing', action='store_true',
                        help='Use multiprocessing DataLoader (only safe with memory dataset)')
    parser.add_argument('--drop_prob', type=float, default=0.8,
                        help='Probability of dropping a modality during training')
    parser.add_argument('--model_save_path', type=str, default='OPTIMIZED_EMBEDDINGS_gradient_adaptive_drop_modality',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save_every_n_epochs', type=int, default=2,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--lambda_1', type=float, default=1.0,
                        help='Weight for contrastive loss')
    parser.add_argument('--lambda_2', type=float, default=1.0,
                        help='Weight for ITM loss')
    parser.add_argument('--lambda_3', type=float, default=1.0,
                        help='Weight for IC50 classification loss')
    parser.add_argument('--gradient_std_multiplier', type=float, default=1.5,
                        help='Standard deviation multiplier for gradient-based dropping')
    parser.add_argument('--gradient_history_length', type=int, default=5,
                        help='Number of past gradient norms to keep in history')

    args = parser.parse_args()

    main_worker(
        embedding_dir=args.embedding_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        dataset_type=args.dataset_type,
        use_multiprocessing=args.use_multiprocessing,
        drop_prob=args.drop_prob,
        model_save_path=args.model_save_path,
        save_every_n_epochs=args.save_every_n_epochs,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        gradient_std_multiplier=args.gradient_std_multiplier,
        gradient_history_length=args.gradient_history_length
    )

if __name__ == '__main__':
    main()
