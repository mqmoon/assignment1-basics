import os
import time
import argparse
import torch
import numpy as np
import wandb
import pprint

from cs336_basics.TrainingLoop.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.TrainingLoop.dataloader import get_batch
from cs336_basics.TransformerLMA.pre_norm_transformer_block import TransformerLM
from cs336_basics.Training.optimizer import AdamW
from cs336_basics.Training.loss import run_cross_entropy_loss

def get_args():
    parser = argparse.ArgumentParser(description='A comprehensive training script for a Transformer LM.')
    # Data and Paths
    parser.add_argument('--train_data_path', type=str, default='data/TinyStoriesV2-GPT4-train.npy', help='Path to memory-mapped training data (.npy)')
    parser.add_argument('--val_data_path', type=str, default='data/TinyStoriesV2-GPT4-valid.npy', help='Path to memory-mapped validation data (.npy)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--checkpoint_name', type=str, default='ckpt.pth', help='Name of the checkpoint file.')

    # Model Hyperparameters
    parser.add_argument('--context_length', type=int, default=256, help='Input sequence length.')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size (GPT-2 has 50257, but pad to multiple of 64 for performance).')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers.')
    parser.add_argument('--n_head', type=int, default=16, help='Number of attention heads.')
    parser.add_argument('--d_model', type=int, default=512, help='LM\'s hidden size.')
    parser.add_argument('--d_ff', type=int, default=1344, help='FFN hidden size.')
    parser.add_argument('--rope_theta', type=int, default=10000, help='RoPE theta value.')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--max_iters', type=int, default=5000, help='Total number of training iterations.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Maximum learning rate.')
    
    # Logging and Saving
    parser.add_argument('--eval_interval', type=int, default=250, help='How often to evaluate on validation set.')
    parser.add_argument('--save_interval', type=int, default=500, help='How often to save a checkpoint.')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of batches to average for evaluation loss.')

    # System
    parser.add_argument('--device', type=str, default='auto', help="Device to use ('cpu', 'cuda', 'mps', or 'auto').")
    parser.add_argument('--seed', type=int, default=1337, help="Random seed for reproducibility.")

    return parser.parse_args()

@torch.no_grad()
def estimate_loss(model, train_data, val_data, args):
    out = {}
    model.eval()
    for split_name, split_data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(x=split_data, batch_size=args.batch_size, context_length=args.context_length,
                             device=args.device)
            logits = model(X)
            loss = run_cross_entropy_loss(logits=logits, targets=Y)
            losses[k] = loss.item()
        out[split_name] = losses.mean()
    model.train()
    return out

def main():
    args = get_args()
    print("=" * 50)
    print(" " * 15, "Training Configuration")
    print("=" * 50)
    # 使用 pprint 格式化打印
    pprint.pprint(vars(args))
    print("=" * 50 + "\n")
    wandb.init(config=args, mode='offline')

    # --- Setup ---
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"INFO: Using device: {args.device}")
    
    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)

    # --- Data Loading ---
    print("INFO: Loading data using memory-mapping...")
    try:
        train_data = np.load(args.train_data_path, mmap_mode='r')
        val_data = np.load(args.val_data_path, mmap_mode='r')
        print(f"INFO: Training data loaded with {len(train_data):,} tokens.")
        print(f"INFO: Validation data loaded with {len(val_data):,} tokens.")
    except FileNotFoundError:
        print("ERROR: Data files not found. Please run the data preparation script first.")
        return

    # --- Model and Optimizer ---
    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.n_head,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.n_layers,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(args.device)
    
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

    # --- Resume from Checkpoint (if exists) ---
    start_iter = 0
    if os.path.exists(checkpoint_path):
        start_iter = load_checkpoint(checkpoint_path, model, optimizer)

    # --- Main Training Loop ---
    print("INFO: Starting training loop...")
    X, Y = get_batch(train_data, args.batch_size, args.context_length, args.device) # pre-fetch first batch
    t0 = time.time()
    for iter_num in range(start_iter + 1, args.max_iters + 1):

        # --- Evaluate and Log periodically ---
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, args)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Here you would also log to W&B if using it:
            wandb.log({'train_loss': losses['train'], 'val_loss': losses['val']})

        # --- Save Checkpoint periodically ---
        if iter_num % args.save_interval == 0:
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)

        # --- Forward, Backward, Update ---
        logits = model(X)
        loss = run_cross_entropy_loss(logits=logits, targets=Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # --- Timing and Batch Fetching ---
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
        
        X, Y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        # --- Termination ---
        if iter_num >= args.max_iters:
            print("INFO: Maximum iterations reached. Training finished.")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path) # final save
            break
    
if __name__ == '__main__':
    main()