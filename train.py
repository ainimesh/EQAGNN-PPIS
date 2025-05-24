#!/usr/bin/env python3
"""
EQAGNN Model Training Script
============================
This script trains the EQAGNN model with configurable parameters.
    
Author: Animesh
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import time
import random
import csv
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Import custom modules
from utils import processing_fasta_file
from data import ProDataset
from evalution import compute_roc, compute_aupr, compute_mcc, micro_score, acc_score, compute_performance
from main.method.model import EQAGNN_Model



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed=0):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Initialize worker with fixed seed"""
    seed = 3
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return


# Dataset configurations
DATASET_CONFIGS = {
    'train': {
        'data_file': 'Train_332.fa',
        'res_type': 'train_pdb_332_CA',
        'adj_type': '335_train_CA'
    },
    'test_60': {
        'data_file': 'Test_60.fa',
        'res_type': 'test_pdb_60_CA',
        'adj_type': '60_test_CA'
    },
    'test_315': {
        'data_file': 'Test_315.fa',
        'res_type': 'test_pdb_315_CA',
        'adj_type': '315_test_CA'
    },
    'ubtest': {
        'data_file': 'Ubtest.fa',
        'res_type': 'ubtest_pdb_CA',
        'adj_type': 'ubtest_CA'
    }
}


def create_data_loader(threshold=14, batch_size=1, num_workers=2,
                    which_data='Train_332.fa', res_type='train_pdb_332_CA',
                    adj_type='335_train_CA', FP='./main/Feature/', shuffle=True):
    """Create data loader for the specified dataset"""
    
    data_path = f'./main/Dataset/{which_data}'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    pdb_ids, sequence, labels, input_files = processing_fasta_file(data_path)
    
    RPP = f'./main/Res_positions/{res_type}_res_pos.pkl'
    AP = f'./main/Input_adj/Adj_matrix_{adj_type}'
    
    if not os.path.exists(RPP):
        raise FileNotFoundError(f"Residue position file not found: {RPP}")
    if not os.path.exists(AP):
        raise FileNotFoundError(f"Adjacency matrix path not found: {AP}")
    
    dataset = ProDataset(
        pdb_ids, sequence, labels,
        threshold=threshold,
        Res_Position_Path=RPP,
        Adj_path=AP,
        Feat_path=FP,
        seq=False,
        pbert=False,
        pstruct=False,
        patom=False,
        all_feat=True
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn
    )
    
    return data_loader


def train_epoch(model, dataloader, optimizer, device, epoch, all_epochs, print_freq=100):
    """Train for one epoch"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        
        # Class weights for imbalanced dataset
        sc = torch.tensor([1.0, 4.0]).to(device)
        loss = F.cross_entropy(out, batch.y, weight=sc)
        batch_size = 1

        MiP, MiR, MiF, PNum, RNum = micro_score(
            out.argmax(dim=1).cpu().numpy(),
            batch.y.cpu().numpy()
        )
        
        losses.update(loss.item(), batch_size)
        
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, all_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(dataloader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'F1:%.4f' % (MiF),
                'Prec:%.4f' % (MiP),
                'Rec:%.4f' % (MiR)
            ])
            print(res)

    return batch_time.avg, losses.avg


def evaluate(model, dataloader, device, print_freq=10, is_test=True):
    """Evaluate model on validation/test set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    
    y_true = []
    y_pred = []

    end = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            out = model(batch)
            
            sc = torch.tensor([1.0, 4.0]).to(device)
            loss = F.cross_entropy(out, batch.y, weight=sc)
            batch_size = 1
            losses.update(loss.item(), batch_size)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(dataloader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                ])
                print(res)
                
            y_true.append(batch.y.cpu().numpy())
            y_pred.append(out.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    # Compute metrics
    auc = compute_roc(y_pred, y_true)
    aupr = compute_aupr(y_pred, y_true)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(y_pred, y_true)
    acc_val = acc_score(predictions_max, y_true)
    mcc = compute_mcc(predictions_max, y_true)

    return batch_time.avg, losses.avg, acc_val, f_max, p_max, r_max, auc, aupr, t_max, mcc


def save_checkpoint(state, filename):
    """Save training checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer=None):
    """Load training checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_metrics', {})


def train(model, args, train_dataloader, val_dataloader, device):
    """Main training function"""
    # Count parameters
    total_param = sum(np.prod(list(param.data.size())) for param in model.parameters())
    print(f'Total parameters: {total_param:,}')
    
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_factor, 
        patience=args.lr_patience, min_lr=args.min_lr
    )
    
    # Create directories
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Setup best metrics tracking
    best_val_metrics = {
        'mcc': args.min_mcc,
        'f1': args.min_f1,
        'auprc': args.min_auprc,
        'epoch': 0
    }
    
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        start_epoch, loaded_metrics = load_checkpoint(args.resume, model, optimizer)
        best_val_metrics.update(loaded_metrics)
        print(f"Resumed from epoch {start_epoch}")
    
    # Setup results CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{args.exp_name}_{timestamp}.csv"
    
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Epoch', 'Train Loss', 'Val Loss', 'Val Accuracy', 
            'Val F1', 'Val Precision', 'Val Recall', 
            'Val AUC', 'Val AUPRC', 'Val MCC', 'LR'
        ])
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Results will be saved to: {results_file}")
    print("-" * 80)
    
    # Training loop
    training_start_time = time.time()
    
    for epoch in tqdm(range(start_epoch, args.epochs), desc="Training"):
        epoch_start_time = time.time()
        
        # Train
        _, train_loss = train_epoch(
            model, train_dataloader, optimizer, device, 
            epoch, args.epochs, print_freq=args.print_freq
        )
        
        # Evaluate
        print(f"\nEvaluating on {args.val_dataset}...")
        _, val_loss, acc, f_max, p_max, r_max, auc, aupr, t_max, mcc = evaluate(
            model, val_dataloader, device, print_freq=args.print_freq, is_test=False
        )
        
        # Adjust learning rate
        scheduler.step(acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs} Summary (Time: {epoch_time:.1f}s)")
        print(f"{'='*80}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics - Acc: {acc:.4f}, F1: {f_max:.4f}, MCC: {mcc:.4f}")
        print(f"             AUC: {auc:.4f}, AUPR: {aupr:.4f}")
        print(f"             Precision: {p_max:.4f}, Recall: {r_max:.4f}")
        
        # Save to CSV
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch+1, round(train_loss, 4), round(val_loss, 4), 
                round(acc, 4), round(f_max, 4), round(p_max, 4), 
                round(r_max, 4), round(auc, 4), round(aupr, 4), 
                round(mcc, 4), current_lr
            ])
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': {
                'acc': acc, 'f1': f_max, 'mcc': mcc,
                'auc': auc, 'aupr': aupr
            },
            'best_metrics': best_val_metrics,
            'args': args
        }
        
        # Save latest checkpoint
        save_checkpoint(checkpoint, model_dir / f"{args.exp_name}_latest.pt")
        
        # Save best model if criteria met
        if (mcc > best_val_metrics['mcc'] and 
            f_max > best_val_metrics['f1'] and 
            aupr > best_val_metrics['auprc']):
            
            best_val_metrics.update({
                'mcc': mcc, 'f1': f_max, 'auprc': aupr, 'epoch': epoch + 1
            })
            
            save_checkpoint(
                checkpoint, 
                model_dir / f"{args.exp_name}_best_epoch{epoch+1}.pt"
            )
            print(f"*** New best model saved! ***")
        
        # Early stopping check
        if args.early_stopping and (epoch - best_val_metrics['epoch']) >= args.early_stopping:
            print(f"\nEarly stopping triggered after {args.early_stopping} epochs without improvement")
            break
        
        print(f"{'='*80}\n")
    
    # Training complete
    total_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/3600:.2f} hours)")
    print(f"Best model from epoch {best_val_metrics['epoch']} with:")
    print(f"  MCC: {best_val_metrics['mcc']:.4f}")
    print(f"  F1:  {best_val_metrics['f1']:.4f}")
    print(f"  AUPR: {best_val_metrics['auprc']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train EQAGNN model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train.py --epochs 50 --val_dataset test_60
    
    # Training with custom parameters
    python train.py --epochs 100 --val_dataset test_315 --lr 0.001 --batch_size 2
    
    # Resume training from checkpoint
    python train.py --resume checkpoints/model_latest.pt --epochs 20
    
    # Training with early stopping
    python train.py --epochs 100 --early_stopping 10
        """
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--threshold', type=int, default=14,
                        help='Threshold parameter for dataset')
    
    # Model parameters
    parser.add_argument('--num_layers', type=int, default=8,
                        help='Number of model layers')
    parser.add_argument('--in_dim', type=int, default=62,
                        help='Input dimension')
    parser.add_argument('--out_dim', type=int, default=2,
                        help='Output dimension')
    parser.add_argument('--s_dim', type=int, default=62,
                        help='Hidden dimension')
    parser.add_argument('--s_dim_edge', type=int, default=8,
                        help='Edge dimension')
    
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--lr_factor', type=float, default=0.9,
                        help='Learning rate reduction factor')
    parser.add_argument('--lr_patience', type=int, default=2,
                        help='Patience for learning rate reduction')
    parser.add_argument('--min_lr', type=float, default=0.00001,
                        help='Minimum learning rate')
    
    # Dataset parameters
    parser.add_argument('--val_dataset', type=str, default='test_60',
                        choices=['test_60', 'test_315', 'ubtest'],
                        help='Validation dataset to use')
    
    # Checkpoint parameters
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--model_dir', type=str, default='model_trained',
                        help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--exp_name', type=str, default='EQAGNN',
                        help='Experiment name for saving')
    
    # Early stopping
    parser.add_argument('--early_stopping', type=int, default=None,
                        help='Early stopping patience (disabled if None)')
    
    # Best model criteria
    parser.add_argument('--min_mcc', type=float, default=0.486,
                        help='Minimum MCC threshold for best model')
    parser.add_argument('--min_f1', type=float, default=0.564,
                        help='Minimum F1 threshold for best model')
    parser.add_argument('--min_auprc', type=float, default=0.562,
                        help='Minimum AUPRC threshold for best model')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='Print frequency during training')
    
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    model = EQAGNN_Model(
        num_layers=args.num_layers,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        s_dim=args.s_dim,
        s_dim_edge=args.s_dim_edge,
        equivariant_pred=False
    )
    
    # Create data loaders
    print("\nCreating data loaders...")
    
    # Training data
    train_config = DATASET_CONFIGS['train']
    train_loader = create_data_loader(
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        which_data=train_config['data_file'],
        res_type=train_config['res_type'],
        adj_type=train_config['adj_type'],
        shuffle=True
    )
    print(f"Training data loaded: {len(train_loader)} batches")
    
    # Validation data
    val_config = DATASET_CONFIGS[args.val_dataset]
    val_loader = create_data_loader(
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        which_data=val_config['data_file'],
        res_type=val_config['res_type'],
        adj_type=val_config['adj_type'],
        shuffle=False
    )
    print(f"Validation data loaded ({args.val_dataset}): {len(val_loader)} batches")
    
    # Start training
    train(model, args, train_loader, val_loader, device)


if __name__ == '__main__':
    main()