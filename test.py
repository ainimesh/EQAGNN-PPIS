#!/usr/bin/env python3
"""
EQAGNN Model Testing Script
===========================
This script evaluates the EQAGNN model on different test datasets.

Usage:
    python test.py --dataset test_60 --model_path path/to/model.pt
    python test.py --dataset test_315 --model_path path/to/model.pt
    python test.py --dataset ubtest --model_path path/to/model.pt
    python test.py --dataset all --use_default_models

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
from pathlib import Path

# Import your custom modules
from data import ProDataset
from utils import processing_fasta_file
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
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return


# Dataset configurations
DATASET_CONFIGS = {
    'test_60': {
        'data_file': 'Test_60.fa',
        'res_type': 'test_pdb_60_CA',
        'adj_type': '60_test_CA',
        'default_model': 'model_trained/saved_models/Best_EQAGNNModel_test_60.pt'
    },
    'test_315': {
        'data_file': 'Test_315.fa',
        'res_type': 'test_pdb_315_CA',
        'adj_type': '315_test_CA',
        'default_model': 'model_trained/saved_models/Best_EQAGNNModel_test_315.pt'
    },
    'ubtest': {
        'data_file': 'Ubtest.fa',  # Update this with correct filename
        'res_type': 'ubtest_pdb_CA',  # Update this with correct res_type
        'adj_type': 'ubtest_CA',  # Update this with correct adj_type
        'default_model': 'model_trained/saved_models/Best_EQAGNNModel_Ubtest.pt'
    }
}


def create_data_loader(threshold=14, batch_size=1, num_workers=2,
                    which_data='Test_60.fa', res_type='test_pdb_60_CA',
                    adj_type='60_test_CA', FP='./main/Feature/', shuffle=False):
    """Create data loader for the specified dataset"""
    
    data_path = f'./main/Dataset/{which_data}'
    
    # Check if data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    pdb_ids, sequence, labels, input_files = processing_fasta_file(data_path)
    
    RPP = f'./main/Res_positions/{res_type}_res_pos.pkl'
    AP = f'./main/Input_adj/Adj_matrix_{adj_type}'
    
    # Check if required files exist
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


def evaluate(model, dataloader, device, dataset_name="Test", print_freq=10):
    """Evaluate model on the given dataloader"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    
    y_true = []
    y_pred = []
    
    end = time.time()
    
    print(f"\nEvaluating on {dataset_name} dataset...")
    print("-" * 80)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            out = model(batch)
            
            # Class weights for imbalanced dataset
            sc = torch.tensor([1.0, 4.0]).to(device)
            loss = F.cross_entropy(out, batch.y, weight=sc)
            
            batch_size = 1
            losses.update(loss.item(), batch_size)
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    f'{dataset_name}',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(dataloader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                ])
                print(res)
            
            y_true.append(batch.y.cpu().numpy())
            y_pred.append(out.argmax(dim=1).cpu().numpy())
    
    # Concatenate all predictions
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    # Compute metrics
    auc = compute_roc(y_pred, y_true)
    aupr = compute_aupr(y_pred, y_true)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(y_pred, y_true)
    acc_val = acc_score(predictions_max, y_true)
    mcc = compute_mcc(predictions_max, y_true)
    
    results = {
        'loss': losses.avg,
        'accuracy': acc_val,
        'f1_score': f_max,
        'precision': p_max,
        'recall': r_max,
        'auc': auc,
        'aupr': aupr,
        'mcc': mcc,
        'threshold': t_max,
        'avg_time': batch_time.avg
    }
    
    return results


def print_results(dataset_name, results):
    """Print evaluation results in a formatted manner"""
    print("\n" + "=" * 80)
    print(f"RESULTS FOR {dataset_name.upper()}")
    print("=" * 80)
    print(f"Loss:       {results['loss']:.4f}")
    print(f"Accuracy:   {results['accuracy']:.4f}")
    print(f"F1 Score:   {results['f1_score']:.4f}")
    print(f"Precision:  {results['precision']:.4f}")
    print(f"Recall:     {results['recall']:.4f}")
    print(f"AUPR:       {results['aupr']:.4f}")
    print(f"MCC:        {results['mcc']:.4f}")
    print(f"Threshold:  {results['threshold']:.4f}")
    print(f"Avg Time:   {results['avg_time']:.3f}s per batch")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Test EQAGNN model on different datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test on Test_60 dataset with default model
    python test.py --dataset test_60
    
    # Test on Test_315 dataset with custom model
    python test.py --dataset test_315 --model_path path/to/custom_model.pt
    
    # Test on all datasets
    python test.py --dataset all
    
    # Test with different parameters
    python test.py --dataset test_60 --batch_size 2 --num_workers 4
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['test_60', 'test_315', 'ubtest', 'all'],
                        help='Dataset to evaluate on')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (uses default if not specified)')
    parser.add_argument('--use_default_models', action='store_true',
                        help='Use default models for each dataset (only with --dataset all)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--threshold', type=int, default=14,
                        help='Threshold parameter for dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for evaluation')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency during evaluation')
    
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
    
    # Initialize model
    model = EQAGNN_Model(
        num_layers=8,
        in_dim=62,
        out_dim=2,
        s_dim=62,
        s_dim_edge=8,
        equivariant_pred=False
    )
    
    # Determine which datasets to evaluate
    if args.dataset == 'all':
        datasets_to_eval = ['test_60', 'test_315', 'ubtest']
    else:
        datasets_to_eval = [args.dataset]
    
    # Evaluate on each dataset
    all_results = {}
    
    for dataset_name in datasets_to_eval:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name}")
        print(f"{'='*80}")
        
        config = DATASET_CONFIGS[dataset_name]
        
        # Determine model path
        if args.dataset == 'all' and args.use_default_models:
            model_path = config['default_model']
        elif args.model_path:
            model_path = args.model_path
        else:
            model_path = config['default_model']
        
        # Load model
        print(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            print(f"Skipping {dataset_name}")
            continue
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print(f"Skipping {dataset_name}")
            continue
        
        # Create data loader
        try:
            dataloader = create_data_loader(
                threshold=args.threshold,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                which_data=config['data_file'],
                res_type=config['res_type'],
                adj_type=config['adj_type'],
                shuffle=False
            )
            print(f"Data loader created successfully!")
        except Exception as e:
            print(f"ERROR creating data loader: {e}")
            print(f"Skipping {dataset_name}")
            continue
        
        # Evaluate
        results = evaluate(model, dataloader, device, dataset_name, args.print_freq)
        all_results[dataset_name] = results
        
        # Print results
        print_results(dataset_name, results)
    
    # Summary if multiple datasets
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("SUMMARY OF ALL RESULTS")
        print("="*80)
        print(f"{'Dataset':<15} {'Acc':<8} {'F1':<8} {'AUC':<8} {'AUPR':<8} {'MCC':<8}")
        print("-"*80)
        for dataset_name, results in all_results.items():
            print(f"{dataset_name:<15} "
                f"{results['accuracy']:<8.4f} "
                f"{results['f1_score']:<8.4f} "
                f"{results['auc']:<8.4f} "
                f"{results['aupr']:<8.4f} "
                f"{results['mcc']:<8.4f}")
        print("="*80)


if __name__ == '__main__':
    main()