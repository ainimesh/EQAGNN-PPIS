import numpy as np
import torch
from torch_geometric.loader import DataLoader
import time, random
import torch.nn.functional as F
from tqdm import tqdm
import os, csv

from utils import processing_fasta_file
from data import ProDataset
from evalution import compute_roc, compute_aupr, compute_mcc, micro_score, acc_score, compute_performance
from main.method.model import EQAGNN_Model, GVPGNNModel



class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all the metrics to their initial state.
        """
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# writing efficient code to handle data

def worker_init_fn(worker_id):                                                                                                                                
    seed = 3                                                                                                             
    torch.manual_seed(seed)                                                                                                                                   
    torch.cuda.manual_seed(seed)                                                                                                                              
    torch.cuda.manual_seed_all(seed)                                                                                          
    np.random.seed(seed)                                                                                                             
    random.seed(seed)                                                                                                       
    torch.manual_seed(seed)                                                                                                                                   
    return

def create_data_loader(threshold = 14, batch_size = 1, num_workers = 2, 
                    which_data = 'Train_332.fa' , res_type = 'train_pdb_332_SC', adj_type = '335_train_SC', shuffle = True):
    
    data_path = f'./Dataset/{which_data}'
    pdb_ids, sequence, labels, input_files_train = processing_fasta_file(data_path)

    RPP = f'./Res_positions/{res_type}_res_pos.pkl'
    AP = f'./Input_adj/Adj_matrix_{adj_type}'
    
    dataset = ProDataset(pdb_ids, sequence, labels, threshold = threshold, Res_Position_Path= RPP, Adj_path=AP,
                seq = False, pbert = False, pstruct = False, patom = False, all_feat = True)
    data_loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)
    
    return data_loader





# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f"Using device: {device}")




def train_epoch(model, dataloader, optimizer, device, epoch, all_epochs, print_freq=100):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        sc = torch.tensor([1.0, 4.0]).to(device)
        loss = F.cross_entropy(out, batch.y, weight=sc)
        batch_size = 1

        MiP, MiR, MiF, PNum, RNum = micro_score(out.argmax(dim=1).cpu().numpy(),
                                                batch.y.cpu().numpy())
        
        losses.update(loss.item(), batch_size)
        optimizer.zero_grad()


        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, all_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(dataloader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'f_max:%.6f' % (MiP),
                'p_max:%.6f' % (MiR),
                'r_max:%.6f' % (MiF),
                't_max:%.2f' % (PNum)])
            print(res)


    return batch_time.avg, losses.avg

def evaluate(model, dataloader, device, print_freq=10, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    val_loss = 0.0
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

            # measure elapsed time
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


            # val_loss += loss.item()
            y_true.append(batch.y.cpu().numpy())
            y_pred.append(out.argmax(dim=1).cpu().numpy())
            # print(batch.y.cpu().numpy())
            # print(out.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    auc = compute_roc(y_pred, y_true)
    aupr = compute_aupr(y_pred, y_true)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(y_pred, y_true)
    acc_val = acc_score(predictions_max, y_true)
    mcc = compute_mcc(predictions_max, y_true)

    return batch_time.avg, losses.avg, acc_val, f_max, p_max, r_max, auc, aupr, t_max, mcc



def train(model, num_epochs, train_dataloader = None, val_dataloader = None, verbose=True, device='cpu'):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    model = model.to(device)

    # Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    


    # AdamW
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-5)

    #  ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=2, min_lr=0.00001)
    
    #  cosine annealing learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)


    if not os.path.exists('gvp_model_trained'):
        os.makedirs('gvp_model_trained')

    if not os.path.exists('gvp_results'):
        os.makedirs('gvp_results')

    best_val_metrics = {
        'mcc': 0.486,
        'f1': 0.564,
        'auprc': 0.562
    }

    results_file = f"gvp_results/{type(model).__name__}_results_test_315_seed3.csv" 
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy', 'Test F1', 'Test Precision', 'Test Recall', 'Test AUC', 'Test AUPRC', 'Test MCC'])

    if verbose:
        print(f"Running experiment for {type(model).__name__}.")
        print(f'Total parameters: {total_param}')
        print("\nStart training:")
    
    # for time measurment
    t = time.time()

    for epoch in tqdm(range(num_epochs)):
        _, train_loss = train_epoch(model, train_dataloader, optimizer, device, epoch, all_epochs=num_epochs)
        _, valid_loss, acc, f_max, p_max, r_max, auc, aupr, t_max, mcc = evaluate(model, val_dataloader, device, is_test=True)


        scheduler.step(acc)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{num_epochs}, LR: {lr:.5f}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {valid_loss:.4f}, Test Accuracy: {acc:.4f}, Test F1: {f_max:.4f}, Test MCC: {mcc:.4f}, Test AUPRC: {aupr:.4f}, Test AUC: {auc:.4f}, Test Precision: {p_max:.4f}, Test Recall: {r_max:.4f}")
        print()

        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, round(train_loss, 3), round(valid_loss, 3), round(acc, 3), round(f_max, 3), round(p_max, 3), round(r_max, 3), round(auc, 3), round(aupr, 3), round(mcc, 3) ])

        if mcc > best_val_metrics['mcc'] and f_max > best_val_metrics['f1'] and aupr > best_val_metrics['auprc']:
            torch.save(model.state_dict(), f"gvp_model_trained/best_{type(model).__name__}_epoch{epoch}_test_315_seed_3.pt")
    
        
    t = time.time() - t
    train_time = t
    print(f"\nDone! Training took {train_time:.2f}s.")

if __name__ == '__main__':

    # set seed
    seed_everything(seed=3) # to ensure reproducibility

    # creating dataloaders

    train_loader = create_data_loader(threshold = 14, batch_size = 1, num_workers = 2, 
                                        which_data = 'Train_332.fa' , res_type = 'train_pdb_332_CA', 
                                        adj_type = '335_train_CA', shuffle = True)
    
    test_loader_315 = create_data_loader(threshold = 14, batch_size = 1, num_workers = 2, 
                                        which_data = 'Test_315.fa' , res_type = 'test_pdb_315_CA', 
                                        adj_type = '315_test_CA', shuffle = False)


    # Set model
    model_name = "gvp_att"
    model_ = {
        "gvp": GVPGNNModel,
        "gvp_att": GVP_AAMPModel,
    }[model_name](num_layers=8, in_dim=62, out_dim=2, s_dim=62, s_dim_edge=8, equivariant_pred = False)

    train(model_, train_dataloader = train_loader, val_dataloader = test_loader_315, num_epochs = 50, verbose = True, device = device)
