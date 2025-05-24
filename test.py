import numpy as np
import torch
from data import ProDataset
from utils import processing_fasta_file
from torch_geometric.loader import DataLoader
import time, random
import torch.nn.functional as F


# Imporing for performace computation
from evalution import compute_roc, compute_aupr, compute_mcc, micro_score, acc_score, compute_performance
from main.method.model import EQAGNN_Model

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# writing efficient code to handle data

def worker_init_fn(worker_id):                                                                                                                              
    seed = 0                                                                                                             
    torch.manual_seed(seed)                                                                                                                                   
    torch.cuda.manual_seed(seed)                                                                                                                              
    torch.cuda.manual_seed_all(seed)                                                                                          
    np.random.seed(seed)                                                                                                             
    random.seed(seed)                                                                                                       
    torch.manual_seed(seed)                                                                                                                               
    return

def create_data_loader(threshold = 14, batch_size = 1, num_workers = 2, 
                    which_data = 'Train_332.fa' , res_type = 'train_pdb_332_SC', adj_type = '335_train_SC',  FP='./main/Feature/', shuffle = True):
    
    data_path = f'./main/Dataset/{which_data}'
    pdb_ids, sequence, labels, input_files_train = processing_fasta_file(data_path)

    RPP = f'./main/Res_positions/{res_type}_res_pos.pkl'
    AP = f'./main/Input_adj/Adj_matrix_{adj_type}'
    
    dataset = ProDataset(pdb_ids, sequence, labels, threshold = threshold, Res_Position_Path= RPP, Adj_path=AP, Feat_path=FP,
                seq = False, pbert = False, pstruct = False, patom = False, all_feat = True)
    data_loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)
    
    return data_loader

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

            y_true.append(batch.y.cpu().numpy())
            y_pred.append(out.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    auc = compute_roc(y_pred, y_true)
    aupr = compute_aupr(y_pred, y_true)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(y_pred, y_true)
    acc_val = acc_score(predictions_max, y_true)
    mcc = compute_mcc(predictions_max, y_true)

    return batch_time.avg, losses.avg, acc_val, f_max, p_max, r_max, auc, aupr, t_max, mcc


if __name__ == '__main__':

    # set seed
    seed_everything(seed=0) # to ensure reproducibility

    # Set model
    model = EQAGNN_Model(num_layers=8, in_dim=62, out_dim=2, s_dim=62, s_dim_edge=8, equivariant_pred = False)

    #Loading saved models
    saved_model_paths = ['gvp_model_trained/saved_models/best-model-26.pt','gvp_model_trained/best_GVP_AAMPModel_epoch38_test_315.pt','gvp_model_trained/best_GVP_AAMPModel_epoch40_test_315_seed_38.pt']

    model.load_state_dict(torch.load(saved_model_paths[0]))
    model.to(device)


    # creating dataloaders

    # train_loader = create_data_loader(threshold = 14, batch_size = 1, num_workers = 2, 
    #                                     which_data = 'Train_332.fa' , res_type = 'train_pdb_332', 
    #                                     adj_type = '335_train', shuffle = True)
    
    # test_loader_315 = create_data_loader(threshold = 14, batch_size = 1, num_workers = 2, 
    #                                     which_data = 'Test_315.fa' , res_type = 'test_pdb_315_CA', 
    #                                     adj_type = '315_test_CA', shuffle = False)
    
    test_loader_60 = create_data_loader(threshold = 14, batch_size = 1, num_workers = 2, 
                                        which_data = 'Test_60.fa' , res_type = 'test_pdb_60_CA', 
                                        adj_type = '60_test_CA', shuffle = False)
    

    _, valid_loss, acc, f_max, p_max, r_max, auc, aupr, t_max, mcc = evaluate(model, test_loader_60, device, is_test=True)
    print(f"Test Loss: {valid_loss:.4f}, Test Accuracy: {acc:.4f}, Test F1: {f_max:.4f}, Test MCC: {mcc:.4f}, Test AUPRC: {aupr:.4f}, Test AUC: {auc:.4f}, Test Precision: {p_max:.4f}, Test Recall: {r_max:.4f}")
    print()