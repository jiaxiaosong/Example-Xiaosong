
from ogb.nodeproppred import DglNodePropPredDataset
import os
num_thread = 2
os.environ["OMP_NUM_THREADS"] = str(num_thread) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(num_thread) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(num_thread) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_thread) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(num_thread) # export NUMEXPR_NUM_THREADS=1
import dgl
import pandas as pd 
import numpy as np
import torch
import random
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import argparse
import time
import datetime
import math
import sys
import pickle
import copy
import shutil
import importlib
from sklearn.metrics import roc_auc_score
import scipy.sparse as spp
import sklearn.utils

parser = argparse.ArgumentParser('Interface')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--n_layer', type=int, default=3, help='number of network layers')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
parser.add_argument('--device', type=str, default="cuda:0", help='idx for the gpu to use')
parser.add_argument('--name', type=str, default="", help='The name of this setting')
parser.add_argument('--pretrained', type=str, default="None", help='The position of pretrained model')
parser.add_argument('--val_interval', type=int, default=1, help='every number of epoches to evaluate')
parser.add_argument('--test_interval', type=int, default=1, help='every number of epoches to test')
parser.add_argument('--snapshot_interval', type=int, default=5, help='every number of epoches to save snapshot of model')
parser.add_argument('--expand_factor', type=int, default=100, help='sampling neighborhood size')
parser.add_argument('--model_file', type=str, default="model", help='the model file')
parser.add_argument('--d_attn_ratio', type=int, default=1, help='the ratio for dimension of attention layer compared to node dimension')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--inner_layer', type=int, default=1, help='the number of Transformers within each GNN layers')
parser.add_argument('--seed_num', type=int, default=0, help='random seed num')
parser.add_argument('--encoding', type=str, default="temporal", help='temporal encoding method')
parser.add_argument('--weighted_loss', type=str, default="False", help='whether using weighted loss for each class')
args = parser.parse_args()



d_name = "ogbn-arxiv"
dataset = DglNodePropPredDataset(name = d_name)
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
graph.readonly(False)
class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(label.numpy()), y=label.numpy().ravel())
class_weights = torch.Tensor(class_weights/class_weights.sum())
## 0 - self-loop, 1 - origin_edges, 2 - reverse edges
graph.add_edges(graph.edges()[1], graph.edges()[0])
graph.add_edges(torch.arange(0, graph.number_of_nodes()).long(), torch.arange(0, graph.number_of_nodes()).long())
graph.edata["etype"] = torch.cat([torch.ones((graph.number_of_edges()-graph.number_of_nodes())//2).long(), (torch.ones((graph.number_of_edges()-graph.number_of_nodes())//2)*2).long(), torch.zeros(graph.number_of_nodes()).long()], dim=0)
graph.readonly()


label = label.squeeze()
seed_num = args.seed_num
torch.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)
model_file = args.model_file
model_module = importlib.import_module(model_file)
class Logger():
    def __init__(self, lognames):
        self.terminal = sys.stdout
        self.logs = []
        for log_name in lognames:
            self.logs.append(open(log_name, 'w'))
    def write(self, message):
        self.terminal.write(message)
        for log in self.logs:
            log.write(message)
            log.flush()
    def flush(self):
        pass
dataset_name = "Arxiv"
setting_name = args.name
device = args.device#"cuda:0"#"cpu"
log_dir = str(dataset_name+"_"+setting_name+"_"+time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())))
os.mkdir(log_dir)
sys.stdout = Logger(["%s.log"%(dataset_name+setting_name), os.path.join(log_dir, "%s.log"%(dataset_name+"_"+setting_name))])
sys.stderr = Logger(["%s.log"%(dataset_name+setting_name), os.path.join(log_dir, "%s.log"%(dataset_name+"_"+setting_name))])
snapshot_dir = os.path.join(log_dir, "snapshot")
if not os.path.isdir(snapshot_dir):
    os.makedirs(snapshot_dir)
print("Process Id:", os.getpid())
print(os.path.join(log_dir, sys.argv[0]))
print(args)
shutil.copyfile(__file__, os.path.join(log_dir, "train.py"))
shutil.copyfile(model_file+".py", os.path.join(log_dir, model_file+".py"))

class LR(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, (input_dim+output_dim)//2)
        self.fc2 = torch.nn.Linear((input_dim+output_dim)//2, output_dim)
        self.act = torch.nn.functional.gelu
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

####Normalization
fea_mean, fea_std = graph.ndata["feat"][train_idx].mean(dim=0), graph.ndata["feat"][train_idx].std(dim=0)
graph.ndata["feat"] = (graph.ndata["feat"] - fea_mean) / fea_std


gnn_model = model_module.Model(input_dim=fea_mean.shape[-1], args=args)
gnn_model.to(device)
node_classifier = LR(input_dim=fea_mean.shape[-1]*args.d_attn_ratio, output_dim=40)
node_classifier.to(device)
if args.weighted_loss == "True":
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device)/40).to(device)
else:
    criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(list(gnn_model.parameters())+list(node_classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)

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

epoch_num = args.n_epoch
best_acc = 0
best_acc_epoch = 0
def run_model(data_index, state):
    global best_acc, best_acc_epoch
    all_acc, all_loss = AverageMeter(), AverageMeter()
    if state == "Train":
        is_train = True
        gnn_model.train()
        node_classifier.train()
    else:
        is_train = False
        gnn_model.eval()
        node_classifier.eval()
    random.shuffle(data_index)

    start_time = time.time()
    with torch.set_grad_enabled(is_train):
        for batch_index in range(0, len(data_index), args.batch_size):
            batch_sample_global_index = data_index[batch_index:batch_index+args.batch_size]
            
            batch_label = label[batch_sample_global_index].to(device)
            for nf in dgl.contrib.sampling.NeighborSampler(g=graph, batch_size=args.batch_size, expand_factor=args.expand_factor, neighbor_type='in', shuffle=False, num_hops=args.n_layer, seed_nodes=batch_sample_global_index, num_workers=1, add_self_loop=True):
                nf.copy_from_parent(ctx=torch.device(device))
                optimizer.zero_grad()
                node_emb = gnn_model(nf)[nf.map_from_parent_nid(layer_id=args.n_layer, parent_nids=torch.Tensor(batch_sample_global_index).type(torch.int64), remap_local=True),...]
                prob = node_classifier(node_emb).squeeze()
                loss = criterion(prob, batch_label)
                with torch.no_grad():
                    all_loss.update(loss.item(), batch_label.shape[0])
                    _, preds = torch.max(prob.data, 1)
                    all_acc.update((preds==batch_label).sum().item()/batch_label.shape[0], batch_label.shape[0])
                if is_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        print("Time:", time.time()-start_time)
        print_text = state+' Epoch: [{0}][{1}/{2}], Acc {all_acc.avg:.4f}, Loss {all_loss.avg:.6f}'.format(epoch, batch_index, len(data_index), all_acc=all_acc, all_loss=all_loss)
        if state == "Val":
            if best_acc < all_acc.avg:
                best_acc = all_acc.avg
                best_acc_epoch = epoch
            print_text = "**** " + print_text + ", Best Val Acc {1:.4f} (Epoch {2})".format(epoch, best_acc, best_acc_epoch)
        if state == "Test":
            print_text = "!!!! " + print_text
        print(print_text)

train_idx = train_idx.tolist()
valid_idx = valid_idx.tolist()
test_idx = test_idx.tolist()
for epoch in range(1, args.n_epoch+1):
    #Train
    run_model(train_idx, state = "Train")
    if  epoch % args.snapshot_interval == 0:
        print("Epoch %d Save Model"%(epoch))
        file_path1 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_gnn.model")
        file_path2 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_node_cls.model")
        torch.save(gnn_model.state_dict(), file_path1)
        torch.save(node_classifier.state_dict(), file_path2)
        
    if epoch % args.val_interval == 0:
       run_model(valid_idx, state = "Val")
    if epoch % args.test_interval == 0:
        run_model(test_idx, state ="Test")
