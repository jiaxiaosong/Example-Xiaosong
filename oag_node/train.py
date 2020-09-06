
import os
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
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser('Interface')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
parser.add_argument('--n_layer', type=int, default=1, help='number of network layers')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
parser.add_argument('--device', type=str, default="cuda:0", help='idx for the gpu to use')
parser.add_argument('--name', type=str, default="", help='The name of this setting')
parser.add_argument('--pretrained', type=str, default="None", help='The position of pretrained model')
parser.add_argument('--val_interval', type=int, default=1, help='every number of epoches to evaluate')
parser.add_argument('--test_interval', type=int, default=1, help='every number of epoches to test')
parser.add_argument('--snapshot_interval', type=int, default=1, help='every number of epoches to save snapshot of model')
parser.add_argument('--expand_factor', type=int, default=100, help='sampling neighborhood size')
parser.add_argument('--model_file', type=str, default="model", help='the model file')
parser.add_argument('--d_attn_ratio', type=int, default=1, help='the ratio for dimension of attention layer compared to node dimension')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--inner_layer', type=int, default=1, help='the number of Transformers within each GNN layers')
parser.add_argument('--encoding', type=str, default="temporal", help='temporal encoding method')
parser.add_argument('--weighted_loss', type=str, default="False", help='whether using weighted loss for each class')
args = parser.parse_args()

if not os.path.exists("./oag_node.bin"):
    import urllib.request
    from tqdm import tqdm
    import tarfile
    import os
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

        def download_url(self, url, output_path):
            with DownloadProgressBar(unit='B', unit_scale=True,
                                    miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    downloader = DownloadProgressBar()
    print("Download OAG CS Node Classification dATA")
    downloader.download_url("https://s3.us-west-2.amazonaws.com/dgl-data/dataset/temporal/oag_cs.dgl", "oag_node.bin")

from dgl.data.utils import load_graphs
graph, label_dic = load_graphs("./oag_node.bin")
graph = graph[0]
graph.readonly(False)
graph.add_edges(graph.edges()[1], graph.edges()[0])
graph.add_edges(torch.arange(0, graph.number_of_nodes()).long(), torch.arange(0, graph.number_of_nodes()).long())
graph.edata["etype"] = torch.cat([torch.ones((graph.number_of_edges()-graph.number_of_nodes())//2).long(), (torch.ones((graph.number_of_edges()-graph.number_of_nodes())//2)*2).long(), torch.zeros(graph.number_of_nodes()).long()], dim=0)
graph.readonly()
label = label_dic["label"]
train_idx, valid_idx, test_idx = label_dic["train_idx"], label_dic["valid_idx"], label_dic["test_idx"]
num_workers = 16

fea_mean, fea_std = graph.ndata["feat"][train_idx].mean(dim=0), graph.ndata["feat"][train_idx].std(dim=0)
graph.ndata["feat"] = (graph.ndata["feat"] - fea_mean) / fea_std
train_idx = train_idx.tolist()
valid_idx = valid_idx.tolist()
test_idx = test_idx.tolist()


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
dataset_name = "OAG"
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
        self.fc2 = torch.nn.Linear((input_dim+output_dim)//2, output_dim, bias=False)
        self.act = torch.nn.functional.gelu
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

####Normalization
gnn_model = model_module.Model(input_dim=fea_mean.shape[-1], args=args)
gnn_model.to(device)
node_classifier = LR(input_dim=fea_mean.shape[-1]*args.d_attn_ratio, output_dim=label.shape[1])
node_classifier.to(device)
criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
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
best_macro_f1 = 0
best_macro_epoch = 0
best_micro_f1 = 0
best_micro_epoch = 0
def run_model(data_index, state):
    global best_macro_f1, best_macro_epoch, best_micro_f1, best_micro_epoch
    all_loss = AverageMeter()
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
    tmp_label = []
    tmp_pred = []
    with torch.set_grad_enabled(is_train):
        for batch_index in range(0, len(data_index), args.batch_size):
            batch_sample_global_index = data_index[batch_index:batch_index+args.batch_size]
            
            batch_label = label[batch_sample_global_index].to(device)
            for nf in dgl.contrib.sampling.NeighborSampler(g=graph, batch_size=args.batch_size, expand_factor=args.expand_factor, neighbor_type='in', shuffle=False, num_hops=args.n_layer, seed_nodes=batch_sample_global_index, num_workers=num_workers, add_self_loop=True):
                nf.copy_from_parent(ctx=torch.device(device))
                optimizer.zero_grad()
                node_emb = gnn_model(nf)[nf.map_from_parent_nid(layer_id=args.n_layer, parent_nids=torch.Tensor(batch_sample_global_index).type(torch.int64), remap_local=True),...]
                prob = node_classifier(node_emb)
                loss = criterion(prob, batch_label)

                with torch.no_grad():
                    all_loss.update(loss.item(), batch_label.shape[0])
                    preds = (prob.sigmoid().cpu().detach().numpy() > 0.5)
                    batch_label = (batch_label.cpu().detach().numpy())
                    tmp_label.append(batch_label)
                    tmp_pred.append(preds)
                    
                if is_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        tmp_label = np.concatenate(tmp_label, axis=0)
        tmp_pred = np.concatenate(tmp_pred, axis=0)
        report = classification_report(y_true=tmp_label, y_pred=tmp_pred, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]
        micro_f1 = report["micro avg"]["f1-score"]
        print("Time:", time.time()-start_time)
        print_text = state+' Epoch: [{0}][{1}/{2}], Macro-F1 {macro_f1:.4f}, Micro-F1 {micro_f1:.4f}, Loss {all_loss.avg:.6f}'.format(epoch, batch_index, len(data_index), macro_f1=macro_f1, all_loss=all_loss, micro_f1=micro_f1)        
        if state == "Val":
            if best_macro_f1 < macro_f1:
                best_macro_f1 = macro_f1
                best_macro_epoch = epoch
            if best_micro_f1 < micro_f1:
                best_micro_f1 = micro_f1
                best_micro_epoch = epoch
            print_text = "**** " + print_text + ", Best Val Macro-F1 {1:.4f} (Epoch {2}), Best Val Micro-F1 {3:.4f} (Epoch {4})".format(epoch, best_macro_f1, best_macro_epoch, best_micro_f1, best_micro_epoch)
        if state == "Test":
            print_text = "!!!! " + print_text
        print(print_text)

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