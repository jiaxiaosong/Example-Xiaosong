
import os
# num_thread = 2
# os.environ["OMP_NUM_THREADS"] = str(num_thread) # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = str(num_thread) # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = str(num_thread) # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_thread) # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = str(num_thread) # export NUMEXPR_NUM_THREADS=1
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
from ogb.linkproppred import DglLinkPropPredDataset
from ogb.linkproppred import Evaluator


parser = argparse.ArgumentParser('Interface')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
parser.add_argument('--device', type=str, default="cuda:0", help='idx for the gpu to use')
parser.add_argument('--name', type=str, default="", help='The name of this setting')
parser.add_argument('--pretrained', type=str, default="None", help='The position of pretrained model')
parser.add_argument('--val_interval', type=int, default=1, help='every number of epoches to evaluate')
parser.add_argument('--test_interval', type=int, default=1, help='every number of epoches to test')
parser.add_argument('--snapshot_interval', type=int, default=5, help='every number of epoches to save snapshot of model')
parser.add_argument('--expand_factor', type=int, default=20, help='sampling neighborhood size')
parser.add_argument('--model_file', type=str, default="model", help='the model file')
parser.add_argument('--d_attn_ratio', type=int, default=1, help='the ratio for dimension of attention layer compared to node dimension')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--inner_layer', type=int, default=1, help='the number of Transformers within each GNN layers')
parser.add_argument('--encoding', type=str, default="temporal", help='temporal encoding method')
args = parser.parse_args()



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
dataset_name = "Citation"
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


evaluator = Evaluator(name = "ogbl-citation")
print(evaluator.expected_input_format) 
print(evaluator.expected_output_format) 


dataset = DglLinkPropPredDataset(name="ogbl-citation")
split_edge = dataset.get_edge_split()
num_worker = 16
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
graph = dataset[0]
origin_graph = copy.deepcopy(graph)
graph.readonly(False)
graph.add_edges(graph.edges()[1], graph.edges()[0])
graph.add_edges(torch.arange(0, graph.number_of_nodes()).long(), torch.arange(0, graph.number_of_nodes()).long())
graph.edata["etype"] = torch.cat([torch.ones((graph.number_of_edges()-graph.number_of_nodes())//2).long(), (torch.ones((graph.number_of_edges()-graph.number_of_nodes())//2)*2).long(), torch.zeros(graph.number_of_nodes()).long()], dim=0)
graph.readonly()
neg_num = 1000


all_node =  set(graph.nodes().tolist())
class Matcher(torch.nn.Module):
    "Neural Tensor Network"
    def __init__(self, n_hid):
        super(Matcher, self).__init__()
        self.n_hid = n_hid
        self.W = torch.nn.Bilinear(in1_features=n_hid, in2_features=n_hid, out_features=n_hid//2, bias=True)
        self.V = torch.nn.Linear(in_features=n_hid*2, out_features=n_hid//2, bias=False)
        self.U = torch.nn.Linear(in_features=n_hid//2, out_features=1, bias=False)
    def forward(self, x, y):
        return self.U(torch.nn.functional.gelu(self.W(x, y)+ self.V(torch.cat([x,y], dim=-1))))


class Matcher_v2(torch.nn.Module):
    def __init__(self, n_hid):
        super(Matcher_v2, self).__init__()
        self.n_hid = n_hid
        self.left_fc1 = torch.nn.Linear(in_features=n_hid, out_features=n_hid)
        self.left_fc2 = torch.nn.Linear(in_features=n_hid, out_features=n_hid)
        self.left_ln = torch.nn.LayerNorm(n_hid)
        self.right_fc1 = torch.nn.Linear(in_features=n_hid, out_features=n_hid)
        self.right_fc2 = torch.nn.Linear(in_features=n_hid, out_features=n_hid)
        self.right_ln = torch.nn.LayerNorm(n_hid)
        self.fc1 = torch.nn.Linear(in_features=n_hid, out_features=n_hid//2)
        self.fc2 = torch.nn.Linear(in_features=n_hid//2, out_features=1, bias=False)
        self.act = torch.nn.functional.gelu
        self.n_hid = n_hid
    def forward(self, x, y):
        x = self.left_ln(self.left_fc2(self.act(self.left_fc1(x)))+x)
        y = self.right_ln(self.right_fc2(self.act(self.right_fc1(y)))+y).view(neg_num+1, x.shape[0],self.n_hid)
        out = x.unsqueeze(0)*y
        out = self.fc2(self.act(self.fc1(out)))
        return out

####Normalization
fea_mean, fea_std = graph.ndata["feat"].mean(dim=0), graph.ndata["feat"].std(dim=0)
graph.ndata["feat"] = (graph.ndata["feat"] - fea_mean) / fea_std

gnn_model = model_module.Model(input_dim=fea_mean.shape[-1], args=args)
link_classifier = Matcher(n_hid=fea_mean.shape[-1]*args.d_attn_ratio)
if args.pretrained != "None":
    print("Load:", args.pretrained)
    gnn_model.load_state_dict(torch.load(args.pretrained+"_gnn.model", map_location="cpu"))
    link_classifier.load_state_dict(torch.load(args.pretrained+"_link_cls.model", map_location="cpu"))
gnn_model.to(device)
link_classifier.to(device)
optimizer = torch.optim.AdamW(list(gnn_model.parameters())+list(link_classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)



criterion = torch.nn.BCEWithLogitsLoss().to(device)


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






def train(edge_dic):
    all_acc, all_loss = AverageMeter(), AverageMeter()
    gnn_model.train()
    link_classifier.train()

    data_index = list(range(edge_dic["source_node"].shape[0]))
    random.shuffle(data_index)
    start_time = time.time()
    print_interval = len(data_index)//2//args.batch_size*args.batch_size
    for batch_index in range(0, len(data_index), args.batch_size):
        optimizer.zero_grad()
        batch_sample_global_index = data_index[batch_index:batch_index+args.batch_size]
        
        src_node = edge_dic["source_node"][batch_sample_global_index]
        tgt_node = edge_dic["target_node"][batch_sample_global_index]
        neg_tgt_node = torch.randint(low=0, high=len(all_node), size=(int(src_node.shape[0]),), dtype=torch.long)

        label = torch.Tensor([1]*src_node.shape[0]+[0]*src_node.shape[0])
        seed_nodes = torch.cat([src_node, tgt_node, neg_tgt_node], dim=-1)
        for nf in dgl.contrib.sampling.NeighborSampler(g=graph, batch_size=seed_nodes.shape[0], expand_factor=args.expand_factor, neighbor_type='in', shuffle=False, num_hops=args.n_layer, seed_nodes=seed_nodes, num_workers=num_worker, add_self_loop=True):
            nf.copy_from_parent(ctx=torch.device(device))
            optimizer.zero_grad()
            node_emb = gnn_model(nf)[nf.map_from_parent_nid(layer_id=args.n_layer, parent_nids=seed_nodes.type(torch.int64), remap_local=True),...]
            src_emb, tgt_emb, neg_tgt_emb = torch.split(node_emb, node_emb.shape[0]//3)
            prob = link_classifier(src_emb.repeat(2,1), torch.cat([tgt_emb, neg_tgt_emb], dim=0)).squeeze()
            loss = criterion(prob, label.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                label = label.cpu().detach().numpy()
                pred_score = prob.sigmoid().cpu().detach().numpy()
                pred_label = pred_score > 0.5
                
                all_loss.update(loss.item(), label.shape[0])
                all_acc.update((pred_label==label).mean(), label.shape[0])

        if batch_index!=0 and  batch_index%print_interval == 0:
            print("Time:", time.time()-start_time)
            print_text = 'Epoch: [{0}][{1}/{2}], Acc {all_acc.avg:.4f}, Loss {all_loss.avg:.6f}'.format(epoch, batch_index, len(data_index), all_acc=all_acc, all_loss=all_loss)
            print(print_text)

            file_path1 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_batch_index_"+str(batch_index)+"_gnn.model")
            file_path2 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_batch_index_"+str(batch_index)+"_link_cls.model")
            torch.save(gnn_model.state_dict(), file_path1)
            torch.save(link_classifier.state_dict(), file_path2)
            evaluate(test_edge, state ="Test")
            evaluate(valid_edge, state = "Val")

epoch_num = args.n_epoch
best_mrr = 0
best_mrr_epoch = 0
def evaluate(edge_dic, state):
    global best_mrr, best_mrr_epoch
    gnn_model.eval()
    link_classifier.eval()

    start_time = time.time()
    batch_size = 3

    mrr_lis = []
    with torch.no_grad():
        for batch_index in range(0, len(edge_dic["source_node"]), batch_size):
            src_node = edge_dic["source_node"][batch_index:batch_index+batch_size]
            tgt_node = edge_dic["target_node"][batch_index:batch_index+batch_size]
            neg_tgt_node = edge_dic["target_node_neg"][batch_index:batch_index+batch_size]
            seed_nodes = torch.cat([src_node, tgt_node, neg_tgt_node.permute(1,0).reshape(-1)], dim=-1)
            for nf in dgl.contrib.sampling.NeighborSampler(g=graph, batch_size=seed_nodes.shape[0], expand_factor=args.expand_factor, neighbor_type='in', shuffle=False, num_hops=args.n_layer, seed_nodes=seed_nodes, num_workers=num_worker, add_self_loop=True):
                nf.copy_from_parent(ctx=torch.device(device))
                optimizer.zero_grad()
                node_emb = gnn_model(nf)[nf.map_from_parent_nid(layer_id=args.n_layer, parent_nids=seed_nodes.type(torch.int64), remap_local=True),...]
                src_emb, tgt_emb, neg_tgt_emb = node_emb[:src_node.shape[0]], node_emb[src_node.shape[0]:src_node.shape[0]+tgt_node.shape[0]], node_emb[src_node.shape[0]+tgt_node.shape[0]:]
                prob = link_classifier(src_emb.repeat(neg_num+1,1), torch.cat([tgt_emb, neg_tgt_emb], dim=0)).sigmoid().squeeze().detach()
                output = evaluator.eval(({"y_pred_pos":prob[:prob.shape[0]//(neg_num+1)], "y_pred_neg":prob[prob.shape[0]//(neg_num+1):].view(neg_num,-1).permute(1,0)}))
                mrr_lis.append(output["mrr_list"].cpu().detach().numpy())


    mrr = np.concatenate(mrr_lis, axis=0).mean()
    print("Time:", time.time()-start_time)
    print_text = 'Epoch: [{0}][{1}/{2}], MRR {3:.4f}'.format(epoch, batch_index, len(edge_dic["source_node"]), mrr)
    if state == "Val":
        if best_mrr < mrr:
            best_mrr = mrr
            best_mrr_epoch = epoch
        print_text = "**** " + print_text + ", Best Val MRR {1:.4f} (Epoch {2})".format(epoch, best_mrr, best_mrr_epoch)
    if state == "Test":
        print_text = "!!!! " + print_text
    print(print_text)

for epoch in range(1, args.n_epoch+1):
    #Train
    train(train_edge)
    if epoch % args.snapshot_interval == 0:
        print("Epoch %d Save Model"%(epoch))
        file_path1 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_gnn.model")
        file_path2 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_link_cls.model")
        torch.save(gnn_model.state_dict(), file_path1)
        torch.save(link_classifier.state_dict(), file_path2)
    if epoch % args.val_interval == 0:
       evaluate(valid_edge, state = "Val")
    if  epoch % args.test_interval == 0:
        print("Test Epoch %d"%(epoch))
        evaluate(test_edge, state ="Test")
