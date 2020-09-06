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
import scipy.sparse as spp


parser = argparse.ArgumentParser('Interface')

parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=4, help='number of network layers')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.15, help='dropout probability')
parser.add_argument('--device', type=str, default="cuda:0", help='idx for the gpu to use')
parser.add_argument('--name', type=str, default="", help='the name of this setting')
parser.add_argument('--tbatch_num', type=int, default=500, help='tbatch_num')
parser.add_argument('--val_interval', type=int, default=1, help='every number of epoches to evaluate')
parser.add_argument('--test_interval', type=int, default=1, help='every number of epoches to test')
parser.add_argument('--snapshot_interval', type=int, default=5, help='every number of epoches to save snapshot of model')
parser.add_argument('--neg_sampling_ratio', type=int, default=1, help='The ratio of negative sampling')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the tbatch')
parser.add_argument('--expand_factor', type=int, default=20, help='sampling neighborhood size')
parser.add_argument('--inner_layer', type=int, default=1, help='the number of Transformers within each GNN layers')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clip')
parser.add_argument('--d_hidden_ratio', type=int, default=1, help='the ratio for dimension of attention layer compared to node dimension')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--model_file', type=str, default="model", help='the model file')
parser.add_argument('--encoding', type=str, default="temporal", help='temporal encoding method')
args = parser.parse_args()

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, dropout=0.1):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.dropout = torch.nn.Dropout(p=dropout)
        dim_3 = (dim1+dim2)//2
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.LeakyReLU(negative_slope=0.2)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)

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

dataset_name = args.data
setting_name = "Transformer_Edge"+args.name
device = args.device#"cuda:0"#"cpu"
log_dir = str(dataset_name+"_"+setting_name+"_"+time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())))
os.mkdir(log_dir)

grad_dir = os.path.join(log_dir, "grad")
os.mkdir(grad_dir)

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

if not os.path.isdir("%s"%(args.data)):
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
    print("Download %s data"%(args.data))
    downloader.download_url("https://s3.us-west-2.amazonaws.com/dgl-data/dataset/temporal/%s.tar.gz"%(args.data), "%s.tar.gz"%(args.data))
    tarfile.open("%s.tar.gz"%(args.data)).extractall(path = "./")
    os.remove("%s.tar.gz"%(args.data))


linkage_df = pd.read_csv("./{}/processed_linkage_{}.csv".format(dataset_name, dataset_name), index_col=0)
#src, dst, t, label
edge_feature = np.load('./{}/processed_edge_feat_{}.npy'.format(dataset_name, dataset_name))
node_feature = np.load('./{}/processed_node_feat_{}.npy'.format(dataset_name, dataset_name))
num_node = max(linkage_df.u.max(), linkage_df.i.max())+1
num_edge = linkage_df.shape[0]

print("Node Num:", num_node, "Edge Num:", num_edge, "edge_feature_dim", edge_feature.shape[1])

##Normalize Features
linkage_df.ts = (linkage_df.ts - linkage_df.ts.mean()) / linkage_df.ts.std()
edge_feature = (edge_feature - edge_feature.mean(axis=0)) / (edge_feature.std(axis=0)+1e-17)
node_feature = (node_feature - node_feature.mean(axis=0)) / (node_feature.std(axis=0)+1e-17)
#linkage_df["feat"] = edge_feature.tolist()
node_feature = torch.Tensor(node_feature)#.to(device)
edge_feature = torch.Tensor(edge_feature)#.to(device)
timestamp_feature = torch.Tensor(linkage_df.ts).unsqueeze(-1)#.to(device)

### Train Val Test 0.7 - 0.15 - 0.15
entire_start_timestamp = float(linkage_df.ts.min())
val_start_timetamp, test_start_timetamp = list(np.quantile(linkage_df.ts, [0.70, 0.85]))
entire_end_timestamp = float(linkage_df.ts.max())

train_edge = linkage_df[linkage_df.ts<val_start_timetamp]
val_edge =  linkage_df[(linkage_df.ts>val_start_timetamp) & (linkage_df.ts<test_start_timetamp)]
test_edge = linkage_df[linkage_df.ts>test_start_timetamp]

###mask
all_val_u_node = set(list(val_edge.u) + list(test_edge.u))
all_val_i_node = set(list(val_edge.i) + list(test_edge.i))
masked_u_node = random.sample(all_val_u_node, int(len(all_val_u_node)*0.1))
masked_i_node = random.sample(all_val_i_node, int(len(all_val_i_node)*0.1))
train_edge = train_edge[(~train_edge['u'].isin(masked_u_node))&(~train_edge['i'].isin(masked_i_node))]

###Different from the TGAT paper, we use the model in t-batch instead of only one loss for each item-node in one of the three sets. It is the same evaluation method as in the original dataset paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
tbatch_num = args.tbatch_num#500.0
tbatch_timespan = float((train_edge.ts.max()-train_edge.ts.min())/float(tbatch_num))



###With cache
temporal_graph_dic = {}
def build_temporal_graph_cache(node_feature, edge_list, time_index, start_timestamp, tbatch_timespan, state):
    now_time = start_timestamp + (time_index+1) * tbatch_timespan
    cache_key = state+str(time_index)
    if cache_key not in temporal_graph_dic:
        tmp_dic = {}
        ###"now" prefix means all the node and edge until now_time; 
        ###"new" prefix means the edge appearing during [t_now-timespan, t_now)
        ### "local" means in the index system of local graph; if there is no "local", it means in the index system of global data
        now_all_edge = edge_list[edge_list.ts<=now_time].copy()
        new_edge = now_all_edge[now_all_edge.ts>=now_time-tbatch_timespan]
        ###No new sample to evaluate
        if new_edge.shape[0] == 0:
            temporal_graph_dic[cache_key] = None
            return None, None, None, None, None, None, None
        ##Each new edge is a sample and if there are multiple same edge (u, i) in the new graph, only keep the latest one
        new_edge = new_edge.groupby(["u"]).tail(1)
        now_all_edge.drop(new_edge.index, inplace=True)

        now_src_node = list(now_all_edge.u.unique())
        now_dst_node = list(now_all_edge.i.unique())
        now_all_node = now_src_node + now_dst_node
        now_all_node.sort()
        node_index_global2local = {}
        node_index_local2global = {}
        for i in range(len(now_all_node)):
            node_index_global2local[now_all_node[i]] = i
            node_index_local2global[i] = now_all_node[i]
        now_src_node = set(now_src_node)
        now_dst_node = set(now_dst_node)

        ##New Edge
        new_edge = new_edge[(new_edge['u'].isin(now_src_node).to_numpy()) & (new_edge['i'].isin(now_dst_node).to_numpy())]
        if new_edge.shape[0] == 0:
            temporal_graph_dic[cache_key] = None
            return None, None, None, None, None, None, None

        now_all_local_u = np.array(now_all_edge['u'].map(node_index_global2local))
        now_all_local_i = np.array(now_all_edge['i'].map(node_index_global2local))
        new_u_local = np.array(new_edge['u'].map(node_index_global2local))
        new_i_local = np.array(new_edge["i"].map(node_index_global2local))

        tmp_dic["induct_edge_lis_bool"] = None
        if state != "Train":
            induct_u_node =  new_edge['u'].isin(masked_u_node).to_numpy()
            induct_i_node = new_edge['i'].isin(masked_i_node).to_numpy()
            induct_edge_lis = induct_u_node | induct_i_node
            tmp_dic["induct_edge_lis_bool"] = induct_edge_lis
        
        #add bidirection + add self-loop
        adj = spp.coo_matrix((np.ones(int(now_all_local_u.shape[0]+now_all_local_i.shape[0]+len(now_all_node))), (np.concatenate([now_all_local_u, now_all_local_i, list(range(len(now_all_node)))]), np.concatenate([now_all_local_i, now_all_local_u, list(range(len(now_all_node)))]))))
        now_graph = dgl.DGLGraph(adj)
        now_graph.readonly()
        tmp_dic["graph"] = now_graph
        tmp_dic["now_all_node"] = now_all_node
        tmp_dic["now_all_edge"] = np.array(now_all_edge.index)
        tmp_dic["now_dst_node_local"] = set(now_all_local_i.tolist())
        tmp_dic["t_now"] = now_time
        tmp_dic["new_src_local"] = new_u_local
        tmp_dic["new_dst_local"] = new_i_local
        tmp_dic["new_src_node_label"] = np.array(new_edge.label)
        temporal_graph_dic[cache_key] = tmp_dic
    if temporal_graph_dic[cache_key]:
        tmp_dic = temporal_graph_dic[cache_key]
        now_graph = copy.deepcopy(tmp_dic["graph"])
        now_graph.ndata["node_raw_feat"] = node_feature[tmp_dic["now_all_node"]]
        now_all_edge_feat = edge_feature[tmp_dic["now_all_edge"]]
        now_graph.edata["edge_raw_feat"] = torch.cat([now_all_edge_feat, now_all_edge_feat, torch.zeros(len(tmp_dic["now_all_node"]), edge_feature.shape[-1])], dim=0)     ###undirected
        now_all_edge_t = timestamp_feature[tmp_dic["now_all_edge"]]
        now_graph.edata["t"] = torch.cat([now_all_edge_t, now_all_edge_t, torch.zeros(len(tmp_dic["now_all_node"]), timestamp_feature.shape[-1])], dim=0)
        return now_graph, tmp_dic["new_src_local"], tmp_dic["new_dst_local"], tmp_dic["now_dst_node_local"], tmp_dic["new_src_node_label"], tmp_dic["induct_edge_lis_bool"], now_time
    else:
        return None, None, None, None, None, None, None

neg_sampling_ratio = args.neg_sampling_ratio


gnn_model = model_module.Model(num_layers=args.n_layer, n_head=args.n_head, node_dim=node_feature.shape[-1], d_k=node_feature.shape[-1]*args.d_hidden_ratio//args.n_head, d_v=node_feature.shape[-1]*args.d_hidden_ratio//args.n_head, d_T=node_feature.shape[-1], args=args, edge_dim=node_feature.shape[-1], device=device, dropout=args.drop_out)

link_classifier = MergeLayer(node_feature.shape[-1], node_feature.shape[-1], node_feature.shape[-1], 1)
optimizer = torch.optim.AdamW(list(gnn_model.parameters())+list(link_classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.neg_sampling_ratio])).to(device)
gnn_model.to(device)
link_classifier.to(device)



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

###Link Prediction
##Build Graph Based on the Current Time
epoch_num = args.n_epoch

best_val_ap = 0
best_val_ap_epoch = 0
best_val_acc = 0
best_val_acc_epoch = 0

def np_sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def run_model(start_timestamp, end_timestamp, state):
    global best_val_acc, best_val_acc_epoch, best_val_ap, best_val_ap_epoch
    tmp_tbatch_num = int((end_timestamp-start_timestamp)/tbatch_timespan)+1
    print_freq = tmp_tbatch_num // 3 + 1
    start_index = 0
    if state == "Train":
        is_train = True
        edge_lis = train_edge
        if "small" not in args.data: ##not in writing code mode
            start_index = 5 ##do not train with very first graph
        else:
            start_index = 1
    else:
        is_train = False
        edge_lis = linkage_df
    tbatch_index = list(range(start_index, tmp_tbatch_num+1))
    if args.shuffle:
        random.shuffle(tbatch_index)
    start_time = time.time()
    if is_train:
        gnn_model.train()
        link_classifier.train()
    else:
        gnn_model.eval()
        link_classifier.eval()
    
    loss_lis = []
    label_lis = []
    pred_score_lis = []
    induct_pred_score_lis = []
    induct_label_lis = []
    transduct_pred_score_lis = []
    transduct_label_lis = []
    for index, time_index in enumerate(tbatch_index):
        #print(start_timestamp, end_timestamp, time_index, time_index * tbatch_timespan)
        now_graph, new_src_node_local, new_dst_node_local, now_dst_node_local, new_src_node_label, induct_edge_lis_bool, t_now = build_temporal_graph_cache(node_feature, edge_lis, time_index, start_timestamp, tbatch_timespan, state)
        if now_graph is None:
            continue
        #now_graph.to(torch.device(device))
        with torch.no_grad():
            ##dim: num_of_new_edge
            neg_dst_node = np.array(np.array([random.sample(now_dst_node_local-set([new_dst_node_local[i]]), neg_sampling_ratio) for i in range(len(new_dst_node_local))]))
            sample_src_node = np.repeat(new_src_node_local, neg_sampling_ratio+1, axis=-1).ravel()
            sample_dst_node = np.concatenate([np.expand_dims(new_dst_node_local, axis=-1), neg_dst_node], axis=-1).ravel()
            label = torch.Tensor(([1]+[0]*neg_sampling_ratio)*new_src_node_local.shape[0]).to(device)
        with torch.set_grad_enabled(is_train):
            #node_embedding = gnn_model(now_graph, t_now)
            for nf in dgl.contrib.sampling.NeighborSampler(g=now_graph, batch_size=int(label.shape[0]*2), expand_factor=args.expand_factor, neighbor_type='in', shuffle=False, num_hops=args.n_layer, seed_nodes=torch.Tensor(np.concatenate([sample_src_node, sample_dst_node], axis=-1)).type(torch.int64), num_workers=1, add_self_loop=True):
                nf.copy_from_parent(ctx=torch.device(device))
                optimizer.zero_grad()
                node_embedding = gnn_model(nf, t_now)[nf.map_from_parent_nid(layer_id=args.n_layer, parent_nids=torch.Tensor(np.concatenate([sample_src_node, sample_dst_node], axis=-1)).type(torch.int64), remap_local=True),...]
                prob = link_classifier(node_embedding[:node_embedding.shape[0]//2,...], node_embedding[node_embedding.shape[0]//2:,...]).squeeze()
                loss = criterion(prob, label)/args.neg_sampling_ratio
                if is_train:
                    loss.backward()
                    optimizer.step()
                    torch.nn.utils.clip_grad_norm_(list(gnn_model.parameters())+list(link_classifier.parameters()), args.clip)
                    optimizer.zero_grad()
                torch.cuda.empty_cache()

                with torch.no_grad():
                    label = label.cpu().detach().numpy()
                    pred_score = prob.cpu().detach().numpy()
                    label_lis.append(label)
                    pred_score_lis.append(pred_score)
                    loss_lis.append(loss.item() * label.shape[0])
                    if state != "Train":
                        induct_edge_lis_bool = np.repeat(induct_edge_lis_bool, args.neg_sampling_ratio+1)
                        trasduct_score = pred_score[~induct_edge_lis_bool]
                        if trasduct_score.shape[0] != 0:
                            transduct_pred_score_lis.append(trasduct_score)
                            transduct_label_lis.append(label[~induct_edge_lis_bool])
                            induct_score = pred_score[induct_edge_lis_bool]
                            induct_label_lis.append(label[induct_edge_lis_bool])
                            induct_pred_score_lis.append(induct_score)
        del now_graph
            
    label_lis =  np.concatenate(label_lis).astype(bool)
    pred_score_lis = np_sigmoid(np.concatenate(pred_score_lis))
    pred_label_lis = (pred_score_lis > 0.5)
    overall_ap = average_precision_score(label_lis, pred_score_lis)
    overall_acc = (pred_label_lis==label_lis).mean()
    overall_loss = sum(loss_lis)/label_lis.shape[0]
    overall_f1 = f1_score(label_lis, pred_label_lis)
    overall_auc = roc_auc_score(label_lis, pred_score_lis)
    print_text = state +' Epoch: [{0}][{1}/{2}][{3}], AP {overall_ap:.4f}, Acc {overall_acc:.4f}, Loss {overall_loss:.2e}, Auc {overall_auc:.4f}, F1 {overall_f1:.4f}'.format(epoch, tmp_tbatch_num, tmp_tbatch_num, label_lis.shape[0], overall_ap=overall_ap, overall_acc=overall_acc, overall_loss=overall_loss, overall_auc=overall_auc, overall_f1=overall_f1)
    
    if state != "Train":
        induct_label_lis = np.concatenate(induct_label_lis)
        induct_pred_score_lis =  np_sigmoid(np.concatenate(induct_pred_score_lis))
        indcut_pred_label_lis = induct_pred_score_lis > 0.5
        induct_ap = average_precision_score(induct_label_lis, induct_pred_score_lis)
        induct_acc = (indcut_pred_label_lis==induct_label_lis).mean()
        
        transduct_label_lis =   np.concatenate(transduct_label_lis)
        transduct_pred_score_lis =  np_sigmoid(np.concatenate(transduct_pred_score_lis))
        indcut_pred_label_lis = transduct_pred_score_lis > 0.5
        transduct_ap = average_precision_score(transduct_label_lis, transduct_pred_score_lis)
        transduct_acc = (indcut_pred_label_lis==transduct_label_lis).mean()
    
    if state == "Val":
        if best_val_acc < overall_acc:
            best_val_acc = overall_acc
            best_val_acc_epoch = epoch
        if best_val_ap <  overall_ap:
            best_val_ap =  overall_ap
            best_val_ap_epoch = epoch
        print_text =  "**** " + print_text + ", Tran_Acc {transduct_acc:.4f}, Tran_AP {transduct_ap:.4f}, Indu_Acc {induct_acc:.4f}, Indu_AP {induct_ap:.4f}\n".format(transduct_acc=transduct_acc, transduct_ap=transduct_ap, induct_acc=induct_acc, induct_ap=induct_ap) + "Best Val: Acc {1:.4f} (Epoch {2}), AP {3:.4f} (Epoch {4})".format(epoch, best_val_acc, best_val_acc_epoch, best_val_ap, best_val_ap_epoch)
    if state == "Test":
        print_text = "!!!! " + print_text + ", Tran_Acc {transduct_acc:.4f}, Tran_AP {transduct_ap:.4f}, Indu_Acc {induct_acc:.4f}, Indu_AP {induct_ap:.4f}".format(transduct_acc=transduct_acc, transduct_ap=transduct_ap, induct_acc=induct_acc, induct_ap=induct_ap)
    print(print_text)
    print("Time:", time.time()-start_time)

for epoch in range(1, args.n_epoch+1):
    #Train
    run_model(start_timestamp = entire_start_timestamp, end_timestamp = val_start_timetamp, state = "Train")
    if epoch % args.val_interval == 0:
        run_model(start_timestamp = val_start_timetamp, end_timestamp = test_start_timetamp, state = "Val")
    if epoch % args.test_interval == 0:
        run_model(start_timestamp = test_start_timetamp, end_timestamp=entire_end_timestamp, state ="Test")
    if epoch % args.snapshot_interval == 0:
        print("Epoch %d Save Model"%(epoch))
        file_path1 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_gnn.model")
        file_path2 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_link_cls.model")
        torch.save(gnn_model.state_dict(), file_path1)
        torch.save(link_classifier.state_dict(), file_path2)
