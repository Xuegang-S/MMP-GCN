import os
import copy
import time
import csv
import numpy as np
import torch.nn as nn
import math
import torch
import argparse
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from scipy.io import loadmat
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
root_folder3 = "E:\\code-new\\population-gcn-master\\fmri-dti-291+guangxi\\"
phenotype3 = os.path.join(root_folder3, 'Phenototal1.csv')
class GraphConvolution1(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class GraphConvolution1_sxg(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution1_sxg, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class GCN_multi_channle(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_multi_channle, self).__init__()

        self.gc1 = GraphConvolution1_sxg(nfeat, nhid)
        self.gc2 = GraphConvolution1(5*nhid, nclass)
        self.dropout = dropout
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, adj):
        adj = self.conv1(adj)[0, 0]
        x1 = F.relu(self.gc1(x[:,0:40], adj))  # relu
        x2 = F.relu(self.gc1(x[:,40:80], adj))  # relu
        x3 = F.relu(self.gc1(x[:,80:120], adj))  # relu
        x4 = F.relu(self.gc1(x[:,120:160], adj))  # relu
        x5 = F.relu(self.gc1(x[:,160:200], adj))  # relu
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
def get_keys(d, value):
    return [k for k, v in d.items() if v == value]
def get_ids(num_subjects=None, sub=''):
    subject_IDs = np.genfromtxt(os.path.join(root_folder3, sub), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype3) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['Subject'] in subject_list:
                scores_dict[row['Subject']] = row[score]

    return scores_dict
def get_networks(data_folder_FC, subject_list):
    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder_FC, subject + '.mat')
        matrix0 = loadmat(fl)
        matrix = matrix0['brainNet']
        all_networks.append(matrix)
    idx = np.triu_indices_from(all_networks[0], 1)
    vec_networks = [mat[idx] for mat in all_networks]
    matrix = np.vstack(vec_networks)
    return matrix
def create_graph_from_phenotypic_information(scores, subject_list):

    num_nodes = len(subject_list)
    graph = np.zeros((num_nodes, num_nodes))
    for l in scores:
        label_dict = get_subject_score(subject_list, l)
        if l in ['Sex']:  # we run the part
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[subject_list[k % len(subject_list)]] == label_dict[
                        subject_list[j % len(subject_list)]]:
                        graph[k, j] += 1
                        graph[j, k] += 1
        if l in ['SITE_ID']:  # we  run the part
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[subject_list[k % len(subject_list)]] == label_dict[
                        subject_list[j % len(subject_list)]]:
                        graph[k, j] += 1
                        graph[j, k] += 1
        if l in ['ADNI']:  # we  run the part
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[subject_list[k % len(subject_list)]] == label_dict[
                        subject_list[j % len(subject_list)]]:
                        graph[k, j] += 1
                        graph[j, k] += 1
        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k % len(subject_list)]]) - float(
                            label_dict[subject_list[j % len(subject_list)]]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass
    return graph
def feature_selection(matrix, labels, train_ind, fnum):
    estimator = RidgeClassifier()
    selector = RFE(estimator, fnum, step=10, verbose=0)   # step 表示每次删除的特征个数
    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)
    feature_index = selector.get_support(True)
    return x_data, feature_index
def feature_selection_statistics(matrix, labels, train_ind, fnum):

    featureX = matrix[train_ind, :]
    mean0 = np.mean(featureX, axis=0)
    featureX = featureX - mean0
    featureY = labels[train_ind]
    train_ind1 = list()
    train_ind2 = list()
    for i in range(len(featureY)):
        if featureY[i] == 1:
            train_ind1.append(i)
        if featureY[i] == 2:
            train_ind2.append(i)
    featureX1 = featureX[train_ind1, :]
    featureX2 = featureX[train_ind2, :]
    featureX1_mean = np.mean(featureX1,axis=0)
    featureX2_mean = np.mean(featureX2,axis=0)
    featureX1_std = np.std(featureX1, axis = 0)
    featureX2_std = np.std(featureX2, axis = 0)
    feature_std = 0.5 * featureX1_std + 0.5 * featureX2_std
    results = abs(featureX1_mean-featureX2_mean) / (0.1 + 2 * feature_std)
    ranks = np.argsort(-results)  ### -results 表示降序
    ranks_top = ranks[0:fnum,]
    x_data = matrix[:,ranks_top]
    return x_data
def bulid_graph_shuangzhi(fea):
    def f(x):
        return x * x
    u = fea.shape[0]
    v = fea.shape[1]
    adj = np.zeros([fea.shape[0], fea.shape[0]])
    b = np.zeros([u, v])
    c = np.zeros([u])
    for i in range(u):
        a = 0.
        d = 0.
        for j in range(v):
            a = a + fea[i][j]
        a = a/v
        fea[i] = fea[i] - a
        b[i] = list(map(f, list(fea[i])))
        for p in range(v):
            d = d + b[i][p]
        c[i] = np.sqrt(d)
    for k in range(u):
        for s in range(u):

            if c[k]*c[s] == 0.:
                adj[k][s] = 0.
            else:
                adj[k][s] = np.dot(fea[k], fea[s])/c[k]*c[s] # 相似性矩阵s
    return adj  # MCI-graph
def graph_pool(train_ind,final_graph1,num_retain,y1) :

    adj_class1 = np.zeros((len(y1), 1))
    adj_class2 = np.zeros((len(y1), 1))
    num1 = 0
    num2 = 0
    for k in range(0, len(y1)) :
        for l in range(0, len(y1)) :
            if y1[l] == 1 and l in train_ind :
                adj_class1[k] = adj_class1[k] + final_graph1[k,l]
                num1 = num1 + 1
            if y1[l] == 2 and l in train_ind:
                adj_class2[k] = adj_class2[k] + final_graph1[k,l]
                num2 = num2 + 1
    num1 = num1 / len(y1)
    num2 = num2 / len(y1)
    adj_class1 = adj_class1 / num1
    adj_class2 = adj_class2 / num2
    adj_class1_2 = abs(adj_class1 - adj_class2)
    ranks = np.argsort(-adj_class1_2[:,0])  ### -results 表示降序， 返回索引
    ranks_top = ranks[0:num_retain, ]  ### 保留前 rr 个索引
    for i0 in range(0, len(y1)):
        if i0 not in ranks_top:
            final_graph1[:, i0] = 0
    return final_graph1, ranks_top
def train_model(model, fts, final_graph, lbls, train_ind, val_ind,
                criterion, optimizer, scheduler, device,
                num_epochs=500, print_freq=500):
    since = time.time()

    model_wts_best_val_acc = copy.deepcopy(model.cpu().state_dict())
    model_wts_lowest_val_loss = copy.deepcopy(model.cpu().state_dict())
    model = model.to(device)
    best_acc = 0.0
    loss_min = 100

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            idx = train_ind if phase == 'train' else val_ind
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts,final_graph)
                soft_max = F.softmax(outputs)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(soft_max, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])
            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)
            if epoch % (print_freq-1) == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_wts_best_val_acc = copy.deepcopy(model.cpu().state_dict())
                model = model.to(device)

            if phase == 'val' and epoch_loss < loss_min:
                loss_min = epoch_loss
                model_wts_lowest_val_loss = copy.deepcopy(model.cpu().state_dict())
                model = model.to(device)

            if epoch % print_freq == 0 and phase == 'val':
                print(f'Best val Acc: {best_acc:4f}, Min val loss: {loss_min:4f}')
                print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    return model_wts_best_val_acc, model_wts_lowest_val_loss, best_acc, loss_min
def test_model(model, best_model_wts, fts,final_graph, lbls, test_ind, device):
    model.load_state_dict(best_model_wts)
    model = model.to(device)
    model.eval()

    running_corrects = 0.0
    with torch.set_grad_enabled(False):
        outputs = model(fts,final_graph)
        soft_max = F.softmax(outputs)
        _, preds = torch.max(soft_max, 1)
        score_pred = soft_max[:, 1]

    running_corrects += torch.sum(preds[test_ind] == lbls.data[test_ind])
    test_acc = running_corrects.double() / len(test_ind)
    pred = preds[test_ind]
    lab = lbls.data[test_ind]
    score_pred = score_pred[test_ind]

    return test_acc, pred, lab, score_pred
def train_test_multi_channle_sparse(feature1,graph1,graph2,graph3,y_data1,y1,train_ind,test_ind,jk,xishu):
    ######################
    num_nodes = feature1.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i in train_ind and j in train_ind:
                if y1[i] == y1[j] :
                    graph1[i, j] = graph1[i, j] * xishu
                    graph2[i, j] = graph2[i, j] * xishu
                    graph3[i, j] = graph3[i, j] * xishu

    feature2, feature_index1 = feature_selection(feature1, y_data1[:,1], train_ind, 200)
    x_data1_rank = feature_selection_statistics(feature2, y1, train_ind, len(feature2[1, :]))

    labels = y_data1[:,1]
    graph_adj = bulid_graph_shuangzhi(feature1)
    num_l = 1-0.05*jk
    num_retain = round(num_l * len(y1))
    graph_adj, adj_rank_top = graph_pool(train_ind, graph_adj, num_retain, y1)

    train_size = int(0.9 * len(train_ind))
    valid_size = len(train_ind) - train_size
    train_ind, valid_ind = torch.utils.data.random_split(train_ind, [train_size, valid_size])
    train_ind = torch.LongTensor(train_ind).to(device)
    valid_ind = torch.LongTensor(valid_ind).to(device)
    test_ind = torch.LongTensor(test_ind).to(device)
##################
    x_data1_rank_1 = x_data1_rank[:, 0:40]
    x_data1_rank_2 = x_data1_rank[:, 0:80]
    x_data1_rank_3 = x_data1_rank[:, 0:120]
    x_data1_rank_4 = x_data1_rank[:, 0:160]
    x_data1_rank_5 = x_data1_rank[:, 0:200]

    final_graph1 = bulid_graph_shuangzhi(x_data1_rank_1)
    final_graph1_new1 = graph_adj * final_graph1
    final_graph2 = bulid_graph_shuangzhi(x_data1_rank_2)
    final_graph1_new2 = graph_adj * final_graph2
    final_graph3 = bulid_graph_shuangzhi(x_data1_rank_3)
    final_graph1_new3 = graph_adj * final_graph3
    final_graph4 = bulid_graph_shuangzhi(x_data1_rank_4)
    final_graph1_new4 = graph_adj * final_graph4
    final_graph5 = bulid_graph_shuangzhi(x_data1_rank_5)
    final_graph1_new5 = graph_adj * final_graph5

    for k in range(len(y1)):
        final_graph1_new5[k, :] = final_graph1_new5[k, :] / (0.1+np.sum(final_graph1_new5[k, :]))
        final_graph1_new4[k, :] = final_graph1_new4[k, :] / (0.1+np.sum(final_graph1_new4[k, :]))
        final_graph1_new3[k, :] = final_graph1_new3[k, :] / (0.1+np.sum(final_graph1_new3[k, :]))
        final_graph1_new2[k, :] = final_graph1_new2[k, :] / (0.1+np.sum(final_graph1_new2[k, :]))
        final_graph1_new1[k, :] = final_graph1_new1[k, :] / (0.1+np.sum(final_graph1_new1[k, :]))
    x_data_sxg1 = np.dot(final_graph1_new1, x_data1_rank_1)
    x_data_sxg2 = np.dot(final_graph1_new2, x_data1_rank_2)
    x_data_sxg3 = np.dot(final_graph1_new3, x_data1_rank_3)
    x_data_sxg4 = np.dot(final_graph1_new4, x_data1_rank_4)
    x_data_sxg5 = np.dot(final_graph1_new5, x_data1_rank_5)

    fea_sxg = np.hstack((x_data_sxg1, x_data_sxg2[:, 40:80], x_data_sxg3[:, 80:120], x_data_sxg4[:, 120:160],
                            x_data_sxg5[:, 160:200]))

    fts = torch.FloatTensor(fea_sxg).to(device) # Convert to tensor and pass to device
    lbls = torch.LongTensor(labels).to(device) # Squeeze along axis that are 1 and convert the values to 64 bit integers and pass to device
    graph_adj0 = graph_adj
    graph_adj1 = graph1 * graph_adj
    graph_adj2 = graph2 * graph_adj
    graph_adj3 = graph3 * graph_adj
    graph_adj = [graph_adj0,graph_adj1,graph_adj2,graph_adj3]
    final_graph = torch.FloatTensor([graph_adj]).to(device)

##################

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nclass', type=int, default=2,
                        help='number of classes.')
    parser.add_argument('--milestones', type=list, default=[100],
                        help='number .')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='number .')
    args = parser.parse_args()

    model = GCN_multi_channle(nfeat=fts[:,0:40].shape[1],
                nhid=args.hidden,
                nclass=args.nclass,
                dropout=args.dropout)
    print(model)

    # initialize model
    state_dict = model.state_dict()
    for key in state_dict:
        if 'weight' in key:
            nn.init.xavier_uniform_(state_dict[key])
        elif 'bias' in key:
            state_dict[key] = state_dict[key].zero_()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.95, weight_decay=cfg['weight_decay'])
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.milestones,
                                               gamma=args.gamma)
    # criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()

    model_wts_lowest_val_loss, model_wts_best_val_acc, best_acc, loss_min = train_model(model, fts, final_graph,lbls,
                                                                                       train_ind, valid_ind,
                                                                                       criterion, optimizer,
                                                                                       schedular, device,
                                                                                       num_epochs=1000, print_freq=500)

    if test_ind is not None:
        print('**** Model of lowest val loss ****')
        test_acc_lvl, pred_lvl, lab_lvl, score_pred_lvl = test_model(model, model_wts_lowest_val_loss, fts, final_graph,lbls, test_ind, device)
        print("test_acc_lvl: ", test_acc_lvl)
        print('**** Model of best val acc ****')
        test_acc_bva, pred_bva, lab_bva, score_pred_bva = test_model(model, model_wts_best_val_acc, fts, final_graph,lbls, test_ind, device)
        print("test_acc_bva: ", test_acc_bva)
        if test_acc_lvl >= test_acc_bva:
            return test_acc_lvl, pred_lvl, lab_lvl, score_pred_lvl
        else:
            return test_acc_bva, pred_bva, lab_bva, score_pred_bva

def _main():

    subject_IDs1 = get_ids(None, sub='fmri-dti.txt')
    labels1 = get_subject_score(subject_IDs1, score='Group')

    # ------SMC vs Normal--------
    Normal_IDs1 = get_keys(labels1, 'CN')
    SMC_IDs1 = get_keys(labels1, 'SMC')
    EMCI_IDs1 = get_keys(labels1, 'EMCI')
    LMCI_IDs1 = get_keys(labels1, 'LMCI') # + get_keys(labels1, 'MCI')
    total_Normal_SMC_IDs1 = Normal_IDs1 + SMC_IDs1

    for i in Normal_IDs1:
        labels1[i] = 1
    for j in SMC_IDs1:
        labels1[j] = 2
    num_classes = 2
    num_nodes = len(total_Normal_SMC_IDs1)
    y_data1 = np.zeros([num_nodes, num_classes])
    y1 = np.zeros([num_nodes, 1])
    for i in range(num_nodes):
        y_data1[i, int(labels1[total_Normal_SMC_IDs1[i]]) - 1] = 1
        y1[i] = int(labels1[total_Normal_SMC_IDs1[i]])
    sfolder = StratifiedKFold(n_splits=10, random_state=9, shuffle=True)

    for j0 in range(5, 6):
        jk = 3
        test_acc_ave = []
        test_pred = []
        test_lab = []
        test_score_pred = []
        data_folder_FC = "E:\\妮娜\\xuegangBrainNetwork-291\\new_sxg_wsr_dti\\NCSMC\\"
        root_folder = "L" + str(j0)
        data_folder = os.path.join(data_folder_FC, root_folder)
        feature1 = get_networks(data_folder, total_Normal_SMC_IDs1)

        graph1 = create_graph_from_phenotypic_information(['Sex'], total_Normal_SMC_IDs1)
        graph2 = create_graph_from_phenotypic_information(['SITE_ID'],total_Normal_SMC_IDs1)
        graph3 = create_graph_from_phenotypic_information(['ADNI'], total_Normal_SMC_IDs1)  # ADNI represents center info

        mn = 0
        for train_ind, test_ind in sfolder.split(np.zeros(len(y1)), y1):
            mn += 1
            print("--------------", mn, "------------")
            xishu = 1.5
            test_acc, pred, lab, score_pred = train_test_multi_channle_sparse(feature1,graph1,graph2,graph3,y_data1,y1,train_ind,test_ind,jk,xishu)
            test_acc_ave.append(test_acc)
            test_pred += pred
            test_lab += lab
            score_pred = score_pred.cpu().numpy()
            test_score_pred.append(score_pred)
    print("acc_total: ", )
    print("acc_total_mean: ", sum(test_acc_ave)/10)

if __name__ == '__main__':
    _main()

