""" Training and testing of the model
"""
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter
import matplotlib.pyplot as plt
from SupConLoss import SupConLoss
import numpy as np
import pandas as pd
cuda = True if torch.cuda.is_available() else False
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score  
sns.set_style("whitegrid")
from sklearn.cluster import KMeans  


def augment_data(x_batch, noise_scale=0.03):
    noise = torch.randn_like(x_batch) * noise_scale
    return x_batch + noise
def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    # 读取的都是标签，然后转化成数字

    data_tr_list = []
    data_te_list = []
    # 先创建数据的列表

    for i in view_list:
        # 读取训练数据并处理非数值内容
        data_tr = np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=',', dtype=str)
        data_tr = np.where(data_tr == '', '0', data_tr)  # 将空字符串替换为 '0'
        data_tr = data_tr.astype(np.float64)
        data_tr_list.append(data_tr)

        # 读取测试数据并处理非数值内容
        data_te = np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=',', dtype=str)
        data_te = np.where(data_te == '', '0', data_te)  # 将空字符串替换为 '0'
        data_te = data_te.astype(np.float64)
        data_te_list.append(data_te)
    # 读取三个组学的数据

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    # 得到总共的数量
    print("训练的样本数num_tr:", num_tr)
    print("测试的样本数num_te:", num_te)
    #print(data_tr_list)
    #print(data_te_list)

    data_mat_list = []  # 存储数据的pytorch张量
    for i in range(num_view):  # 对于每一个视图，都将其拼接到一起
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):  # 把每个都转换成pytorch张量
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()  # 放到GPU
    # 字典中前tr存储的是训练样本的索引，te是测试
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))

    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                        data_tensor_list[i][idx_dict["te"]].clone()), 0))
    labels = np.concatenate((labels_tr, labels_te))
    

    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))

    return adj_train_list, adj_test_list


def train_epoch(data_list,  label, one_hot_label, sample_weight, model_dict, optim_dict, train_OCDN=True,tem=0.07,Cratio=0.5,noise=0.01):
    #print(tem)
    loss_dict = {}
    contrastive_criterion = SupConLoss(temperature=tem)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)
    elist=[]
    for i in range(num_view) :

        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0

        # 第一次前向传播
        e = model_dict["E{:}".format(i + 1)](augment_data(data_list[i],noise))
        #elist.append(e)
        ci = model_dict["C{:}".format(i + 1)](e)


        # 计算交叉熵损失
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
        total_loss=ci_loss


        contrastive_loss = contrastive_criterion(e, labels=label)

        total_loss =2*((1-Cratio)*ci_loss+ Cratio*contrastive_loss)



        total_loss.backward()#反向传播
        optim_dict["C{:}".format(i+1)].step()#更新参数
        loss_dict["C{:}".format(i+1)] = total_loss.detach().cpu().numpy().item()

    
    if train_OCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 1

        ci_list = []
        ei_list=[]
        for i in range(num_view):

            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i])))
            ei_list.append(model_dict["E{:}".format(i+1)](data_list[i]))

        c = model_dict["C"](ci_list)
        c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))

        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()

    return loss_dict




def test_epoch(data_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    ei_list=[]
    for i in range(num_view):

        ci_list.append(model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i])))
        ei_list.append(model_dict["E{:}".format(i + 1)](data_list[i]))

    # 保存每个视图的预测结果
    individual_probs = []

    for i in range(num_view):#专门为了2个组学的改动
        
        prob = F.softmax(ci_list[i][te_idx, :], dim=1).data.cpu().numpy()
        individual_probs.append(prob)



    
    
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]


    c = c[te_idx,:]
    fused_prob = F.softmax(c, dim=1).data.cpu().numpy()

    return fused_prob, individual_probs

def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c,
               num_epoch_pretrain, num_epoch,dropout, data_tr_list, data_trte_list, trte_idx, labels_trte,tem,Cratio=0.5,alpha=4,dr=0.1,noise=0.01):
    test_inverval = 1#每50轮测试一次
    num_view = len(view_list)
    dim_hOCDN = pow(num_class,num_view)#用作OCDN的输入的维，即numclass的num_view次方
    adj_parameter = 10
    dim_he_list = [500,400,200]



    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    #所有的训练的标签torch张量

    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)


    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    #print("sample_weight_tr:",sample_weight_tr)


    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()

    dim_list = [x.shape[1] for x in data_tr_list]#存储各个视图的特征数量


    #换成新的多头注意力的初始化
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hOCDN, dropout=dropout, dr=dr)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    print(model_dict)

    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list,  labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_OCDN=False,tem=tem,Cratio=Cratio,noise=noise)

    print("\nTraining...")

    maxViewAcc = 0

    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)

    maxF1=0
    maxF1_macro=0
    maxAuc=0
    for epoch in range(num_epoch+1):
        loss_dict = train_epoch(data_tr_list,  labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict,tem=tem,Cratio=Cratio)

        if epoch % test_inverval == 0:
            fused_prob, individual_probs = test_epoch(data_trte_list, trte_idx["te"], model_dict)
            print("\nTest: Epoch {:d}".format(epoch))

            # 计算融合模块的 ACC
            fused_acc = accuracy_score(labels_trte[trte_idx["te"]], fused_prob.argmax(1))
            print("Fused Test ACC: {:.3f}".format(fused_acc))

            # 计算每个视图的 ACC

            veiewaveAcc=0
            for i in range(num_view-1):
                individual_acc = accuracy_score(labels_trte[trte_idx["te"]], individual_probs[i].argmax(1))
                veiewaveAcc = individual_acc+veiewaveAcc
                #print("View {:d} Test ACC: {:.3f}".format(i + 1, individual_acc))
            veiewaveAcc=veiewaveAcc/num_view
            print("veiewaveAcc=: {:.3f}".format(veiewaveAcc))

            #print("\nTest: Epoch {:d}".format(epoch))

            if num_class == 2:
                fused_acc = accuracy_score(labels_trte[trte_idx["te"]], fused_prob.argmax(1))
                fused_f1=f1_score(labels_trte[trte_idx["te"]],  fused_prob.argmax(1))
                fused_auc=roc_auc_score(labels_trte[trte_idx["te"]],  fused_prob[:,1])
                
                print("Test ACC: {:.3f}".format(fused_acc))
                print("Test F1: {:.3f}".format(fused_f1))
                print("Test AUC: {:.3f}".format(fused_auc))
                print()

            else:
                fused_acc=accuracy_score(labels_trte[trte_idx["te"]],  fused_prob.argmax(1))
                fused_f1=f1_score(labels_trte[trte_idx["te"]],  fused_prob.argmax(1), average='weighted')
                fusedf1_macro=f1_score(labels_trte[trte_idx["te"]], fused_prob.argmax(1), average='macro')
                
                
                print("Test ACC: {:.3f}".format( fused_acc))
                print("Test F1 weighted: {:.3f}".format(fused_f1))
                print("Test F1 macro: {:.3f}".format( fusedf1_macro))



            

        