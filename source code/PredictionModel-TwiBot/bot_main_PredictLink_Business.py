from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

import math
import networkx as nx
import random
import pickle
import numpy as np
import scipy.sparse as sp

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import time,datetime
import pandas
import itertools
from os.path import join as pjoin

from sklearn.manifold import TSNE
import time
import argparse
import numpy as np

import os
###############utils,model,data_loader################
from utils import initialize_S_A, initialize_Graph_matrix, date_timestamp_switch, loss_ploting, result_ploting, create_node_list, find_nearest_eventtime,timestamp_date_switch
from model import DyRep
from data_loader import build_ini_feature, load_data_from_file, test_data_split, load_dicts
######################################################
def parse_args_pool():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--fig_save_dir', type=str, default='./Output_file/fig2/Business/')
    parser.add_argument('--rst_save_dir', type=str, default='./Output_file/rst2/Business/')
    # parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--BATCH_SIZE', type=int, default=300, help='Number of batch size in each iteration.')
    parser.add_argument('--MAX_ITER', type=int, default=184, help='Maximum iterations in a training set.')
    parser.add_argument('--N_SURVIVE_SAMPLES', type=int, default=5, help='Number of surviving samples in calculating Loss.')
    parser.add_argument('--GRADIENT_CLIP', type=int, default=100, help='To prevent gradient explosion.')
    parser.add_argument('--PREDICT_N', type=int, default=200, help='Number of samples in calculating conditional density to predict Time.')
    parser.add_argument('--node_num', type=int, default=3016, help='Number of nodes.')
    parser.add_argument('--embed_size', type=int, default=128, help='Number of nodes.')
    parser.add_argument('--cluster', type=str, default='Business', help='cluster type')
    

    return parser.parse_args()

def train(epoch, model, optimizer, train_data, Lambda_time, ite_arr, los, args, node_time):
    begin_date = datetime.datetime(2020, 7, 1)
    begin_date = date_timestamp_switch(begin_date)
    cur_index = 0
    with torch.autograd.set_detect_anomaly(True):
        for ite in range(args.MAX_ITER):
            if(ite == args.MAX_ITER - 1):
                train_data_batch = train_data[cur_index: ]
            else:
                train_data_batch = train_data[cur_index: cur_index + args.BATCH_SIZE]
                cur_index = cur_index + args.BATCH_SIZE
            
            node_list = create_node_list(train_data_batch)

            losses_events = torch.tensor([0.],requires_grad=True).float().to(device)
            losses_nonevents = torch.tensor([0.],requires_grad=True).float().to(device)
                 
            for i in train_data_batch: 
                model.train()
                optimizer.zero_grad() 
#                 print("当前事件: ",i)
                lambda_t, L_surv, node_time = model(i, node_list, begin_date, args.N_SURVIVE_SAMPLES, node_time)
                
                
                Lambda_time[i[0]][i[1]][0] = lambda_t
                Lambda_time[i[0]][i[1]][1] = i[2]
                Lambda_time[i[1]][i[0]][0] = lambda_t
                Lambda_time[i[1]][i[0]][1] = i[2]
                lambda_t = lambda_t.to(device)
                L_surv = L_surv.to(device)
                
                losses_events = losses_events -torch.log(lambda_t).float()
                losses_nonevents = (losses_nonevents + L_surv).float()

            loss_train = losses_events + losses_nonevents
#             loss_train = loss_train.to(device)
            loss_train.backward()
#             loss_train = losses_nonevents
#             loss_train = loss_train.to(device)
#             loss_train.backward()
            
            nn.utils.clip_grad_value_(model.parameters(), args.GRADIENT_CLIP)
            optimizer.step()

            model.f = model.f.detach()  # to reset the computational graph and avoid backpropagating second time
            model.S = model.S.detach()

            los.append(loss_train.item())
            ite_arr.append(ite)
            print('Epoch: {:04d}'.format(epoch+1),
                  'Iteration: {:04d}'.format(ite),
                  'loss_train: {:.4f}'.format(loss_train.item()))

        loss_ploting(ite_arr, los, args.fig_save_dir)
        return node_time, Lambda_time

def test(test_data_haveslot,model,Lambda_time,ite_test_arr, MAR_arr, HITS_10_arr, MAE_arr, args, node_time, test_timeslots):
    csvFile = open(args.rst_save_dir + 'testrst.csv', "w")            #创建csv文件
    writer = csv.writer(csvFile)                  #创建写的对象
    #先写入columns_name                             
    writer.writerow(["Slot num","idx_u","idx_v(ground_truth)","Current Time", "True time","Predict time","Conditional_Density","Ranking","Ranked_No1_idx"])     #写入列的名称
    
    ######获取身份字典，human or robot？########
    dicts = load_dicts(args.cluster)
    
    ######为了可视化###### 
    dict_of_set = {}
    dict_of_node = {}
    for i in range(args.node_num):
        dict_of_node[i] = []
    set_num = 0
    
    begin_date = datetime.datetime(2020, 7, 1)
    begin_date = date_timestamp_switch(begin_date)
    concatlab = []
    node_appeared = []
    end_time = date_timestamp_switch(datetime.datetime(2020, 9, 1, 0, 0, 0))/3600

    for ith_slot in range(len(test_data_haveslot)): 
        # slot_time = test_timeslots[ith_slot]
        # slot_time_stamp = date_timestamp_switch(slot_time)
        slot_time_stamp = test_timeslots[ith_slot]
        slot_size = len(test_data_haveslot[ith_slot])
        iteration_counts = int(slot_size/args.BATCH_SIZE) + 1
        node_list_slot=[]
        for event in test_data_haveslot[ith_slot]:
            if(event[0] not in node_list_slot): 
                node_list_slot.append(event[0])
            if(event[1] not in node_list_slot): 
                node_list_slot.append(event[1])
        print("node list slot:",len(node_list_slot))
        
        MAR_slot = 0
        HITS_10_slot = 0
        cur_index = 0
        MAE_slot = np.array([0],dtype = np.float32)
        for ite in range(iteration_counts):
            if(ite == iteration_counts - 1):
                test_data_batch = test_data_haveslot[ith_slot][cur_index: ]
            else:
                test_data_batch = test_data_haveslot[ith_slot][cur_index:cur_index+args.BATCH_SIZE]                
                cur_index = cur_index+args.BATCH_SIZE

            node_list = create_node_list(test_data_batch)

            losses_events = torch.tensor(0)
            losses_nonevents = torch.tensor(0)                       
#             for name, parameters in model.named_parameters():
#                 print(name, ':', parameters)
            MAR_part = 0
            HITS10_son = 0
            MAE_part = np.array([0],dtype = np.float32);
            useless = 0
            for i in test_data_batch: 
                model.eval()
                u = i[0]
                v = i[1]
                lambda_t = model.intensity(model.f[u], model.f[v], i[3])
                t_max = find_nearest_eventtime(Lambda_time,u,v)

                ###################### count conditional density - MAR HITS@10##########################
                delta_t = i[2]-Lambda_time[i[0]][i[1]][1]
                delta_t_sec = delta_t/3600
                # condi_density = lambda_t * torch.exp(-(lambda_t + Lambda_time[i[0]][i[1]][0])*(delta_t_sec)/2)
                condi_density = lambda_t * torch.exp(-(lambda_t)*(delta_t_sec))
                v_condi = condi_density.item()

                for other in node_list_slot:
                    if(other in node_appeared):
                        continue
                    lambda_t_other = model.intensity(model.f[u], model.f[other], i[3])
                    # t_max_o = find_nearest_eventtime(Lambda_time,u,other)
                    delta_t = i[2] - Lambda_time[i[0]][other][1]
                    delta_t_sec = delta_t/3600
                    condi_density_other = lambda_t * torch.exp(-(lambda_t_other)*(delta_t_sec))
                    condi_density = torch.cat((condi_density,condi_density_other), dim = 0)
                
                ranking = torch.sort(condi_density, dim=0, descending=True)[1]
                rankno = -1
                for each in range(ranking.shape[0]):
                    if ranking[each].item() == 0:
                        MAR_part = MAR_part + each
                        rankno = each
                        if(each<=9):
                            HITS10_son = HITS10_son+1
                        break
                
                ########################################################################################
                

                optimizer.zero_grad() 
                lambda_t, L_surv, node_time = model(i, node_list, begin_date, args.N_SURVIVE_SAMPLES, node_time)
                
                Lambda_time[i[0]][i[1]][0] = lambda_t
                Lambda_time[i[0]][i[1]][1] = i[2]
                Lambda_time[i[1]][i[0]][0] = lambda_t
                Lambda_time[i[1]][i[0]][1] = i[2]
                losses_events = losses_events - torch.log(lambda_t)
                losses_nonevents = losses_nonevents + L_surv
                if(u not in node_appeared):
                    node_appeared.append(u)
                if(v not in node_appeared):
                    node_appeared.append(v)
                
                ########################################## MAE ##########################################                
                t_next = 0.0     
                ##i[2]表示目前的时间，i[5]表示未来将要发生的时间
                cur_time = i[2]/3600
                t_truth = i[4]/3600
                
                if(i[4] == date_timestamp_switch(datetime.datetime(2020, 9, 1, 0, 0, 0))):
                    useless = useless+1
                    continue
                
#                 L_surv = model.Survival(u, v, model.f, node_list)
                
                f_poss = 0.0
                for k in range(args.PREDICT_N):
                    t_rand = np.random.uniform(cur_time, end_time)
#                     Lambda_pred = model.intensity(model.f[u], model.f[v], i[3])
#                     Surviv_pred = model.Survival(u, v, model.f, node_list)
#                     f_pred = Lambda_pred * torch.exp(-Surviv_pred)
                    delta_t_stamp = int(t_rand-cur_time)
                    f_pred = lambda_t * torch.exp(-(lambda_t)*(delta_t_stamp))
                    f_pred = f_pred.cpu()
                    f_pred = f_pred.data.numpy()
                    t_next += t_rand * f_pred
                    f_poss += f_pred
                #####为什么是curr + 预测到的值呢？######
#                 t_predict = cur_time + (t_next / PREDICT_N)
                if(f_poss == 0):
                    t_predict = cur_time
                else:
                    t_predict = t_next/f_poss 
#                 t_predict = cur_time + (t_next / args.PREDICT_N)
                t_truth_np = np.array(t_truth, np.float)
                t_predict_np = np.array(t_predict, np.float)
                MAE_part = MAE_part + abs(t_truth_np - t_predict_np)
                ########################################################################################## 
                if(abs(t_truth_np - t_predict_np)<=10):          
                    with open(args.rst_save_dir + 'testrst.csv','a+') as f:
                        csv_write = csv.writer(f)
                        data_row = [ith_slot+1,u,v, i[2] , i[4] , t_predict[0]*3600 ,v_condi,rankno,ranking[0].item()]
                        csv_write.writerow(data_row) 

                    d = {'id':v,'time':t_predict}       
                    if(dict_of_node.get(u) == []):
                        d2 = {'id':u,'time':t_predict}
                        set_num+=1
                        dict_of_set[set_num] = [d,d2] 
                        dict_of_node[u].append(set_num)
                        dict_of_node[v].append(set_num)

                    else:
                        set_ids = dict_of_node.get(u)
                        for setid in set_ids:
                            dict_of_set[setid].append(d)
                            if(setid not in dict_of_node[v]):
                                dict_of_node[v].append(setid)
            loss_test = losses_events + losses_nonevents
            MAR = MAR_part/len(test_data_batch)
            HITS_10 = HITS10_son/len(test_data_batch)
            if((len(test_data_batch)-useless)!=0):
                MAE = MAE_part/(len(test_data_batch)-useless)
            else:
                MAE = np.array([0],dtype = np.float32);
           
            MAR_slot = MAR_slot + MAR
            HITS_10_slot = HITS_10_slot + HITS_10
            MAE_slot = MAE_slot + MAE

            model.f = model.f.detach()  # to reset the computational graph and avoid backpropagating second time
            model.S = model.S.detach()

        
            print('Epoch: {:04d}'.format(epoch+1),
                  'Iteration: {:04d}'.format(ite),
                  'loss_test: {:.4f}'.format(loss_test.item()),
                 'MAR: {:.4f}'.format(MAR),
                 'HITS@10: {:.4f}'.format(HITS_10),
                 'MAE: {:.4f}'.format(MAE[0]))
        
        MAR_slot = MAR_slot/iteration_counts
        HITS_10_slot = HITS_10_slot/iteration_counts
        MAE_slot = MAE_slot/iteration_counts

        print('Epoch: {:04d}'.format(epoch+1),
              'Time Slot: {:04d}'.format(ith_slot),
             'MAR: {:.4f}'.format(MAR_slot),
             'HITS@10: {:.4f}'.format(HITS_10_slot),
             'MAE: {:.4f}'.format(MAE_slot[0]))

        MAR_arr.append(MAR_slot)
        HITS_10_arr.append(HITS_10_slot)        
        MAE_arr.append(MAE_slot)

        ite_test_arr.append(ith_slot+1)     
        node_appeared.clear()
    result_ploting(ite_test_arr, MAR_arr, HITS_10_arr, MAE_arr, args.fig_save_dir)

    with open(args.rst_save_dir + 'Business_dict_of_set' + '.pkl', 'wb') as f:
        pickle.dump(dict_of_set, f, pickle.HIGHEST_PROTOCOL)
    with open(args.rst_save_dir + 'Business_dict_of_node' + '.pkl', 'wb') as f:
        pickle.dump(dict_of_node, f, pickle.HIGHEST_PROTOCOL)
    
    return node_time, Lambda_time 

if __name__ == '__main__':
    args = parse_args_pool()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    MAR_arr = []
    HITS_10_arr = []
    MAE_arr = []
    los = []
    ite_arr = []
    ite_test_arr = []
    
    
    
    feature = build_ini_feature(args.node_num)
    initial_data, train_data, test_data = load_data_from_file(args.cluster)
    
    ###if want to switch to small run, edit here!############
#     train_data = train_data[0:600]
    test_timeslots, test_data_haveslot = test_data_split( test_data )
    G, Lambda_time, node_time = initialize_Graph_matrix( initial_data, args.node_num )
    A0, S0 = initialize_S_A( G , args.node_num )

    A = torch.tensor(A0,dtype=torch.float32)
    A = A.to(device)
    S = torch.tensor(S0,dtype=torch.float32)
    S = S.to(device)
    
    model = DyRep(graph = G,
                feature = feature,
                embed_size = args.embed_size,
                initial_embedding = feature,
                S_initial = S,
                A_initial = A
                )
    params_main, params_enc = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
                params_main.append(param)
    optimizer = optim.Adam(params_main,
                            lr = args.lr,
                            weight_decay=args.weight_decay)
    model.to(device)

    for epoch in range(args.epochs):
        log_dir = './dyrep_Business_para.pkl'
        model_dir = './Business_model.pkl'
        if os.path.exists(log_dir) and os.path.exists(model_dir):
            model=torch.load(model_dir)
            checkpoint = torch.load(log_dir)            
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.f = checkpoint['feature']
            node_time = checkpoint['node_time']
            Lambda_time = checkpoint['Lambda_time']
            print('Loading epoch {} succeed！'.format(epoch))
#             print(model.S == S )
        else:
            print('No saved model, training from beginning! ')
            node_time, Lambda_time = train(epoch, model, optimizer, train_data, Lambda_time, ite_arr, los, args, node_time)
            state = {'optimizer':optimizer.state_dict(), 'feature':model.f, 'node_time':node_time, 'Lambda_time':Lambda_time}
            torch.save(state, log_dir)
            torch.save(model, model_dir)
            print("Optimization Finished!")
    node_time, Lambda_time = test(test_data_haveslot,model,Lambda_time,ite_test_arr, MAR_arr, HITS_10_arr, MAE_arr, args, node_time, test_timeslots)

    MAR_arr.clear()
    MAE_arr.clear()
    HITS_10_arr.clear()
    los.clear()
    ite_arr.clear()
    ite_test_arr.clear()

    torch.cuda.empty_cache()