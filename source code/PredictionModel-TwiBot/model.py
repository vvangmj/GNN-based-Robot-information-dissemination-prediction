import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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

class DyRep(nn.Module):
    def __init__(self, graph, feature, embed_size,initial_embedding,S_initial, A_initial, bias = True):
        super(DyRep, self).__init__()
        self.g = graph
        self.embed_size = embed_size
        self.W_h = nn.Linear(embed_size, embed_size,bias=True)
        self.W_struct = nn.Linear(embed_size, embed_size)
        self.W_rec = nn.Linear(embed_size,embed_size)
        self.omega_0 = nn.Linear(2 * embed_size,1)
        self.omega_1 = nn.Linear(2 * embed_size,1)
        self.W_t = nn.Linear(1, embed_size)        
        self.psi = nn.Parameter(0.5 * torch.ones(2))
        self.f = feature
        self.S = S_initial
        self.A = A_initial

    def update_S_A(self, event, graph, lambda_t):
        with torch.no_grad():
            S_matrix = self.S.clone()
            A_matrix = self.A.clone()
            if(event[3] == 1 and A_matrix[event[0]][event[1]] == 0):
                return self.A, self.S

            i = event[1]
            j = event[0]

            if(len(list(self.g.neighbors(j))) == 0):
                b_p_j = torch.tensor([0.]).float().to(device)
            else:
                b_p_j = torch.tensor([1/len(list(self.g.neighbors(j)))]).float().to(device)
            if(len(list(self.g.neighbors(i))) == 0):
                b_p_i = torch.tensor([0.]).float().to(device)
            else:
                b_p_i = torch.tensor([1/len(list(self.g.neighbors(i)))]).float().to(device) 

            if(event[3] == 0):
                self.g.add_edge(event[0],event[1],time = event[2],conn = event[3])

            if(len(list(self.g.neighbors(j))) == 0):
                b_j = torch.tensor([0.]).float().to(device)
            else:
                b_j = torch.tensor([1/len(list(self.g.neighbors(j)))]).float().to(device) 
            if(len(list(self.g.neighbors(i))) == 0):
                b_i = torch.tensor([0.]).float().to(device)
            else:
                b_i = torch.tensor([1/len(list(self.g.neighbors(i)))]).float().to(device)

            y = S_matrix[j].clone()       
            if(event[3] == 1 and A_matrix[event[0]][event[1]] >= 1):
                y[i] = b_j + lambda_t
            elif(event[3] == 0 and A_matrix[event[0]][event[1]] == 0):
                x = b_p_j - b_j
                y[i] = b_j + lambda_t
                for s in range(self.embed_size):
                    if(s!=i and y[s]!=0 ):
                        y[s] = y[s] - x
            S_matrix[j] = y

            y = S_matrix[i].clone()
            if(event[3] == 1 and A_matrix[event[0]][event[1]] >= 1):
                y[j] = b_i + lambda_t
            elif(event[3] == 0 and A_matrix[event[0]][event[1]] == 0):
                x = b_p_i - b_i
                y[j] = b_i + lambda_t
                for s in range(self.embed_size):
                    if(s!=j and y[s]!=0):
                        y[s] = y[s] - x            
            S_matrix[i] = y

            if(event[3] == 0):
                A_matrix[event[0]][event[1]] = 1
                A_matrix[event[1]][event[0]] = 1

            self.A = A_matrix
            self.S = S_matrix

    def intensity(self, repre_u, repre_v, event):
#         with torch.no_grad(): 
        if(event == 0):
            psi = self.psi[0]
            cat_0 = torch.cat((repre_u, repre_v),0)
            # print("cat_0",cat_0.size())
            g = self.omega_0(cat_0)
            f = psi * torch.log(1+torch.exp(g/psi))

        elif(event == 1):
            psi = self.psi[1]
            cat_1 = torch.cat((repre_u, repre_v),0)
            # print("cat_1",cat_1.size())
            g = self.omega_1(cat_1)
            f = psi * torch.log(1+torch.exp(g/psi))
        
        return f

    def Survival(self, u, v, node_embed, node_list,N_survive):
#         with torch.no_grad(): 
        u_surv = torch.tensor([0.]).float().to(device)
        v_surv = torch.tensor([0.]).float().to(device)

        for serv in range(N_survive):  
            others = [0,0]    
            for i in range(2):      
                random.shuffle(node_list)
                idx = 0
                while(node_list[idx] == u or node_list[idx] == v):
                    idx=idx+1
                    if(idx==len(node_list)-1):
                        break
                others[i] = node_list[idx]

            lambda_surv_u_0= DyRep.intensity(self, node_embed[u], node_embed[others[1]], 0).to(device)
            lambda_surv_v_0= DyRep.intensity(self, node_embed[others[0]], node_embed[v], 0).to(device)
            lambda_surv_u_1= DyRep.intensity(self, node_embed[u], node_embed[others[1]], 1).to(device)
            lambda_surv_v_1= DyRep.intensity(self, node_embed[others[0]], node_embed[v], 1).to(device)

            u_surv = u_surv+lambda_surv_u_0+lambda_surv_u_1
            v_surv = v_surv+lambda_surv_v_0+lambda_surv_v_1        
        N_survive = torch.tensor(N_survive).to(device)
        L_surv = (u_surv + v_surv)/N_survive   

        return L_surv
    def time_process(self, cur_date, begin_date):
        delta_t_ori = str(cur_date - begin_date)
        delta_t = delta_t_ori.split(',')
        if(delta_t == [delta_t_ori]):
            delta_t = ['0']+delta_t[0].split(':')
            dt = torch.tensor([float(delta_t[0]),float(delta_t[1]),float(delta_t[2]),float(delta_t[3])])*0.1            
        else:
            delta_t = delta_t[0].split(' ')+delta_t[1].split(' ')[1].split(':')
            dt = torch.tensor([float(delta_t[0]),float(delta_t[2]),float(delta_t[3]),float(delta_t[4])])*0.1
        
        return dt

    def forward(self, event, node_list,begin_date,N_survive, node_time):
#         with torch.no_grad(): 
        v = event[1]
        u = event[0]        
        node_embed = self.f
#         print("????????????")
#         print("??????node embedding: ",node_embed)
        z_new = node_embed.clone()
        feature_u = node_embed[u]
        feature_v = node_embed[v]
        lambda_t = torch.tensor([0.],requires_grad=True).float().to(device)
        L_surv = torch.tensor([0.],requires_grad=True).float().to(device)

    # print(event)
#         with torch.no_grad():
        lambda_t = DyRep.intensity(self, feature_u, feature_v, event[3])
        L_surv = DyRep.Survival(self, u, v, node_embed, node_list, N_survive)
#         with torch.no_grad():
        if(len(list(self.g.neighbors(u))) == 0):
            h_u_struct = node_embed[u]      
            # dt_u = self.time_process(event[2], begin_date)    
            dt_u =  (event[2] - begin_date)/3600
            dt_u = torch.tensor([dt_u])
            dt_u = dt_u.to(device) 

        if(len(list(self.g.neighbors(v))) == 0):
            h_v_struct = node_embed[v]    
            # dt_v = self.time_process(event[2], begin_date)     
            dt_v =  (event[2] - begin_date)/3600
            dt_v = torch.tensor([dt_v])
            dt_v = dt_v.to(device)        

        if(len(list(self.g.neighbors(u))) != 0):
        #????????????z_v
            neighbor_u = list(self.g.neighbors(u))
            sum_q_ui=torch.tensor(0).to(device)
            #q_ui
            a = np.zeros((len(neighbor_u),1), dtype=np.float32)
            q_u = torch.tensor(a)
            q_u = q_u.to(device)
            idx=0
            for i in neighbor_u:
                sum_q_ui = sum_q_ui + torch.exp(self.S[u][i])
            for i in neighbor_u:
                q_ui = torch.exp(self.S[u][i])/sum_q_ui
                q_u[idx]= q_ui
                idx = idx + 1

            h_i = self.W_h(node_embed[neighbor_u])
            q_h = torch.sigmoid(q_u * h_i)
            h_u_struct = torch.max(q_h,dim=0)[0]

            if(node_time[u] == None):
                node_time[u] = begin_date
            # dt_u = self.time_process(event[2], node_time[u])     
            dt_u = (event[2] - node_time[u])/3600
            dt_u = torch.tensor([dt_u])
            dt_u = dt_u.to(device)

        if(len(list(self.g.neighbors(v))) != 0):
            neighbor_v = list(self.g.neighbors(v))
            sum_q_vi=torch.tensor(0).to(device)
            a = np.zeros((len(neighbor_v),1), dtype=np.float32)
            q_v = torch.tensor(a)
            q_v = q_v.to(device)
            idx=0
            for i in neighbor_v:
                sum_q_vi = sum_q_vi + torch.exp(self.S[v][i])

            for i in neighbor_v:
                q_vi = torch.exp(self.S[v][i])/sum_q_vi
                q_v[idx]= q_vi
                idx = idx + 1

            h_i = self.W_h(node_embed[neighbor_v])
            q_h_v = torch.sigmoid(q_v * h_i).to(device)
            h_v_struct = torch.max(q_h_v,dim=0)[0]

            if(node_time[v] == None):
                node_time[v] = begin_date
            # dt_v = self.time_process(event[2], node_time[v])     
            dt_v = (event[2] - node_time[v])/3600
            dt_v = torch.tensor([dt_v])
            dt_v = dt_v.to(device)
#         with torch.no_grad(): 
        dt_u = dt_u.to(torch.float32)
        dt_v = dt_v.to(torch.float32)
        h1_v = self.W_struct(h_u_struct)
        h2_v = self.W_rec(feature_v)
        h3_v = self.W_t(dt_v)

        h1_u = self.W_struct(h_v_struct)
        h2_u = self.W_rec(feature_u)
        h3_u = self.W_t(dt_u)  

        self.update_S_A(event, self.g , lambda_t)

        node_time[u] = event[2]
        node_time[v] = event[2]

        z_new[u] = torch.sigmoid(h1_u + h2_u + h3_u)
        z_new[v] = torch.sigmoid(h1_v + h2_v + h3_v)
        self.f = z_new 
            #####################################
            #fake
# #             dt_u = dt_u.to(torch.float32)
# #             dt_v = dt_v.to(torch.float32)
#             h1_v = self.W_struct(h_u_struct)
#             h2_v = self.W_rec(feature_v)
# #             h3_v = self.W_t(dt_v)

#             h1_u = self.W_struct(h_v_struct)
#             h2_u = self.W_rec(feature_u)
# #             h3_u = self.W_t(dt_u)  

# #             self.update_S_A(event, self.g , lambda_t)

#             node_time[u] = event[2]
#             node_time[v] = event[2]

#             z_new[u] = torch.sigmoid(h1_u + h2_u)
#             z_new[v] = torch.sigmoid(h1_v + h2_v)
# #             self.f = z_new 
            #######################################
      
        return lambda_t, L_surv, node_time