import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import math
import networkx as nx
import random
import pickle
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import csv
import time,datetime
import pandas
import itertools
from os.path import join as pjoin

def initialize_S_A( G , node_num ):
    A0=np.array(nx.adjacency_matrix(G).todense())
    S0=np.zeros(shape=(node_num, node_num), dtype=float)
    for row in range(node_num):
        for col in range(node_num):
            if (A0[row][col] == 1):
                S0[row][col] = float(1.0/len(list(G.neighbors(col))))
    
    return A0, S0

def initialize_Graph_matrix(initial_data):
    G = nx.MultiGraph()
    G.add_nodes_from(range(0,100))
    for row in initial_data:
        G.add_edge(row[0],row[1],time = row[2],conn = row[3])

    begin_date = datetime.datetime(2008, 9, 11)
    Lambda_time = [[[0 for c in range(2)]for r in range(100)]for d in range(100)]
    for i in range(100):
        for j in range(100):
            Lambda_time[i][j][1] = begin_date

    node_time = np.empty((100), dtype = datetime.datetime)
    for i in initial_data:
        node_time[i[0]] = i[2]
        node_time[i[1]] = i[2]

    return G, Lambda_time, node_time

def date_timestamp_switch(node_time):
    prev_t = node_time.strftime("%Y-%m-%d %H:%M:%S")
    timeArray = time.strptime(prev_t, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(timeArray)
    return timestamp

def timestamp_date_switch(node_timestamp_np):
    localtime = time.localtime(node_timestamp_np.item()*3600)
    datatime = time.strftime("%Y:%m:%d %H:%M:%S",localtime)
    return datatime

def loss_ploting(ite_arr, los, path):
    plt.figure()
    plt.plot(ite_arr, los, 'r-', lw=2)
    plt.title('Loss',fontsize=24)
    plt.xlabel('iteration',fontsize=14)
    plt.ylabel('Loss',fontsize=14)
    plt.savefig(path+'Loss.png')
    # plt.show()

def result_ploting(ite_test_arr, MAR_arr, HITS_10_arr, MAE_arr, path):
    plt.figure()
    plt.plot(ite_test_arr, MAR_arr, 'b-', lw=2)
    plt.title('MAR',fontsize=24)
    plt.xlabel('Time Slot',fontsize=14)
    plt.ylabel('MAR',fontsize=14)
    # plt.xlim(-5,100)
    # plt.ylim(0.8,2.0)
    plt.savefig(path+'MAR.png')
    # plt.show()

    plt.figure()
    plt.plot(ite_test_arr, HITS_10_arr, 'g-', lw=2)
    plt.title('HITS@10',fontsize=24)
    plt.xlabel('Time Slot',fontsize=14)
    plt.ylabel('HITS@10',fontsize=14)
    # plt.xlim(-5,100)
    # plt.ylim(0.8,2.0)
    plt.savefig(path+'HITS10.png')
    # plt.show()
    
    plt.figure()
    plt.plot(ite_test_arr, MAE_arr, 'g-', lw=2)
    plt.title('MAE',fontsize=24)
    plt.xlabel('Time Slot',fontsize=14)
    plt.ylabel('MAE',fontsize=14)
    # lt.xlim(-5,100)
    # plt.ylim(0.8,2.0)
    plt.savefig(path+'MAE.png')
    # plt.show()

def create_node_list(train_data_batch):
    node_list = []
    for event in train_data_batch:
        if(event[0] not in node_list): 
            node_list.append(event[0])
        if(event[1] not in node_list): 
            node_list.append(event[1])
    return node_list

def find_nearest_eventtime(Lambda_time,u,v):
    t_max = Lambda_time[u][0][1]
    for idx in range(100):
        if(t_max< Lambda_time[u][idx][1]):
            t_max = Lambda_time[u][idx][1]
        if(t_max< Lambda_time[v][idx][1]):
            t_max = Lambda_time[v][idx][1]
    return t_max

