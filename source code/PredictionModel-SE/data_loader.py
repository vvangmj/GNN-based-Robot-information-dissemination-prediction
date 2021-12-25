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

def SubjectsReader(csv_path):
    csv = pandas.read_csv(csv_path)
    subjects = csv[list(filter(lambda column: column.find('user') >= 0, list(csv.keys())))[0]].tolist()
    print('Number of subjects', len(subjects))
    features = []
    for column in list(csv.keys()):
        if column.find('user') >= 0:
            continue
        values = list(map(str, csv[column].tolist()))
        features_unique = np.unique(values)
        features_onehot = np.zeros((len(subjects), len(features_unique)))

        for subj, feat in enumerate(values):
            ind = np.where(features_unique == feat)[0]
            assert len(ind) == 1, (ind, features_unique, feat, type(feat))
            features_onehot[subj, ind[0]] = 1
        features.append(features_onehot)

    features_onehot = np.concatenate(features, axis=1)

    return features_onehot

def build_ini_feature():
    data = SubjectsReader(pjoin('SocialEvolution/', 'Subjects.csv'))

    feature = torch.tensor(data)
    feature = feature.to(device)
    feature = feature.to(torch.float32)

    pud0 = torch.zeros((1,15),dtype=torch.float32)
    pud0 = pud0.to(device)
    pudf = torch.zeros((15,15),dtype=torch.float32)
    pudf = pudf.to(device)

    feature = torch.cat((pud0, feature), dim = 0)
    feature = torch.cat((feature, pudf), dim = 0)
    dim=(0,17,0,0)
    feature=F.pad(feature,dim,"constant",value=0)

    return feature

def load_data_from_file():
    path='./processed_data/'
    with open(path+'closeFriend_initial','rb') as f:
        initial_data = pickle.load(f)
    with open(path+'train_data','rb') as f:
        train_data = pickle.load(f)
    with open(path+'preprocessed_test_data','rb') as f:
        test_data = pickle.load(f)

    initial_data=sorted(initial_data, key=lambda x:x[2])

    return initial_data, train_data, test_data

def test_data_split(test_data):
    test_timeslots = [datetime.datetime(2009, 5, 10), \
                    datetime.datetime(2009, 5, 20), \
                    datetime.datetime(2009, 5, 31), \
                    datetime.datetime(2009, 6, 10), \
                    datetime.datetime(2009, 6, 20), \
                    datetime.datetime(2009, 6, 30)]
    
    timeslot_0 = []
    timeslot_1 = []
    timeslot_2 = []
    timeslot_3 = []
    timeslot_4 = []
    timeslot_5 = []

    test_data_haveslot = []

    for i in range(len(test_data)):
        if(test_data[i][2]<=test_timeslots[0]):
            timeslot_0.append(test_data[i])
        elif(test_data[i][2]>test_timeslots[0] and test_data[i][2]<=test_timeslots[1]):
            timeslot_1.append(test_data[i])
        elif(test_data[i][2]>test_timeslots[1] and test_data[i][2]<=test_timeslots[2]):
            timeslot_2.append(test_data[i])
        elif(test_data[i][2]>test_timeslots[2] and test_data[i][2]<=test_timeslots[3]):
            timeslot_3.append(test_data[i])
        elif(test_data[i][2]>test_timeslots[3] and test_data[i][2]<=test_timeslots[4]):
            timeslot_4.append(test_data[i])
        else:
            timeslot_5.append(test_data[i])

    test_data_haveslot.append(timeslot_0)
    test_data_haveslot.append(timeslot_1)
    test_data_haveslot.append(timeslot_2)
    test_data_haveslot.append(timeslot_3)
    test_data_haveslot.append(timeslot_4)
    test_data_haveslot.append(timeslot_5)

    return test_timeslots, test_data_haveslot

    