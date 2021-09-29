import numpy as np
import pandas as pd
import os
import pickle
import random 
import time 
from datetime import datetime
import argparse

from multiprocessing import Pool
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import KFold

from tqdm import tqdm
from functools import partial 
 
import data_generator
import DJL

os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

#========================================
# Configurations
#========================================

# ----------- parameters ------------
parser.add_argument('--envir_type', type = str, default = 'simu1',
                    choices = ['simu1','simu2','simu3','simu4','real'],
                    help = 'Choosing which experiment to do.')
parser.add_argument('--real_data_file', type = str, default = 'real_envir.pickle',
                    help = 'The file containing the calibrated real data.') 
parser.add_argument('--sample_size', type = int, default = 300,
                    help = 'The number of samples of data.')
parser.add_argument('--feature_dim', type = int, default = 20,
                    help = 'The dimension of features.')
parser.add_argument('--seed', type = int, default = 2333, 
                    help = 'Random seed.')
parser.add_argument('--rep_number', type = int, default = 100, 
                    help = 'The number of replication.')  

args = parser.parse_args()
print(args)


#========================================
# Setup Environment 
#========================================
if args.envir_type == 'simu1':
    data_gen = data_generator.DataGeneratorScena1(mu=0, sigma=1, lb=-1., ub=1., p=args.feature_dim)
if args.envir_type == 'simu2':
    data_gen = data_generator.DataGeneratorScena2(mu=0, sigma=1, lb=-1., ub=1., p=args.feature_dim)
if args.envir_type == 'simu3':
    data_gen = data_generator.DataGeneratorScena3(mu=0, sigma=1, lb=-1., ub=1., p=args.feature_dim)
if args.envir_type == 'simu4':
    data_gen = data_generator.DataGeneratorScena4(mu=0, sigma=1, lb=-1., ub=1., p=args.feature_dim) 
if args.envir_type == 'real':
    data_gen = data_generator.RealDataGenerator(file_name=args.real_data_file) 

def pi_behavior(context, lb=0., ub=1.):
    return np.random.uniform(lb, ub, 1)[0] # randomized trial    
    
def pi_optimal(context):
    if args.envir_type == 'simu1':
        val_list = [(1 + context[0]), (context[0] - context[1]), (1 - context[1])]
        idx = val_list.index(np.max(np.array(val_list)))
        return np.random.uniform(0, 0.35, 1)[0] * (idx == 0) + np.random.uniform(0.35, 0.65, 1)[0] * (idx == 1) + np.random.uniform(0.65, 1., 1)[0] * (idx == 2)
    elif args.envir_type == 'simu2':
        val_list = [1., (np.sin(2 * np.pi * context[0])), (0.5 - 8 * (context[0] - 0.75) ** 2), 0.5]
        idx = val_list.index(np.max(np.array(val_list)))
        return np.random.uniform(0, 0.25, 1)[0] * (idx == 0) + np.random.uniform(0.25, 0.5, 1)[0] * (idx == 1) + np.random.uniform(0.5, .75, 1)[0] * (idx == 2) + np.random.uniform(0.75, 1., 1)[0] * (idx == 3)
    elif args.envir_type == 'simu3':
        return 1.0 
    elif args.envir_type == 'simu4':
        return 0.5 * (1 + 0.5 * context[0] + 0.5 * context[1])
    elif args.envir_type == 'real':
        act_list = np.linspace(0, 1, 100)
        x_max = np.max(np.array([data_gen.org_data.iloc[i]['xt'] for i in range(len(data_gen.org_data))]), 0)
        x_min = np.min(np.array([data_gen.org_data.iloc[i]['xt'] for i in range(len(data_gen.org_data))]), 0)
        val = []
        for act in act_list:
            val.append(data_gen.regr_mean.predict(np.append(np.array(context),np.array(act)).reshape(1, len(context)+1))[0])
        return act_list[val.index(max(val))]
    
#========================================
# Main function
#========================================
    
def DJL_exp(data_gen, seed):

    tstart = datetime.now()
    
    ### set seed 
    np.random.seed(seed)
    random.seed(seed)
    
    n = args.sample_size # total number of visit
    m = int(n / 10) # number of initial intervals
    num_fold = 5 # 5-fold cross-validation
    gamma_list = np.linspace(0.1, .5, 5) * n ** 0.4 # penalty term gamma_T
    sample_index = np.linspace(0, n-1, n) 
    pars_num = []
    
    if args.envir_type == 'real':
        agent = DJL.DJL(policy_evaluate=pi_optimal, policy_behavior='behavior', environment=data_gen, m=m, envir_type = 'real', mlp_max_iter=100)
    else:
        agent = DJL.DJL(policy_evaluate=pi_optimal, policy_behavior=pi_behavior, environment=data_gen, m=m, envir_type = 'simu', mlp_max_iter=50)  

    ### generate data based on behavior policy
    dataset = agent.get_dataset(n) 
    kf = KFold(n_splits=num_fold, random_state=seed, shuffle=True) 
    V_hat_list = []

    for train_index, test_index in kf.split(sample_index):
        data_train, data_test = dataset.iloc[train_index], dataset.iloc[test_index]

        loss_list = []
        for i in range(len(gamma_list)):
            gamma = gamma_list[i]
            agent.gamma = gamma
            loss = 0

            train_sample_index = np.linspace(0, len(data_train)-1, len(data_train))
            kf_inner = KFold(n_splits=num_fold, random_state=seed, shuffle=True)

            for inner_train_index, inner_test_index in kf_inner.split(train_sample_index):
                inner_data_train, inner_data_test = data_train.iloc[inner_train_index], data_train.iloc[inner_test_index]

                agent.initialize_training(inner_data_train)
                tau = agent.get_partition() 
                loss += agent.least_square_loss(tau, inner_data_test)

            loss_list.append(loss)

        ### Select the best tuning parameter by minimuming the least square loss function
        agent.gamma = gamma_list[loss_list.index(np.min(np.array(loss_list)))]
        #print('Select gamma: ', agent.gamma)

        agent.initialize_training(data_train)
        ### Apply the Pruned Exact Linear Time method to get the partitions
        tau = agent.get_partition() 
        pars_num.append(len(tau)) 

        ### Evaluation 
        V_hat = agent.evaluate(tau, data_test)
        V_hat_list.append(V_hat)
        #print('Estimated Value: ', V_hat) 
    
    tstop = datetime.now()
    speed = (tstop - tstart).seconds / 60
    
    print('Seed:', seed, 'Estimated Value: ', np.mean(np.array(V_hat_list)), 'Number of Partitions: ', np.mean(np.array(pars_num)), 'Time Spent (Minutes): ', speed)
    return np.mean(np.array(V_hat_list)), np.mean(np.array(pars_num)), speed
     
     
#========================================
# Save Results
#========================================
if args.rep_number == 1:
    result = DJL_exp(data_gen, args.seed)
    with open('DJL_Results_' + str(args.envir_type) + '_SampleSize' + str(args.sample_size) + '.pickle', 'wb') as filehandle:
        pickle.dump(result, filehandle) 
else:        
    np.random.seed(args.seed) # Random seed 
    seeds_list = np.random.randint(1, 1000000, size=args.rep_number)
    with Pool() as pool:
        results = list(tqdm(pool.imap(partial(DJL_exp, data_gen), seeds_list), total=args.rep_number))
    with open('DJL_Results_' + str(args.envir_type) + '_SampleSize' + str(args.sample_size) + '.pickle', 'wb') as filehandle:
        pickle.dump(results, filehandle)       
    
    # Summary
    if args.envir_type == 'simu1':
        mu = np.mean(abs(np.array([x[0] for x in results]) - 1.33))
    if args.envir_type == 'simu2':
        mu = np.mean(abs(np.array([x[0] for x in results]) - 1.0))
    if args.envir_type == 'simu3':
        mu = np.mean(abs(np.array([x[0] for x in results]) - 4.86))
    if args.envir_type == 'simu4':
        mu = np.mean(abs(np.array([x[0] for x in results]) - 1.60))
    if args.envir_type == 'real':
        mu = np.mean(abs(np.array([x[0] for x in results]) - (-0.278)))
 
    sigma = np.std(np.array([x[0] for x in results]))
    par_nums = np.mean(np.array([x[1] for x in results]))
    speed = np.mean(np.array([x[2] for x in results]))
    
    print('Sumary of DJL under Scenario : ' + str(args.envir_type) + ', Sample Size: ' + str(args.sample_size))
    print('Bias : ', np.abs(np.round(mu, 3)))
    print('Standard Deviation : ', np.round(sigma, 3))
    print('Number of Partitions : ', np.round(par_nums, 3))
    print('Time Spent (Minutes) : ', np.round(speed, 3))  
    