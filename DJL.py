"""
Main function for algorithm DJL

"""
from sklearn.neural_network import MLPRegressor 
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pandas as pd

class DJL(object):

    def __init__(self, policy_evaluate, policy_behavior, environment, m, mlp_max_iter=100, envir_type='simu'):
        """
        :param policy_evaluate: policy to be evaluated.
        :param policy_behavior: behavior policy to generate data from environment.
        :param environment: environment class which is equipped with sample() method to return observed value.
        :param m: number of initial candidate intervals.
        :param mlp_max_iter: maximum number of iteration in multilayer perceptrons.
        :param envir_type: the environment, choosen from 'simu' for simulated data and 'real' for calibrated real data.
        """
        self.policy_behavior = policy_behavior
        self.policy_evaluate = policy_evaluate
        self.environment = environment
        self.envir_type = envir_type
        self.m = m
        self.mlp_max_iter = mlp_max_iter
         
        self.context_observe = []
        self.action_taken = []
        self.reward_received = []
        self.time = 1
        
        return
    
                    
    def initialize_training(self, train_data):
        """
        Initialize training process
        """   
        self.bellm = np.zeros(self.m) 
        self.tau = [] # change point location
        self.R_set = [] # candidate points set
 
        self.Cost_Dict = {}
        self.Q_nn = {}
        self.train_data = train_data
        return
        
    def get_cost(self, l, r, seed=1):
        """
        Collect the cost function.
        :return: the cost
        """
        if self.Cost_Dict.get(str(l) + ':' + str(r)) == None:
            if l == r:
                self.Cost_Dict[str(l) + ':' + str(r)] = 0
                
            else: 
                subdata = self.train_data[(self.train_data['at'] >= l / self.m) & (self.train_data['at'] <= r / self.m)]
                if len(subdata) == 0:
                    self.Cost_Dict[str(l) + ':' + str(r)] = 0
                else:
                    regr = MLPRegressor(hidden_layer_sizes=(10,10), random_state=seed, max_iter=self.mlp_max_iter).fit(np.array([x for x in subdata['xt']]), subdata['yt'])
                    y_fit = regr.predict(np.array([x for x in subdata['xt']]))
                    self.Cost_Dict[str(l) + ':' + str(r)] = sum((y_fit - subdata['yt']) ** 2)
                    self.Q_nn[str(l) + ':' + str(r)] = regr
           
        return self.Cost_Dict.get(str(l) + ':' + str(r))
    
    
    def get_prop_score(self, l, r, seed=1, act_method='logistic'):
        """
        Calculate the propensity score function for each interval.
        :return: the propensity score function
        """
        self.train_data[str(l) + ':' + str(r)] = 1 * ((self.train_data['at'] >= l / self.m) & (self.train_data['at'] <= r / self.m))
        regr = MLPRegressor(hidden_layer_sizes=(10,10), random_state=seed, max_iter= self.mlp_max_iter, activation=act_method).fit(np.array([x for x in self.train_data['xt']]), self.train_data[str(l) + ':' + str(r)])
        
        return regr
    
    
    def get_partition(self):
        """
        Apply the Pruned Exact Linear Time method
        :return: the partitions
        """
        self.R_set.append([-1])
        self.tau.append([]) 
        for v_star in range(self.m):
    
            bel_cost_list = []
            for v in self.R_set[v_star]:
                bel = - self.gamma if v == -1 else self.bellm[v]
                cost = self.get_cost(v+1, v_star+1)
                bel_cost_list.append(bel + cost + self.gamma)

            self.bellm[v_star] = np.min(np.array(bel_cost_list))
            
            v1 = self.R_set[v_star][bel_cost_list.index(self.bellm[v_star])]
            
            self.tau.append(sorted(list(set(self.tau[v1 + 1] + [v1])))) 
            
            new_R_set = []
            for v in [ * self.R_set[v_star], v_star]:
                bel = - self.gamma if v == -1 else self.bellm[v]
                cost = self.get_cost(v+1, v_star+1)
                if bel + cost <= self.bellm[v_star]:
                    new_R_set.append(v) 

            self.R_set.append(new_R_set)
        
        return np.array(self.tau[-1]) + 1
    
    def least_square_loss(self, tau, test_data, seed=1):
        """
        Use the left k-fold to calculate the least square loss function.
        :return: the Estimated Value
        """    
        self.test_data = test_data
        ls_loss = 0
        for i in range(len(tau)):
            l = tau[i] 
            r = tau[i + 1] if i < len(tau) - 1 else self.m

            subdata = self.test_data[(self.test_data['at'] >= l / self.m) & 
                                    (self.test_data['at'] < r / self.m)] if i < len(tau) - 1 else self.test_data[(self.test_data['at'] >= l / self.m) & (self.test_data['at'] <= r / self.m)]
            
            if len(subdata) > 0:
                    
                fitted_Q = self.Q_nn[str(l) + ':' + str(r)].predict(np.array([x for x in subdata['xt']]))   
                ls_loss += sum((subdata['yt'] - fitted_Q) ** 2) 
        return ls_loss 
    
    
    def evaluate(self, tau, test_data, seed=1):
        """
        Value Evaluation
        :return: the Estimated Value
        """    
        self.test_data = test_data
        V_hat = 0
        for i in range(len(tau)):
            l = tau[i] 
            r = tau[i + 1] if i < len(tau) - 1 else self.m
            #print('Processing interval: (', l / self.m, ',', r / self.m, ')...')

            subdata = self.test_data[(self.test_data['at'] >= l / self.m) & 
                                    (self.test_data['at'] < r / self.m)] if i < len(tau) - 1 else self.test_data[(self.test_data['at'] >= l / self.m) & (self.test_data['at'] <= r / self.m)]
                
            if len(subdata) > 0:
                prop_score = self.get_prop_score(l, r)
                
                if prop_score == 1:
                    prob_fit = 1
                else:
                    prob_fit = prop_score.predict(np.array([x for x in subdata['xt']]))   
                prob_fit = np.minimum(np.maximum(prob_fit, len(subdata['yt']) / len(self.test_data['yt'])), 1.)
                #print('fitted behavior prob: ', prob_fit)
            
                pi_star_ind = np.array([(self.policy_evaluate(x) >= l / self.m) * (self.policy_evaluate(x) < r / self.m) for x in subdata['xt']]) if i < len(tau) - 1 else np.array([(self.policy_evaluate(x) >= l / self.m) * (self.policy_evaluate(x) <= r / self.m) for x in subdata['xt']])
                    
                fitted_Q = self.Q_nn[str(l) + ':' + str(r)].predict(np.array([x for x in subdata['xt']]))  
                #print('fitted Q: ', fitted_Q)
                V_hat += sum(pi_star_ind / prob_fit * (subdata['yt'] - fitted_Q) + fitted_Q)
                #print('diff: ', (subdata['yt'] - fitted_Q))
        
        V_hat = V_hat / len(self.test_data['at'])
        #print('Estimated Value: ', V_hat)
        
        return V_hat

 
    def sample(self):
        """
        sample data from environment based on behavior policy.
        :return:
        """
        if self.envir_type == 'simu':
            context = self.environment.get_context()
            action = self.policy_behavior(context)
            reward = self.environment.get_reward(context, action)
        elif self.envir_type == 'real':
            context, action, reward = self.environment.get_sample(self.policy_behavior) 
            
        self.context_observe.append(context)
        self.action_taken.append(action)
        self.reward_received.append(reward)
        # increment time
        self.time += 1
        
    def get_dataset(self, n):
        """
        generate data from environment based on behavior policy.
        :return: the dataset
        """
        for i in range(n):
            self.sample()
            
        dataset = pd.DataFrame(columns=['xt', 'at', 'yt'])
        dataset['xt'] = self.context_observe
        dataset['at'] = self.action_taken
        dataset['yt'] = self.reward_received
        
        return dataset
