from abc import ABC
import numpy as np
import pickle
import random
 
class DataGenerator(ABC):
    def __init__(self, mu=0, sigma=1, lb=-1., ub=1., p=20):
        self._mu = mu
        self._sigma = sigma
        self._lb = lb
        self._ub = ub 
        self.p = p
        
    def get_context(self):
        return NotImplementedError
    
    def get_reward(self, context, action):
        return NotImplementedError
    
    def get_mean_reward(self, context, action):
        return NotImplementedError
    
    
class DataGeneratorScena1(DataGenerator):
    def __init__(self, mu=0, sigma=1, lb=-1., ub=1., p=20):
        super(DataGeneratorScena1, self).__init__(mu, sigma, lb, ub, p)
        
    def get_context(self):
        return np.random.uniform(self._lb, self._ub, self.p)
    
    def get_reward(self, context, action):
        return (action < 0.35) * (1 + context[0]) + (action >= 0.35) * (action <= 0.65) * (context[0] - context[1]) + (action > 0.65) * (1 - context[1]) + np.random.normal(self._mu, self._sigma, 1)[0] 
        
    def get_mean_reward(self, context, action):
        return (action < 0.35) * (1 + context[0]) + (action >= 0.35) * (action <= 0.65) * (context[0] - context[1]) + (action > 0.65) * (1 - context[1])
    
    
class DataGeneratorScena2(DataGenerator):
    def __init__(self, mu=0, sigma=1, lb=-1., ub=1., p=20):
        super(DataGeneratorScena2, self).__init__(mu, sigma, lb, ub, p)
        
    def get_context(self):
        return np.random.uniform(self._lb, self._ub, self.p)
    
    def get_reward(self, context, action):
        return (action < 0.25) + (action >= 0.25) * (action < 0.5) * (np.sin(2 * np.pi * context[0])) + (action >= 0.5) * (action < 0.75) * (0.5 - 8 * (context[0] - 0.75) ** 2) + 0.5 * (action >= 0.75) + np.random.normal(self._mu, self._sigma, 1)[0]

     
    def get_mean_reward(self, context, action):
        return (action < 0.25) + (action >= 0.25) * (action < 0.5) * (np.sin(2 * np.pi * context[0])) + (action >= 0.5) * (action < 0.75) * (0.5 - 8 * (context[0] - 0.75) ** 2) + 0.5 * (action >= 0.75)
     
    
class DataGeneratorScena3(DataGenerator):
    def __init__(self, mu=0, sigma=1, lb=-1., ub=1., p=20):
        super(DataGeneratorScena3, self).__init__(mu, sigma, lb, ub, p)
        
    def get_context(self):
        return np.random.uniform(self._lb, self._ub, self.p)
    
    def get_reward(self, context, action):
        return (action > 0.5) * 10 * np.log(context[0] + 2) * max(action ** 2 - 0.25, 0) + np.random.normal(self._mu, self._sigma, 1)[0]  
     
        
    def get_mean_reward(self, context, action):
        return (action > 0.5) * 10 * np.log(context[0] + 2) * max(action ** 2 - 0.25, 0)
      

class DataGeneratorScena4(DataGenerator):
    def __init__(self, mu=0, sigma=1, lb=-1., ub=1., p=20):
        super(DataGeneratorScena4, self).__init__(mu, sigma, lb, ub, p)
        
    def get_context(self):
        return np.random.uniform(self._lb, self._ub, self.p)
    
    def get_reward(self, context, action):
        return (8 + 4 * context[0] - 2 * context[1] - 2 * context[2] - 10 * (1 + 0.5 * context[0] + 0.5 * context[1] - 2 * action) ** 2) / 5 + np.random.normal(self._mu, self._sigma, 1)[0] 
        
    def get_mean_reward(self, context, action):
        return (8 + 4 * context[0] - 2 * context[1] - 2 * context[2] - 10 * (1 + 0.5 * context[0] + 0.5 * context[1] - 2 * action) ** 2) / 5
       
    
    
class RealDataGenerator(DataGenerator):
    def __init__(self, file_name='real_envir.pickle'):
        with open(file_name, 'rb') as handle:
            envir= pickle.load(handle) 
        
        self.org_data = envir[0]
        self.regr_mean = envir[1]
        self.sigma = envir[2]
        self.Q_nn = envir[3]
        self.tau = envir[4]
    
    def get_sample(self, policy):
        idx = random.sample(range(len(self.org_data)), 1)[0]
        context = self.org_data['xt'][idx]
        if policy == 'behavior':
            action = self.org_data['at'][idx] 
        else:
            action = policy(context)
                  
        # mean of y
        y_mean = self.regr_mean.predict(np.append(np.array(context),np.array(action)).reshape(1, len(context)+1))
        # std of y
        #res_mean = self.regr_res.predict(np.append(np.array(context),np.array(action)).reshape(1, len(context)+1))
        
        return context, action, np.random.normal(y_mean[0], self.sigma, 1)[0]  
    
    def get_mean_reward(self, policy):
        idx = random.sample(range(len(self.org_data)), 1)[0]
        context = self.org_data['xt'][idx]
        if policy == 'behavior':
            action = self.org_data['at'][idx] 
        else:
            action = policy(context) 
                  
        # mean of y
        y_mean = self.regr_mean.predict(np.append(np.array(context),np.array(action)).reshape(1, len(context)+1))
          
        return y_mean[0] 
    
    
class ToyExample(DataGenerator):
    def __init__(self, mu=0, sigma=1, lb=0., ub=1., p=20):
        super(ToyExample, self).__init__(mu, sigma, lb, ub, p)
        
    def get_context(self):
        return np.random.uniform(self._lb, self._ub, self.p)[0]
    
    def get_reward(self, context, action):
        return (action > 0.5) * 10 * np.log(context + 2) * max(action ** 2 - 0.25, 0) + np.random.normal(self._mu, self._sigma, 1)[0]  
        
    def get_mean_reward(self, context, action):
        return (action > 0.5) * 10 * np.log(context + 2) * np.maximum(action ** 2 - 0.25, 0)
 
