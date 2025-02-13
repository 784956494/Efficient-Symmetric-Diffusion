import torch

class Beta_Scheduler(object):
    def __init__(self, ts=0.0, tf=1.0, beta_s = 0.1, beta_f = 1.0):
        self.ts = ts
        self.tf = tf
        self.beta_s = beta_s
        self.beta_f = beta_f
    
    def integrate(self, t):
        #only of going to 0 to 1
        return (self.beta_s + 0.5 * (self.beta_f - self.beta_s) * t) * t  
    def step(self, t):
        t = t.clamp(self.ts+1e-8, self.tf-1e-8)
        return self.beta_s + (self.beta_f - self.beta_s) * (t - self.ts) / (self.tf - self.ts)