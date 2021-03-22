
# Copyright (c) 2014 Johannes Kulick
# Copyright (c) 2021 Daniel La Braca
#
# Code initally borrowed from:
#    https://github.com/hildensia/bayesian_changepoint_detection
# under the MIT license.

'''
Bayesian Online Changepoint Detection as seen in:

Ryan P. Adams, David J.C. MacKay, Bayesian Online Changepoint Detection, arXiv 0710.3742 (2007)

Code is built on from the Repo: hildensia/bayesian_changepoint_detection

More Gaussian cases have been added along with a way to find the maximum a posteriori estimator for the joint posterior distribution p(r_1, r_2, ..., r_{t+1}|x_{1:t})
This can be used to estimate the changepoint locations in the timeseries.

How the updated statistics are found can be seen at:
Murphy, Kevin. (2007). Conjugate Bayesian analysis of the Gaussian distribution. 

Predicitve mean and variance has also been added for the 3 Gaussian cases
'''


from __future__ import division
import numpy as np
from scipy import stats


def online_changepoint_detection(data, hazard_func, observation_likelihood):
    maxes_R = np.zeros(len(data) + 1)
    
    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    
    P = np.zeros((len(data) + 1, len(data) + 1))
    P[0, 0] = 1
    
    args = np.zeros(len(data))
    
    pred_mean = np.zeros(len(data)+1)
    pred_var = np.zeros(len(data)+1)
    
    pred_mean[0] = observation_likelihood.mu[0]
    pred_var[0] = observation_likelihood.var[0]
    
    
    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = observation_likelihood.pdf(x)
        
        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(t+1)))
       
        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t+1] = np.sum( R[0:t+1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])



        # Keeping track of the most likely combination - difference from above is that when we reset, we just reset from one point and do not sum over all possible points
        # essentially finding most likely joint runlength distribution
        P[1:t+2, t+1] = P[0:t+1, t] * predprobs * (1-H)
        P[0, t+1] = np.max( P[0:t+1, t] * predprobs * H)
        
        # keep track of which runlength we reset from
        arg = np.argmax( P[0:t+1, t] * predprobs * H)
        args[t] = arg
        # improved numerical stability
        P[:, t+1] = P[:, t+1] / np.sum(P[:, t+1])
        
        # Update the parameter sets for each possible run length.
        observation_likelihood.update_theta(x)

        # make predictions
        pred_mean[t+1] = np.sum(R[:t+2, t+1] * observation_likelihood.mu)
        pred_var[t+1] = np.sum(R[:t+2, t+1] * observation_likelihood.var)
        
    
        maxes_R[t+1] = R[:, t+1].argmax()
     
    # trace back in time finding the most likely path ie the joint distribution of runlengths
    cps = find_path(np.argmax(P[:,-1]), args)
    cps = [cp +1 for cp in cps]


    return R, cps, maxes_R, pred_mean, pred_var


# finds the most likely path
def find_path(max_, args):
    cps = []
    cp = len(args) - max_ -1
    cps.append(cp)
    args = args[:-max_]
    while (cp > 0):
        if len(args) == 0:
            cp = -1
        else:
            fallen_from = int(args[-1])
            cp = cps[-1]- fallen_from - 1
            cps.append(cp)
            args = args[:-fallen_from-1]
        
    cps.reverse()
    
    return cps

def constant_hazard(lam, r):
    return 1/lam * np.ones(r.shape)

#  Gaussian unknown var
class GaussianUnknownVar:
    def __init__(self, mu, alpha, beta):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.mu0 = self.mu = np.array([mu])
        self.var0 = self.var = np.array([beta/alpha])

    def pdf(self, data):
        return stats.t.pdf(x=data, 
                           df=2*self.alpha,
                           loc=self.mu,
                           scale= np.sqrt(self.beta/self.alpha))

    def update_theta(self, data):
        muT0 = np.concatenate((self.mu0, self.mu))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0, self.beta + 0.5 * (data -self.mu)**2))
        varT0 = np.concatenate((self.var0, self.beta/self.alpha))
        
        self.mu = muT0   
        self.alpha = alphaT0
        self.beta = betaT0
        self.var = varT0



# Gaussian unknown mean 
class GaussianUnknownMean:
    def __init__(self, var , mu, sigma2):
        # var is the known varinace and sigam2 is the varinace for the prior
        self.mu0 = self.mu = np.array([mu])
        self.sigma20 = self.sigma2 = np.array([sigma2])
        self.var0 = self.var = np.array([var])
        
    def pdf(self, data):
        return stats.norm.pdf(x=data, 
                           loc=self.mu,
                           scale=np.sqrt(self.sigma2 + self.sigma2known))

    def update_theta(self, data):
        sigma2T0 = np.concatenate((self.sigma20, 1/(1/self.sigma2 + 1/self.var)))
        varT0 = np.concatenate((self.var0, self.var))
        
        self.sigma2 = sigma2T0
        
        muT0 = np.concatenate((self.mu0, self.sigma2[1:]*(self.mu + self.sigma2[:-1]*data/self.var)/self.sigma2[:-1]))
        
        self.mu = muT0
        self.var = varT0
        
        
# Gaussian unknown mean and var
class GaussianUnknownMeanVar:
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])
        self.var0 = self.var = np.array([beta * (kappa+1) / (alpha *kappa)])

    def pdf(self, data):
        return stats.t.pdf(x=data, 
                           df=2*self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa+1) / (self.alpha *
                               self.kappa)))

    def update_theta(self, data):
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0, self.beta + (self.kappa * (data -
            self.mu)**2) / (2. * (self.kappa + 1.))))  
        varT0 = np.concatenate((self.var0, self.beta * (self.kappa+1) / (self.alpha * self.kappa)))
        
        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0
        self.var = varT0

        
        
        
        
        