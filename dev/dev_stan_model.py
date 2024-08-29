"""Try out Stan posterior probability model"""

import os
import numpy as np
from cmdstanpy import CmdStanModel

# Turn off Stan logging
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.WARNING)


if __name__ == '__main__':

    n_samples = 100
    thin_factor = 100

    mu = 2
    sigma = 1
    reduction_factor = 0.5
    z = 2.7

    stan_model = CmdStanModel(stan_file=os.path.join('stan_models','posterior.stan'))
    data = {'mu':mu,'sigma':sigma,'reduction_factor':reduction_factor,'z':z}
    inits = {'theta':2}
    post_fit = stan_model.sample(data=data, inits=inits, iter_warmup=n_samples, iter_sampling=n_samples*thin_factor, chains=1, show_progress=False)
    post_samples = np.round(post_fit.stan_variable('theta')[::thin_factor],2)

    print(post_samples)
    print('mean,std',np.round(np.mean(post_samples),2),np.round(np.std(post_samples),2))
    print('min,max',np.min(post_samples),np.max(post_samples))
    print('bounds',mu-2*sigma,mu+2*sigma)