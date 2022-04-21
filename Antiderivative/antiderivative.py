import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from jax.nn import relu
from jax.config import config

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as onp



if __name__ == '__main__':


    # Define the neural net
    def MLP(layers, activation=relu):
        def init(rng_key):
            def init_layer(key, d_in, d_out):
                k1, k2 = random.split(key)
                glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
                W = glorot_stddev*random.normal(k1, (d_in, d_out))
                b = np.zeros(d_out)
                return W, b
            key, *keys = random.split(rng_key, len(layers))
            params = list(map(init_layer, keys, layers[:-1], layers[1:]))
            return params
        def apply(params, inputs):
            for W, b in params[:-1]:
                outputs = np.dot(inputs, W) + b
                inputs = activation(outputs)
            W, b = params[-1]
            outputs = np.dot(inputs, W) + b
            return outputs
        return init, apply

    # Define the data loader
    class DataGenerator_batch(data.Dataset):
        def __init__(self, u, y, s, norms, m = 100, P = 100,
                    batch_size=64, N_ensemble = 10, rng_key=random.PRNGKey(1234)):
            'Initialization'
            self.N_train_realizations = u.shape[0]
            self.P = P

            self.u = u
            self.y = y[0,:]
            self.s = s
            self.norm = norms 

            self.batch_size = batch_size
            self.N_ensemble = N_ensemble
            self.key = rng_key


        def __getitem__(self, index):
            'Generate one batch of data'
            self.key, subkey = random.split(self.key)
            v_subkey  = random.split(subkey, self.N_train_realizations)
            u_temp, y_temp, s_temp, w_temp = self.__get_realizations()
            self.key, subkey = random.split(self.key)
            v_subkey = random.split(subkey, self.N_ensemble)
            inputs, outputs = vmap(self.__data_generation, (0, None, None, None, None))(v_subkey, u_temp, y_temp, s_temp, w_temp)
            return inputs, outputs

        @partial(jit, static_argnums=(0,))
        def __data_generation(self, key, u_temp, y_temp, s_temp, w_temp):
            'Generates data containing batch_size samples'
            idx = random.choice(key, self.N_train_realizations * self.P, (self.batch_size,), replace=False)
            u = u_temp[idx,:]
            y = y_temp[idx,:]
            s = s_temp[idx,:]
            w = w_temp[idx,:]
            # Construct batch
            inputs = (u, y)
            outputs = (s, w)
            return inputs, outputs

        @partial(jit, static_argnums=(0,))
        def __get_realizations(self):
            idx_train = np.arange(self.N_train_realizations)

            u_temp, y_temp, s_temp, w_temp = vmap(self.__generate_one_realization_data, (0, None, None, None))(idx_train, self.u, self.s, self.norm) 

            u_temp = np.float32(u_temp.reshape(self.N_train_realizations * self.P,-1))
            y_temp = np.float32(y_temp.reshape(self.N_train_realizations * self.P,-1))
            s_temp = np.float32(s_temp.reshape(self.N_train_realizations * self.P,-1))
            w_temp = np.float32(w_temp.reshape(self.N_train_realizations * self.P,-1))

            return u_temp, y_temp, s_temp, w_temp

        def __generate_one_realization_data(self, idx, u, s, norms):
            
            u0 = u[idx]
            ww = norms[idx]
            s = s[idx]
            
            u = np.tile(u0, (self.P, 1))
            w = np.tile(ww, (self.P))

            return u, self.y, s, w


    # Define the model
    class ParallelDeepOnet:
        def __init__(self, branch_layers, trunk_layers, N_ensemble):    
            # Network initialization and evaluation functions
            self.branch_init, self.branch_apply = MLP(branch_layers, activation=relu)
            self.branch_init_prior, self.branch_apply_prior = MLP(branch_layers, activation=relu)
            self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=relu)
            self.trunk_init_prior, self.trunk_apply_prior = MLP(trunk_layers, activation=relu)
            
            # Initialize
            v_branch_params = vmap(self.branch_init)(random.split(random.PRNGKey(1234), N_ensemble))
            v_branch_params_prior = vmap(self.branch_init_prior)(random.split(random.PRNGKey(123), N_ensemble))
            v_trunk_params = vmap(self.trunk_init)(random.split(random.PRNGKey(4321), N_ensemble))
            v_trunk_params_prior = vmap(self.trunk_init_prior)(random.split(random.PRNGKey(321), N_ensemble))
            v_params = (v_branch_params, v_trunk_params)
            v_params_prior = (v_branch_params_prior, v_trunk_params_prior)

            # Use optimizers to set optimizer initialization and update functions
            self.opt_init, \
            self.opt_update, \
            self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                          decay_steps=1000, 
                                                                          decay_rate=0.9))
            self.v_opt_state = vmap(self.opt_init)(v_params)
            self.v_prior_opt_state = vmap(self.opt_init)(v_params_prior)

            self.beta = 1.0

            # Logger
            self.itercount = itertools.count()
            self.loss_log = []

        # Define the operator net
        def operator_net(self, params, params_prior, u, y):
            branch_params, trunk_params = params
            branch_params_prior, trunk_params_prior = params_prior
            B = self.branch_apply(branch_params, u) + self.beta*self.branch_apply_prior(branch_params_prior, u)
            T = self.trunk_apply(trunk_params, y) + self.beta*self.trunk_apply_prior(trunk_params_prior, y)
            outputs = np.sum(B * T)
            return outputs
        
        @partial(jit, static_argnums=(0,))
        def loss(self, params, params_prior, batch):
            # Fetch data
            inputs, outputs = batch
            u, y = inputs
            s, w = outputs
            # Compute forward pass
            pred = vmap(self.operator_net, (None, None, 0, 0))(params, params_prior, u, y)
            # Compute loss
            loss = np.mean(1./w.flatten()**2 * (s.flatten() - pred)**2)
            return loss

        # Define a compiled update step
        # @partial(jit, static_argnums=(0,))
        def step(self, i, opt_state, prior_opt_state, batch):
            params = self.get_params(opt_state)
            params_prior = self.get_params(prior_opt_state)
            g = grad(self.loss, argnums = 0)(params, params_prior, batch)
            return self.opt_update(i, g, opt_state)

        # Optimize parameters in a loop
        def train(self, dataset, nIter = 10000):
            data = iter(dataset)
            
            # Define v_step that vectorize the step operation
            self.v_step = jit(vmap(self.step, in_axes = [None, 0, 0, 0]))
            
            # Main training loop
            for it in range(nIter):
                batch = next(data)
                self.v_opt_state = self.v_step(it, self.v_opt_state, self.v_prior_opt_state, batch)
                # Logger
                if it % 1000 == 0:
                    params = vmap(self.get_params)(self.v_opt_state)
                    params_prior = vmap(self.get_params)(self.v_prior_opt_state)
                    loss_value = vmap(self.loss, (0, 0, 0))(params, params_prior, batch)
                    self.loss_log.append(loss_value)
                    print('iteration', it, 'Loss', loss_value)
           
               
        def operator_net_pred_single(self, params, params_prior, U_star, Y_star):
            s_pred_single = vmap(self.operator_net, (None, None, 0, 0))(params, params_prior, U_star, Y_star)
            return s_pred_single
        
        # Evaluates predictions at test points  
        @partial(jit, static_argnums=(0,))
        def predict_s(self, U_star, Y_star):
            params = vmap(self.get_params)(self.v_opt_state)
            params_prior = vmap(self.get_params)(self.v_prior_opt_state)
            s_pred = vmap(self.operator_net_pred_single, (0, 0, None,None))(params, params_prior, U_star, Y_star)
            return s_pred

        
    def RBF(x1, x2, params):
        output_scale, lengthscales = params
        diffs = np.expand_dims(x1 / lengthscales, 1) - \
                np.expand_dims(x2 / lengthscales, 0)
        r2 = np.sum(diffs**2, axis=2)
        return output_scale * np.exp(-0.5 * r2)

    def antiderivative(key, m=100, P=2, output_scale = 1.0):
        # Generate a GP sample
        N = 512
        gp_params = (10.**output_scale, 0.2)
        jitter = 1e-8
        X = np.linspace(0, 1, N)[:,None]
        K = RBF(X, X, gp_params)
        L = np.linalg.cholesky(K + jitter*np.eye(N))
        gp_sample = np.dot(L, random.normal(key, (N,)))
        # Create a callable interpolation function  
        u_fn = lambda x, t: np.interp(t, X.flatten(), gp_sample)
        # Input sensor locations and measurements
        x = np.linspace(0, 1, m)
        u = vmap(u_fn, in_axes=(None,0))(0.0, x)
        # Output sensor locations and measurements
        y = np.linspace(0, 1, P+1) # random.uniform(key, (P,)).sort() 
        s = odeint(u_fn, 0.0, np.hstack((0.0, y)))[2:] # JAX has a bug and always returns s(0), so add a dummy entry to y and return s[1:]
        
        return u, y[1:], s 

    def antiderivative_test(key, m=100, P=2, output_scale = 1.0):
        # Generate a GP sample
        N = 512
        gp_params = (10.**output_scale, 0.2)
        jitter = 1e-8
        X = np.linspace(0, 1, N)[:,None]
        K = RBF(X, X, gp_params)
        L = np.linalg.cholesky(K + jitter*np.eye(N))
        gp_sample = np.dot(L, random.normal(key, (N,)))
        # Create a callable interpolation function  
        u_fn = lambda x, t: np.interp(t, X.flatten(), gp_sample)
        # Input sensor locations and measurements
        x = np.linspace(0, 1, m)
        u = vmap(u_fn, in_axes=(None,0))(0.0, x)
        # Output sensor locations and measurements
        y = random.uniform(key, (P,)).sort() 
        s = odeint(u_fn, 0.0, np.hstack((0.0, y)))[1:] # JAX has a bug and always returns s(0), so add a dummy entry to y and return s[1:]
        
        # Tile inputs
        u = np.tile(u, (P,1))
                
        return u, y, s 


    # Use double precision to generate data (due to GP sampling)
    config.update("jax_enable_x64", True)

    # Training data
    N = 100
    m = 100 # number of input sensors
    P = 100   # number of output sensors
    K = 10
    Output_scales = np.linspace(-2, 2, K)  # K is the number of different output scales

    output_scale = Output_scales[0] # output scale
    key = random.PRNGKey(0)
    keys = random.split(key, N)
    gen_fn = jit(lambda key: antiderivative(key, m, P, output_scale))
    u_train, y_train, s_train = vmap(gen_fn)(keys)

    for k in range(1, K):
        output_scale = Output_scales[k] # output scale
        print("train output scale", output_scale)
        key = random.PRNGKey(k)
        keys = random.split(key, N)
        gen_fn = jit(lambda key: antiderivative(key, m, P, output_scale))
        u_train_temp, y_train_temp, s_train_temp = vmap(gen_fn)(keys)
        
        u_train = np.vstack((u_train, u_train_temp))
        y_train = np.vstack((y_train, y_train_temp))
        s_train = np.vstack((s_train, s_train_temp))
        
    print('train data shapes', u_train.shape, y_train.shape, s_train.shape)

    # Test data
    N = 100
    m = 100   # number of input sensors (needs to be the same as above!)
    P = 100   # number of output sensors
    output_scale = Output_scales[0] # output scale
    key = random.PRNGKey(K) # start this random key so that they do not be the same as the training data
    keys = random.split(key, N)
    gen_fn = jit(lambda key: antiderivative_test(key, m, P, output_scale))
    u_test, y_test, s_test = vmap(gen_fn)(keys)
    u_test = np.float32(u_test.reshape(N*P,-1))
    y_test = np.float32(y_test.reshape(N*P,-1))
    s_test = np.float32(s_test.reshape(N*P,-1))

    for k in range(1, K):
        output_scale = Output_scales[k] # output scale
        print("test output scale", output_scale)
        key = random.PRNGKey(k + K) # use different key from the training data
        keys = random.split(key, N)
        gen_fn = jit(lambda key: antiderivative_test(key, m, P, output_scale))
        u_test_temp, y_test_temp, s_test_temp = vmap(gen_fn)(keys)
        u_test_temp = np.float32(u_test_temp.reshape(N*P,-1))
        y_test_temp = np.float32(y_test_temp.reshape(N*P,-1))
        s_test_temp = np.float32(s_test_temp.reshape(N*P,-1))

        u_test = np.vstack((u_test, u_test_temp))
        y_test = np.vstack((y_test, y_test_temp))
        s_test = np.vstack((s_test, s_test_temp))

    print('test data shapes', u_test.shape, y_test.shape, s_test.shape)
        
    # compute the norm to balance the loss
    norms = vmap(np.linalg.norm, (0, None))(s_train, np.inf)

    # Switch back to signle precision for training
    config.update("jax_enable_x64", False)



    # Initialize model
    branch_layers = [m, 128, 128, 128] # [m, 40, 40]
    trunk_layers =  [1, 128, 128, 128] # [1, 40, 40]
    N_ensemble = 512
    model = ParallelDeepOnet(branch_layers, trunk_layers, N_ensemble)


    # Create data set
    batch_size = 64
    dataset = DataGenerator_batch(u_train, y_train, s_train, norms, m, P, batch_size, N_ensemble)


    # Train
    model.train(dataset, nIter=30000)


    # We do iterative prediction to avoid out of memory
    idx = 0
    index = np.arange(idx*P,(idx+1)*P)
    s_pred = model.predict_s(u_test[index,:], y_test[index,:])
    print(s_pred.shape)
    for i in range(1, N*K):
        idx = i
        index = np.arange(idx*P,(idx+1)*P)
        s_pred_temp = model.predict_s(u_test[index,:], y_test[index,:])

        s_pred = np.hstack((s_pred, s_pred_temp))

    # Checking the shape of the testing and training data
    print(s_train.shape, u_train.shape, y_train.shape)
    print(s_pred.shape, u_test.shape, y_test.shape) 

    # Save the predictions for plotting
    np.save("s_pred", s_pred)

    # Posterior mean and standard deviation
    s_pred_mu, s_pred_std = np.mean(s_pred, axis = 0)[:,None], np.std(s_pred, axis = 0)[:,None]
    print(s_pred_mu.shape, s_pred_std.shape)

    # Plot loss
    losses = np.asarray(model.loss_log)
    print(losses.shape)

    plt.figure(figsize = (8,4))
    plt.plot(losses, lw=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./Losses.png', dpi = 300)


    # Plot a sample test example
    idx = 835
    index = np.arange(idx*P,(idx+1)*P)
    plt.figure()
    for k in range(1, 20):
        plt.plot(y_test[index, :], s_pred[k,index], 'r--', lw=2)
    plt.plot(y_test[index, :], s_pred[0,index], 'r--', lw=2, label = "Predicted sample")
    plt.plot(y_test[index, :], s_test[index, :], 'b-', lw=2, label = "Exact")
    plt.plot(y_test[index, :], s_pred_mu[index, :], 'k--', lw=2, label = "Predicted mean")
    plt.legend(loc='upper right', frameon=False, prop={'size': 13})
    plt.xlabel('y')
    plt.ylabel('G(u)(y)')
    plt.tight_layout()
    plt.savefig('./Samples.png', dpi = 300)


    # Compute the errors and the uncertainty
    s_pred_mu, s_pred_std

    N_test_total = s_pred_mu.shape[0] // P
    N_test = N_test_total // K 

    errors = onp.zeros((K, N_test))
    uncertainty = onp.zeros((K, N_test))

    for idx in range(N_test_total):
        id1 = idx // N_test
        id2 = idx - id1 * N_test
        index = np.arange(idx*P,(idx+1)*P)
        s_pred_sample = s_pred_mu[index,:]
        s_pred_uncertainty = s_pred_std[index,:]
        s_test_sample = s_test[index,:]
                
        errors[id1, id2] = np.linalg.norm(s_pred_sample - s_test_sample, 2) / np.linalg.norm(s_test_sample, 2) 
        uncertainty[id1, id2] = np.linalg.norm(s_pred_uncertainty, 2) / np.linalg.norm(s_test_sample, 2)

    plt.figure()
    plt.errorbar(Output_scales, errors.mean(axis = 1), yerr=errors.std(axis = 1), fmt='.k')
    plt.errorbar(Output_scales, uncertainty.mean(axis = 1), fmt='r-')
    plt.savefig('./error_vs_uncertainty.png', dpi = 300)


    # Plot different samples from different output scales
    idxs = [30, 130, 230, 330, 430, 530, 630, 730, 830, 930]

    for m in range(len(idxs)):
        
        # Plot a sample 130 example
        idx = idxs[m]
        index = np.arange(idx*P,(idx+1)*P)
        plt.figure()
        for k in range(1, 20):
            plt.plot(y_test[index, :], s_pred[k,index], 'r--', lw=2)
        plt.plot(y_test[index, :], s_pred[0,index], 'r--', lw=2, label = "Predicted sample")
        plt.plot(y_test[index, :], s_test[index, :], 'b-', lw=2, label = "Exact")
        plt.plot(y_test[index, :], s_pred_mu[index, :], 'k--', lw=2, label = "Predicted mean")
        plt.legend(loc='upper right', frameon=False, prop={'size': 13})
        plt.xlabel('y')
        plt.ylabel('G(u)(y)')
        plt.tight_layout()
        plt.savefig('./Samples' + str(idx) + '.png', dpi = 300)


        




        
