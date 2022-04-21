import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from jax.nn import relu
from jax.config import config

import numpy as onp

import itertools
from functools import partial
from torch.utils import data
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import scipy.io
from scipy.interpolate import griddata
        
import timeit

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

    # Data loader
    class DataGenerator_batch(data.Dataset):
        def __init__(self, s_train, u_train, y_train, m = 100, P = 100,
                    batch_size=64, N_ensemble = 10, rng_key=random.PRNGKey(1234)):
            'Initialization'
            self.s_train = s_train
            self.u_train = u_train
            self.N_train_realizations = s_train.shape[0]
            self.norms = vmap(np.linalg.norm, (0, None))(s_train, np.inf) # 2

            self.y = y_train

            self.batch_size = batch_size
            self.N_ensemble = N_ensemble
            self.key = rng_key

        def __getitem__(self, index):
            'Generate one batch of data'
            self.key, subkey = random.split(self.key)
            v_subkey  = random.split(subkey, self.N_train_realizations)
            u_temp, y_temp, s_temp, w_temp = self.__get_realizations(v_subkey)
            self.key, subkey = random.split(self.key)
            v_subkey = random.split(subkey, self.N_ensemble)
            inputs, outputs = vmap(self.__data_generation, (0, None, None, None, None))(v_subkey, u_temp, y_temp, s_temp, w_temp)
            return inputs, outputs

        @partial(jit, static_argnums=(0,))
        def __data_generation(self, key, u_temp, y_temp, s_temp, w_temp):
            'Generates data containing batch_size samples'            
            idx = random.choice(key, self.N_train_realizations * self.batch_size, (self.batch_size,), replace=False)
            u = u_temp[idx,:]
            y = y_temp[idx,:]
            s = s_temp[idx,:]
            w = w_temp[idx,:]
            # Construct batch
            inputs = (u, y)
            outputs = (s, w)
            return inputs, outputs

        @partial(jit, static_argnums=(0,))
        def __get_realizations(self, key):
            idx_train = np.arange(self.N_train_realizations)

            u_temp, y_temp, s_temp, w_temp = vmap(self.__generate_one_realization_data, (0, 0, None, None, None))(key, idx_train, self.s_train, self.u_train, self.norms) 

            u_temp = np.float32(u_temp.reshape(self.N_train_realizations * self.batch_size,-1))
            y_temp = np.float32(y_temp.reshape(self.N_train_realizations * self.batch_size,-1))
            s_temp = np.float32(s_temp.reshape(self.N_train_realizations * self.batch_size,-1))
            w_temp = np.float32(w_temp.reshape(self.N_train_realizations * self.batch_size,-1))

            return u_temp, y_temp, s_temp, w_temp

        def __generate_one_realization_data(self, key, idx, s_train, u_train, norms):

            s = s_train[idx]
            u0 = u_train[idx]
            ww = norms[idx]

            u = np.tile(u0, (self.batch_size, 1))
            w = np.tile(ww, (self.batch_size))

            idx_keep = random.choice(key, s.shape[0], (self.batch_size,), replace=False)
                        
            return u, self.y[idx_keep,:], s[idx_keep], w
        
            
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

            # Logger
            self.itercount = itertools.count()
            self.loss_log = []

        # Define the operator net
        def operator_net(self, params, params_prior, u, y):
            branch_params, trunk_params = params
            branch_params_prior, trunk_params_prior = params_prior
            B = self.branch_apply(branch_params, u) + self.branch_apply_prior(branch_params_prior, u)
            T = self.trunk_apply(trunk_params, y) + self.trunk_apply_prior(trunk_params_prior, y)
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

            
    def Y_encoding(y):
        H = 5
        x = y
        for i in range(H):
            x = np.vstack((x, np.sin(2**i*np.pi*y), np.cos(2**i*np.pi*y)))
        return x.flatten()
    # Load the training data

    P = 72
    m = int(72*72)
    N_train = 1825
    N_test  = 1825

    d = np.load("weather_dataset.npz")
    u_train = d["U_train"][:N_train,:]
    S_train = d["S_train"][:N_train,:]/1000.
    y_train = d["Y_train"]

    u_train = np.array(u_train)
    S_train = np.array(S_train)
    y_train = np.array(y_train)

    d = np.load("weather_dataset.npz")
    u_test = d["U_train"][-N_test:,:]
    S_test = d["S_train"][-N_test:,:]/1000.
    y_test = d["Y_train"]

    u_test = np.array(u_test)
    S_test = np.array(S_test)
    y_test = np.array(y_test)

    # Positional Encoding of Y
    y_train = vmap(Y_encoding, (0))(y_train)
    y_test = vmap(Y_encoding, (0))(y_test)

    u_mu_total, u_std_total = np.mean(u_train), np.std(u_train)
    s_mu_total, s_std_total = np.mean(S_train), np.std(S_train)


    u_train = (u_train - u_mu_total) / u_std_total
    s_train = (S_train - s_mu_total) / s_std_total

    u_test = (u_test - u_mu_total) / u_std_total
    s_test = (S_test - s_mu_total) / s_std_total

    print("normalizing constant", u_mu_total, u_std_total, s_mu_total, s_std_total)

    print("shape of training data", u_train.shape, S_train.shape, y_train.shape)
    print("shape of testing data", u_test.shape, S_test.shape, y_test.shape)

    key = random.PRNGKey(0) # use different key for generating test data 
    keys = random.split(key, N_train)


    # visualize some training samples

    idxs = [30, 130, 230, 330, 430, 530, 630, 730, 830, 930, 1030, 1130, 1230, 1330, 1430, 1530, 1630, 1730]
    lon = np.linspace(0,355,num=72)
    lat = np.linspace(90,-87.5,num=72)

    # lon[-1]= 360
    lons,lats= np.meshgrid(lon,lat)

    for k in range(len(idxs)):
        
        idx = idxs[k]
        u = np.reshape(u_train[idx,:], (72, 72))
        s = np.reshape(s_train[idx,:], (72, 72))

        fig = plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.pcolor(lons, lats, u, cmap='jet')
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.title('Training $u(x,t)$')
        plt.colorbar()
        plt.tight_layout()

        plt.subplot(1,2,2)
        plt.pcolor(lons, lats, s, cmap='jet')
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.title('Training $s(x,t)$')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('./normed_training_Samples' + str(idx) + '.png', dpi = 300)


    # Initialize model
    branch_layers = [m, 128, 128, 128] 
    trunk_layers =  [22, 128, 128, 128] 
    N_ensemble = 128
    model = ParallelDeepOnet(branch_layers, trunk_layers, N_ensemble)

    # Create data set
    batch_size = 512
    dataset = DataGenerator_batch(s_train, u_train, y_train, m, P, batch_size, N_ensemble)


    # Train
    start_time = timeit.default_timer()
    model.train(dataset, nIter=30000)
    elapsed = timeit.default_timer() - start_time

    # Plot loss
    losses = np.asarray(model.loss_log)
    print(losses.shape)

    plt.figure(figsize = (8,4))
    plt.plot(losses, lw=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./normed_Losses.png', dpi = 300)


        
    # visualize some testing samples

    idxs = [30, 130, 230, 330, 430, 530, 630, 730, 830, 930, 1030, 1130, 1230, 1330, 1430, 1530, 1630, 1730]

    errors = onp.zeros((N_test, 1))
    Predict_mu_save = onp.zeros((N_test, 72, 72)) # Store the predicted mean
    Predict_std_save = onp.zeros((N_test, 72, 72)) # Store the predicted std


    for k in range(N_test):
    
        idx = k
        s_test_sample = s_test[idx,:] # May need [:,None]
        u0 = u_test[idx,:]

        u_test_sample = np.tile(u0, (P**2, 1))
        y_test_sample = y_test

        s_pred_sample = model.predict_s(u_test_sample, y_test_sample)[:,None]
        s_pred_sample_mu, s_pred_sample_std = np.mean(s_pred_sample, axis = 0)[:,None], np.std(s_pred_sample, axis = 0)[:,None]
        # print(s_pred_sample_mu.shape, s_pred_sample_std.shape)
        
        S_pred_sample_mu = np.reshape(s_pred_sample_mu, (P, P))
        S_pred_sample_std = np.reshape(s_pred_sample_std, (P, P))


        u = np.reshape(s_test_sample, (72, 72))
        error_s = np.linalg.norm(u - S_pred_sample_mu, 2) / np.linalg.norm(u, 2) 

        errors[k, 0] = error_s
        Predict_mu_save[k, :, :] = S_pred_sample_mu # Store the predicted mean
        Predict_std_save[k, :, :] = S_pred_sample_std # Store the predicted std


    np.save("normed_errors.npy", errors)
    np.save("Predict_mu_save", Predict_mu_save)
    np.save("Predict_std_save", Predict_std_save)


    print("Total time for training", elapsed)









