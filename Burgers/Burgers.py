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
        def __init__(self, usol, m = 101, P = 101,
                     batch_size=64, N_ensemble = 10, rng_key=random.PRNGKey(1234)):
            'Initialization'
            self.usol = usol
            self.N_train_realizations = usol.shape[0]
            self.m = m
            self.P = P
            u_samples_reshape = usol.reshape(self.N_train_realizations, m*P, -1)
            self.norms = vmap(np.linalg.norm, (0, None))(u_samples_reshape, np.inf) # 2
            
            t = np.linspace(0, 1, P)
            x = np.linspace(0, 1, P)
            T, X = np.meshgrid(t, x)
            self.y = np.hstack([T.flatten()[:,None], X.flatten()[:,None]])
            
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

            u_temp, y_temp, s_temp, w_temp = vmap(self.__generate_one_realization_data, (0, 0, None, None))(key, idx_train, self.usol, self.norms) 
            
            u_temp = np.float32(u_temp.reshape(self.N_train_realizations * self.batch_size,-1))
            y_temp = np.float32(y_temp.reshape(self.N_train_realizations * self.batch_size,-1))
            s_temp = np.float32(s_temp.reshape(self.N_train_realizations * self.batch_size,-1))
            w_temp = np.float32(w_temp.reshape(self.N_train_realizations * self.batch_size,-1))
                
            return u_temp, y_temp, s_temp, w_temp

        def __generate_one_realization_data(self, key, idx, usol, norms):

            u = usol[idx]
            u0 = u[0,:]
            ww = norms[idx]

            s = u.T.flatten()
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
                    branch_params_prior, trunk_params_prior = params_prior
                    #print(trunk_params_prior[0][0][0])
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

            

    # Prepare the training data

    # Load data
    path = 'Burger001.mat'  # Please use the matlab script to generate data

    data = scipy.io.loadmat(path)
    usol = np.array(data['output'])

    N = usol.shape[0]  # number of total input samples
    N_train =1000      # number of input samples used for training
    N_test = N - N_train  # number of input samples used for test
    m = 101            # number of sensors for input samples
    P = 101       # resolution of uniform grid for the data

    u0_train = usol[:N_train,0,:]   # Training data
    usol_train = usol[:N_train,:,:]

    u0_test = usol[N_train:N,0,:]   # Testing data
    usol_test = usol[N_train:N,:,:]

    print("shape of training data", u0_train.shape, usol_train.shape)
    print("shape of testing data", u0_test.shape, usol_test.shape)

    key = random.PRNGKey(0) # use different key for generating test data 
    keys = random.split(key, N_train)


    # visualize some training samples
    idxs = [30, 130, 230, 330, 430, 530, 630, 730, 830, 930]
    t = np.linspace(0, 1, P)
    x = np.linspace(0, 1, P)
    T, X = np.meshgrid(t, x)
        
    for k in range(len(idxs)):
        
        idx = idxs[k]
        u = usol[idx,:, :]
        u0 = usol[idx,0,:]

        plt.figure()
        plt.pcolor(T, X, u, cmap='jet')
        plt.title('Exact $s(x,t)$')
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.tight_layout()
        plt.savefig('./normed_training_Samples' + str(idx) + '.png', dpi = 300)


    # Initialize model
    branch_layers = [m, 128, 128, 128] 
    trunk_layers =  [2, 128, 128, 128] 
    N_ensemble = 128
    model = ParallelDeepOnet(branch_layers, trunk_layers, N_ensemble)


    # Create data set
    batch_size = 512
    dataset = DataGenerator_batch(usol_train, m, P, batch_size, N_ensemble)


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
    idxs = [30, 130, 230, 330, 430, 530, 630, 730, 830, 930]
    t = np.linspace(0, 1, P)
    x = np.linspace(0, 1, P)
    T, X = np.meshgrid(t, x)

    errors = onp.zeros((N_test, 1))
    Predict_mu_save = onp.zeros((N_test, 101, 101)) # Store the predicted mean
    Predict_std_save = onp.zeros((N_test, 101, 101)) # Store the predicted std
        
    for k in range(N_test):
    

        idx = k + N_train
        u = usol[idx,:, :]
        u0 = usol[idx,0,:]

        u_test_sample = np.tile(u0, (P**2, 1))
        y_test_sample = np.hstack([T.flatten()[:,None], X.flatten()[:,None]])
        s_test_sample = u.flatten()[:,None]

        s_pred_sample = model.predict_s(u_test_sample, y_test_sample)[:,None]
        s_pred_sample_mu, s_pred_sample_std = np.mean(s_pred_sample, axis = 0)[:,None], np.std(s_pred_sample, axis = 0)[:,None]
        # print(s_pred_sample_mu.shape, s_pred_sample_std.shape)
        
        S_pred_sample_mu = griddata(y_test_sample, s_pred_sample_mu.flatten(), (T, X), method='cubic')
        S_pred_sample_std = griddata(y_test_sample, s_pred_sample_std.flatten(), (T, X), method='cubic')

        S_pred_sample_mu = S_pred_sample_mu.T
        S_pred_sample_std = S_pred_sample_std.T

        error_s = np.linalg.norm(u - S_pred_sample_mu, 2) / np.linalg.norm(u, 2) 

        errors[k, 0] = error_s
        Predict_mu_save[k, :, :] = S_pred_sample_mu # Store the predicted mean
        Predict_std_save[k, :, :] = S_pred_sample_std # Store the predicted std

        if k in idxs:

            print("error_s: {:.3e}".format(error_s))

            fig = plt.figure(figsize=(18,12))
            plt.subplot(2,2,1)
            plt.pcolor(T, X, u, cmap='jet')
            plt.xlabel('$x$')
            plt.ylabel('$t$')
            plt.title('Exact $s(x,t)$')
            plt.colorbar()
            plt.tight_layout()

            plt.subplot(2,2,2)
            plt.pcolor(T, X, S_pred_sample_mu, cmap='jet')
            plt.xlabel('$x$')
            plt.ylabel('$t$')
            plt.title('Predict $s(x,t)$')
            plt.colorbar()
            plt.tight_layout()

            plt.subplot(2,2,3)
            plt.pcolor(T, X, np.abs(S_pred_sample_mu - u), cmap='jet')
            plt.xlabel('$x$')
            plt.ylabel('$t$')
            plt.title('Absolute error')
            plt.colorbar()
            plt.tight_layout()
            
            plt.subplot(2,2,4)
            plt.pcolor(T, X, S_pred_sample_std, cmap='jet')
            plt.xlabel('$x$')
            plt.ylabel('$t$')
            plt.title('Pred uncertainty')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig('./normed_testing_Samples' + str(idx) + '.png', dpi = 300)

            fig = plt.figure(figsize=(15,4))
            plt.subplot(1,3,1)
            plt.plot(x,u[25,:], 'b-', linewidth = 2, label = 'Exact')       
            plt.plot(x,S_pred_sample_mu[25,:], 'r--', linewidth = 2, label = 'Prediction')
            lower = S_pred_sample_mu[25,:] - 2.0*S_pred_sample_std[25,:]
            upper = S_pred_sample_mu[25,:] + 2.0*S_pred_sample_std[25,:]
            plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                            facecolor='orange', alpha=0.5, label="Two std band")
            plt.xlabel('$x$')
            plt.ylabel('$u(t,x)$')  
            plt.title('$t = 0.25$')
            
            plt.subplot(1,3,2)
            plt.plot(x,u[50,:], 'b-', linewidth = 2, label = 'Exact')       
            plt.plot(x,S_pred_sample_mu[50,:], 'r--', linewidth = 2, label = 'Prediction')
            lower = S_pred_sample_mu[50,:] - 2.0*S_pred_sample_std[50,:]
            upper = S_pred_sample_mu[50,:] + 2.0*S_pred_sample_std[50,:]
            plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                            facecolor='orange', alpha=0.5, label="Two std band")
            plt.xlabel('$x$')
            plt.ylabel('$u(t,x)$')
            plt.title('$t = 0.50$')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
            
            plt.subplot(1,3,3)
            plt.plot(x,u[75,:], 'b-', linewidth = 2, label = 'Exact')       
            plt.plot(x,S_pred_sample_mu[75,:], 'r--', linewidth = 2, label = 'Prediction')
            lower = S_pred_sample_mu[75,:] - 2.0*S_pred_sample_std[75,:]
            upper = S_pred_sample_mu[75,:] + 2.0*S_pred_sample_std[75,:]
            plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                            facecolor='orange', alpha=0.5, label="Two std band")
            plt.xlabel('$x$')
            plt.ylabel('$u(t,x)$')  
            plt.title('$t = 0.75$')
            plt.tight_layout()
            plt.savefig('./normed_testing_slices' + str(idx) + '.png')

    plt.figure(figsize = (8,4))
    plt.hist(errors.flatten(), bins = 30)
    plt.tight_layout()
    plt.savefig('./normed_Errors.png', dpi = 300)

    np.save("normed_errors.npy", errors)
    np.save("Predict_mu_save", Predict_mu_save)
    np.save("Predict_std_save", Predict_std_save)


    ########## Identify outlier #########
    idx_max = np.argmax(errors)
    print("outlier index and error", idx_max, errors[idx_max,0])
    idx = idx_max + N_train

    u = usol[idx,:, :]
    u0 = usol[idx,0,:]

    u_test_sample = np.tile(u0, (P**2, 1))
    y_test_sample = np.hstack([T.flatten()[:,None], X.flatten()[:,None]])
    s_test_sample = u.flatten()[:,None]

    s_pred_sample = model.predict_s(u_test_sample, y_test_sample)[:,None]
    s_pred_sample_mu, s_pred_sample_std = np.mean(s_pred_sample, axis = 0)[:,None], np.std(s_pred_sample, axis = 0)[:,None]
    # print(s_pred_sample_mu.shape, s_pred_sample_std.shape)
    
    S_pred_sample_mu = griddata(y_test_sample, s_pred_sample_mu.flatten(), (T, X), method='cubic')
    S_pred_sample_std = griddata(y_test_sample, s_pred_sample_std.flatten(), (T, X), method='cubic')

    S_pred_sample_mu = S_pred_sample_mu.T
    S_pred_sample_std = S_pred_sample_std.T

    error_s = np.linalg.norm(u - S_pred_sample_mu, 2) / np.linalg.norm(u, 2) 

    print("outlier index and error", idx_max, errors[idx_max,0])
    print("maximum error_s: {:.3e}".format(error_s))

    fig = plt.figure(figsize=(18,12))
    plt.subplot(2,2,1)
    plt.pcolor(T, X, u, cmap='jet')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Exact $s(x,t)$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(2,2,2)
    plt.pcolor(T, X, S_pred_sample_mu, cmap='jet')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Predict $s(x,t)$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(2,2,3)
    plt.pcolor(T, X, np.abs(S_pred_sample_mu - u), cmap='jet')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Absolute error')
    plt.colorbar()
    plt.tight_layout()
    
    plt.subplot(2,2,4)
    plt.pcolor(T, X, S_pred_sample_std, cmap='jet')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Pred uncertainty')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./normed_testing_Samples_max.png', dpi = 300)

    fig = plt.figure(figsize=(15,4))
    plt.subplot(1,3,1)
    plt.plot(x,u[25,:], 'b-', linewidth = 2, label = 'Exact')       
    plt.plot(x,S_pred_sample_mu[25,:], 'r--', linewidth = 2, label = 'Prediction')
    lower = S_pred_sample_mu[25,:] - 2.0*S_pred_sample_std[25,:]
    upper = S_pred_sample_mu[25,:] + 2.0*S_pred_sample_std[25,:]
    plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                    facecolor='orange', alpha=0.5, label="Two std band")
    plt.xlabel('$x$')
    plt.ylabel('$u(t,x)$')  
    plt.title('$t = 0.25$')
    
    plt.subplot(1,3,2)
    plt.plot(x,u[50,:], 'b-', linewidth = 2, label = 'Exact')       
    plt.plot(x,S_pred_sample_mu[50,:], 'r--', linewidth = 2, label = 'Prediction')
    lower = S_pred_sample_mu[50,:] - 2.0*S_pred_sample_std[50,:]
    upper = S_pred_sample_mu[50,:] + 2.0*S_pred_sample_std[50,:]
    plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                    facecolor='orange', alpha=0.5, label="Two std band")
    plt.xlabel('$x$')
    plt.ylabel('$u(t,x)$')
    plt.title('$t = 0.50$')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    plt.subplot(1,3,3)
    plt.plot(x,u[75,:], 'b-', linewidth = 2, label = 'Exact')       
    plt.plot(x,S_pred_sample_mu[75,:], 'r--', linewidth = 2, label = 'Prediction')
    lower = S_pred_sample_mu[75,:] - 2.0*S_pred_sample_std[75,:]
    upper = S_pred_sample_mu[75,:] + 2.0*S_pred_sample_std[75,:]
    plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                    facecolor='orange', alpha=0.5, label="Two std band")
    plt.xlabel('$x$')
    plt.ylabel('$u(t,x)$') 
    plt.title('$t = 0.75$')
    plt.tight_layout()
    plt.savefig('./normed_testing_slices_max.png')


    print("Total time for training", elapsed)






