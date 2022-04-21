import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian, lax
from jax.config import config
from jax.ops import index_update, index

from scipy.interpolate import griddata
        
if __name__ == '__main__':


    # Use double precision to generate data (due to GP sampling)
    def RBF(x1, x2, params):
        output_scale, lengthscales = params
        diffs = np.expand_dims(x1 / lengthscales, 1) - \
                np.expand_dims(x2 / lengthscales, 0)
        r2 = np.sum(diffs**2, axis=2)
        return output_scale * np.exp(-0.5 * r2)

    # A diffusion-reaction numerical solver
    def solve_ADR(key, Nx, Nt, P, length_scale):
        """Solve 1D
        u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x)
        with zero initial and boundary conditions.
        """
        xmin, xmax = 0, 1
        tmin, tmax = 0, 1
        k = lambda x: 0.01*np.ones_like(x)
        v = lambda x: np.zeros_like(x)
        g = lambda u: 0.01*u ** 2
        dg = lambda u: 0.02 * u
        u0 = lambda x: np.zeros_like(x)

        # Generate subkeys
        subkeys = random.split(key, 2)

        # Generate a GP sample
        N = 512
        gp_params = (1.0, length_scale)
        jitter = 1e-10
        X = np.linspace(xmin, xmax, N)[:,None]
        K = RBF(X, X, gp_params)
        L = np.linalg.cholesky(K + jitter*np.eye(N))
        gp_sample = np.dot(L, random.normal(subkeys[0], (N,)))
        # Create a callable interpolation function  
        f_fn = lambda x: np.interp(x, X.flatten(), gp_sample)

        # Create grid
        x = np.linspace(xmin, xmax, Nx)
        t = np.linspace(tmin, tmax, Nt)
        h = x[1] - x[0]
        dt = t[1] - t[0]
        h2 = h ** 2

        # Compute coefficients and forcing
        k = k(x)
        v = v(x)
        f = f_fn(x)

        # Compute finite difference operators
        D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
        D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
        D3 = np.eye(Nx - 2)
        M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
        m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
        v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
            v[2:] - v[: Nx - 2]
        )
        mv_bond = m_bond + v_bond
        c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond

        # Initialize solution and apply initial condition
        u = np.zeros((Nx, Nt))
        u = index_update(u, index[:,0], u0(x))
        # Time-stepping update
        def body_fn(i, u):
            gi = g(u[1:-1, i])
            dgi = dg(u[1:-1, i])
            h2dgi = np.diag(4 * h2 * dgi)
            A = mv_bond - h2dgi
            b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
            b2 = (c - h2dgi) @ u[1:-1, i].T
            u = index_update(u, index[1:-1, i + 1], np.linalg.solve(A, b1 + b2))
            return u
        # Run loop
        UU = lax.fori_loop(0, Nt-1, body_fn, u)

        # Input sensor locations and measurements
        xx = np.linspace(xmin, xmax, m)
        u = f_fn(xx)
        # Output sensor locations and measurements
        idx = random.randint(subkeys[1], (P, 2), 0, max(Nx, Nt))
        y = np.concatenate([x[idx[:,0]][:,None], t[idx[:,1]][:,None]], axis = 1)
        s = UU[idx[:,0], idx[:,1]]
        # x, t: sampled points on grid
        return (x, t, UU), (u, y, s)


    # Geneate test data corresponding to one input sample
    def generate_one_test_data(key, P):
        Nx = P
        Nt = P
        (x, t, UU), (u, y, s) = solve_ADR(key, Nx , Nt, P, length_scale)

        return UU.T, u # We need to make sure the initial condition is UU[0,:]. So, we make this shaping change.

    # Geneate test data corresponding to N input sample
    def generate_test_data(key, N, P):

        config.update("jax_enable_x64", True)
        keys = random.split(key, N)

        usol, u = vmap(generate_one_test_data, (0, None))(keys, P)
        print(usol.shape)

        config.update("jax_enable_x64", False)
        return usol, u


    key = random.PRNGKey(0)

    # GRF length scale
    length_scale = 0.2

    # Resolution of the solution
    Nx = 100
    Nt = 100

    N = 2000 # number of input samples
    m = Nx   # number of input sensors
    P_train = 100 # number of output sensors

    usol, u = generate_test_data(key, N, P_train)


    np.save("usol.npy", usol)
    np.save("u.npy", u)






