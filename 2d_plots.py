# %%
import numpy as np
import matplotlib.pyplot as plt
from utils import *

plt.rcParams['font.family'] = 'times new roman'
plt.rcParams['font.size'] = 14

# %% Convergence to zero (Softmax, L2 and multi-head Softmax attention)
np.random.seed(35)

Q = np.random.rand(2, 2)
K = -Q
V = np.random.randn(2, 2)
V = V @ V.T

inits = init_grid(10, 10, 1)
S_lists = [evolution(S, Q, K, V, rhs=rhs_trad, step_size=0.3, tol_low=1e-5)[0] for S in inits]
S_lists_L2 = [evolution(S, Q, K, V, rhs=rhs_L2, step_size=0.3, tol_low=1e-5)[0] for S in inits]
inits = init_grid(10, 10, 1)
S_lists_MH = [evolution(S, Q, K, V, rhs=rhs_multihead, step_size=0.3, tol_low=1e-5, H=2)[0] for S in inits]
inits = init_grid(10, 10, 1)
S_lists_sink = [evolution(S, Q, K, V, rhs=rhs_sinkhorn, step_size=0.5, tol_low=1e-5)[0] for S in inits]

plot_evolution('trad_neg_def_V.pdf', S_lists, elev=20, azim=-174, starting_point=True, save=True)
plot_evolution('L2_neg_def_V.pdf', S_lists_L2, elev=20, azim=-174, starting_point=True, save=True)
plot_evolution('MH_neg_def_V.pdf', S_lists_MH, elev=20, azim=-174, starting_point=True, save=True)
plot_evolution('sink_convergence.pdf', S_lists_sink, elev=20, azim=-174, starting_point=True, save=True)

# %% Convergence to a line (Softmax and L2 attention)
np.random.seed(36)

q = np.random.randn(1, 2)
q = q / np.linalg.norm(q)
Q = q
K = -q      # to prevent the dynamics from diverging
V = np.random.randn(2, 2)
dir_mat_3d = compute_dir_mat(q)

inits = init_grid(10, 10, 1)
S_lists = [evolution(S, Q, K, V=np.eye(2), step_size=0.5, tol_low=1e-4)[0] for S in inits]
inits = init_grid(10, 10, 1)
S_lists_L2 = [evolution(S, Q, K, V=np.eye(2), rhs=rhs_L2, step_size=0.5, tol_low=1e-4)[0] for S in inits]

plot_evolution('trad_rank_1.pdf', S_lists, elev=20, azim=-86, rank_1=True, dir_mat_3d=dir_mat_3d, save=True)
plot_evolution('L2_rank_1.pdf', S_lists_L2, elev=20, azim=-76, rank_1=True, dir_mat_3d=dir_mat_3d, save=True)

# %% Convergence to a plane (multi-head Softmax attention)
np.random.seed(36)

q = np.random.randn(1, 2)
q = q / np.linalg.norm(q)
Q = q
K = -q      # to prevent the dynamics from diverging
V = np.random.randn(2, 2)
dir_mat_3d = compute_dir_mat(q)

inits = init_grid(10, 10, 1)
S_lists_MH = [evolution(S, Q, K, V=np.eye(2), rhs=rhs_multihead, step_size=0.5, tol_low=1e-4, H=2)[0] for S in inits]
plot_evolution('MH_rank_1.pdf', S_lists_MH, elev=20, azim=-45, rank_1=False, dir_mat_3d=dir_mat_3d, save=True)


# %% Convergence to two lines (Softmax and L2 attention)
np.random.seed(36)

for _ in range(1000):
    Q = np.random.randn(2, 2)
    K = np.random.randn(2, 2)
    A = np.dot(K.T, Q)
    eigvals, _ = np.linalg.eigh(A + A.T)
    if eigvals[0] * eigvals[1] < 0:
        break
np.random.seed(32)
V = np.random.randn(2, 2)

# inits = init_grid(20, 20, 1, limits=[-1., 1., -1., 1.])
inits = init_grid(10, 10, 1, limits=[-1., 1., -1., 1.])
S_lists_V = [evolution(S, Q, K, V, step_size=0.01, tol_low=1e-7)[0] for S in inits]
# inits = init_grid(20, 20, 1, limits=[-1., 1., -1., 1.])
inits = init_grid(10, 10, 1, limits=[-1., 1., -1., 1.])
S_lists_V = [evolution(S, Q, K, V, rhs=rhs_L2, step_size=0.01, tol_low=1e-7)[0] for S in inits]

plot_evolution('trad_mixed_3d.pdf', S_lists_V, azim=-88, elev=20, starting_point=False, save=True)
plot_evolution('L2_mixed_3d.pdf', S_lists_V, azim=-88, elev=20, starting_point=False, save=True)

# %% Blow-up or divergence (Softmax attention, multi-head Softmax attention, L2 attention, Sinkhorn attention)
np.random.seed(33)

for _ in range(1000):
    Q = np.random.randn(2, 2)
    K = np.random.randn(2, 2)
    A = np.dot(K.T, Q)
    eigvals, _ = np.linalg.eigh(A + A.T)
    if eigvals[0] > 0:
        break

inits = init_grid(40, 40, 1, limits=[-1., 1., -1., 1.])
S_lists = [evolution(S, Q, K, V=np.eye(2), rhs=rhs_trad, max_iter=100000, step_size=0.01, tol_low=1e-4)[0] for S in inits]
S_lists_renorm = [np.array([S / np.trace(S) for S in S_list]) for S_list in S_lists]

inits = init_grid(40, 40, 1, limits=[-1., 1., -1., 1.])
S_lists_MH = [evolution(S, Q, K, V=np.eye(2), rhs=rhs_multihead, max_iter=100000, step_size=0.002, tol_low=1e-4, H=2)[0] for S in inits]
S_lists_MH_renorm = [np.array([S / np.trace(S) for S in S_list]) for S_list in S_lists_MH]

inits = init_grid(40, 40, 1, limits=[-1., 1., -1., 1.])
S_lists_L2 = [evolution(S, Q, K, V=np.eye(2), rhs=rhs_L2, step_size=0.001, tol_low=1e-5, tol_high=1e4)[0] for S in inits]
S_lists_L2_renorm = [np.array([S / np.trace(S) for S in S_list]) for S_list in S_lists_L2]

inits = init_grid(40, 40, 1, limits=[-1., 1., -1., 1.])
S_lists_sink = [evolution(S, Q, K, V=np.eye(2), rhs=rhs_sinkhorn, step_size=0.01, tol_low=1e-5, tol_high=1e4)[0] for S in inits]
S_lists_sink_renorm = [np.array([S / np.trace(S) for S in S_list]) for S_list in S_lists_sink]

plot_evolution_2d('trad_explosion.pdf', S_lists_renorm, save=True)
plot_evolution_2d('MH_explosion.pdf', S_lists_MH_renorm, save=True)
plot_evolution_2d('L2_divergence.pdf', S_lists_L2_renorm, save=True)
plot_evolution_2d('sink_divergence.pdf', S_lists_sink_renorm, save=True)

# %% Convergence and blow-up depending on initial data (Softmax, L2 and multi-head Softmax attention)
np.random.seed(36)

for _ in range(1000):
    Q = np.random.randn(2, 2)
    K = np.random.randn(2, 2)
    A = np.dot(K.T, Q)
    eigvals, _ = np.linalg.eigh(A + A.T)
    if eigvals[0] * eigvals[1] < 0:
        break
np.random.seed(35)
_ = np.random.randn(8, 7)
V = np.random.randn(2, 2)

# inits = init_grid(30, 30, 1, limits=[-1., 1., -1., 1.])
inits = init_grid(10, 10, 1, limits=[-1., 1., -1., 1.])
S_lists_V = [evolution(S, Q, K, V, step_size=0.003, tol_low=1e-4)[0] for S in inits]
S_lists_renorm_V = [np.array([S / np.trace(S) for S in S_list]) for S_list in S_lists_V]

# inits = init_grid(30, 30, 1, limits=[-1., 1., -1., 1.])
inits = init_grid(10, 10, 1, limits=[-1., 1., -1., 1.])
S_lists_V_L2 = [evolution(S, Q, K, V, rhs=rhs_L2, step_size=0.002, tol_low=1e-4)[0] for S in inits]
S_lists_renorm_V_L2 = [np.array([S / np.trace(S) for S in S_list]) for S_list in S_lists_V_L2]

# inits = init_grid(30, 30, 1, limits=[-1., 1., -1., 1.])
inits = init_grid(10, 10, 1, limits=[-1., 1., -1., 1.])
S_lists_V_MH = [evolution(S, Q, K, V, rhs=rhs_multihead, step_size=0.002, tol_low=1e-4)[0] for S in inits]
S_lists_renorm_V_MH = [np.array([S / np.trace(S) for S in S_list]) for S_list in S_lists_V_MH]

plot_evolution_2d('trad_both_V.pdf', S_lists_renorm_V, save=True)
plot_evolution_2d('L2_both.pdf', S_lists_renorm_V_L2, save=True)
plot_evolution_2d('MH_both.pdf', S_lists_renorm_V_MH, save=True)
