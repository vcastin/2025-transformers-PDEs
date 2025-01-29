# %%
import numpy as np
from utils import *

plt.rcParams["font.family"] = "times new roman"
plt.rcParams["font.size"] = 16

# %% Traditional self-attention
np.random.seed(40)
print("Traditional self-attention")

dimensions = np.array([3, 4, 5])
n_per_dim = 100
dim_repeated = np.repeat(dimensions, n_per_dim)
n_inits = 1
Q_list = [np.random.randn(d // 2, d) for d in dim_repeated]
K_list = [-Q for Q in Q_list]

rank_lists_trad = []
# %% V = I_d
print("V = I_d")
step_size_list = [0.09, 0.05, 0.02]
rank_lists_id, exceptions = create_histograms(
    dimensions,
    Q_list,
    K_list,
    [np.eye(d) for d in dim_repeated],
    rhs_trad,
    n_inits,
    n_per_dim,
    step_size_list=step_size_list,
    tol_evol=1e-4,
    max_iter_evol=60000,
    tol_rank=1e-1,
)
rank_lists_trad += rank_lists_id

# %% V random
np.random.seed(26)
print("V random")
V_list = [np.random.randn(d, d) for d in dim_repeated]
V_list = [np.dot(V.T, V) for V in V_list]
step_size_list = [0.09, 0.06, 0.02]

# tol_low_exceptions = [1e-4]
# step_size_exceptions = [0.1]

rank_lists_V, exceptions = create_histograms(
    dimensions,
    Q_list,
    K_list,
    V_list,
    rhs_trad,
    n_inits,
    n_per_dim,
    step_size_list=step_size_list,
    tol_evol=5e-5,
    max_iter_evol=90000,
    tol_rank=1e-1,
    n_max=100 * n_per_dim,
)
rank_lists_trad += rank_lists_V

# %%
n_rows = 2
f, axes = plt.subplots(n_rows, 3, figsize=(10, 3 * n_rows))
for i, ax in enumerate(axes.flat):
    bins = np.arange(0, dimensions[i % 3] + 1.5) - 0.5
    ax.hist(rank_lists_trad[i], bins=bins)
    ax.set_xticks(bins + 0.5)
    if i == 0:
        ax.set_ylabel("V identity", fontsize=18)
    if i == 3:
        ax.set_ylabel(r"V random", fontsize=18)
    if i < 3:
        ax.set_title(f"dim = {dimensions[i % 3]}")
    if i >= 3 * (n_rows - 1):
        ax.set_xlabel("rank")
plt.tight_layout()
plt.savefig("histograms_trad.pdf")

# %% L2 self-attention

np.random.seed(40)

print("L2 self-attention")
dimensions = np.array([3, 4, 5])
n_per_dim = 100
dim_repeated = np.repeat(dimensions, n_per_dim)
n_inits = 1
Q_list = [np.random.randn(d // 2, d) for d in dim_repeated]
K_list = [-Q for Q in Q_list]

rank_lists_L2 = []
# %% V = I_d
print("V = I_d")
step_size_list = [0.09, 0.06, 0.02]
rank_lists_id, exceptions = create_histograms(
    dimensions,
    Q_list,
    K_list,
    [np.eye(d) for d in dim_repeated],
    rhs_L2,
    n_inits,
    n_per_dim,
    step_size_list=step_size_list,
    tol_evol=1e-4,
    max_iter_evol=60000,
    tol_rank=1e-1,
)
rank_lists_L2 += rank_lists_id

# %% V random
np.random.seed(27)
print("V random")
step_size_list = [0.15, 0.15, 0.15]
V_list = [np.random.randn(d, d) for d in dim_repeated]
V_list = [np.dot(V.T, V) for V in V_list]
rank_lists_V, exceptions = create_histograms(
    dimensions,
    Q_list,
    K_list,
    V_list,
    rhs_L2,
    n_inits,
    n_per_dim,
    step_size_list=step_size_list,
    tol_evol=5e-5,
    max_iter_evol=50000,
    tol_rank=1e-1,
    n_max=1000,
)
rank_lists_L2 += rank_lists_V

# %%
n_rows = 2
f, axes = plt.subplots(n_rows, 3, figsize=(10, 3 * n_rows))
for i, ax in enumerate(axes.flat):
    bins = np.arange(0, dimensions[i % 3] + 1.5) - 0.5
    ax.hist(rank_lists_L2[i], bins=bins)
    ax.set_xticks(bins + 0.5)
    if i == 0:
        ax.set_ylabel("V identity", fontsize=18)
    if i == 3:
        ax.set_ylabel(r"V random", fontsize=18)
    if i < 3:
        ax.set_title(f"dim = {dimensions[i % 3]}")
    if i >= 3 * (n_rows - 1):
        ax.set_xlabel("rank")
plt.tight_layout()
plt.savefig("histograms_L2.pdf")
