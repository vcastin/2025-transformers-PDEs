import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib import cm, colormaps

########## Right-hand side of the covariance ODE

# S is the covariance matrix


def rhs_trad(S, Q, K, V, d, H):  # Softmax self-attention
    A = np.dot(K.T, Q) / np.sqrt(d)
    rhs = np.dot(np.dot(V, S), np.dot(A, S))
    return rhs + rhs.T


def rhs_L2(S, Q, K, V, d, H):  # L2 self-attention
    A = np.dot(K.T, Q) / np.sqrt(d)
    KK = np.dot(K.T, K)
    inv = np.linalg.inv(np.eye(d) + 2 * KK @ S)
    rhs = 2 * V @ S @ inv @ A @ S
    return rhs + rhs.T


def rhs_multihead(S, Q, K, V, d, H):  # Multi-head Softmax self-attention
    W = np.eye(d)
    k, check = d // H, d % H
    if check != 0:
        print("Warning: d must be divisible by H")
    rhs = 0
    for h in range(H):
        A_h = np.dot(K[k * h : k * (h + 1), :].T, Q[k * h : k * (h + 1), :]) / np.sqrt(
            k
        )
        C_h = np.dot(W[k * h : k * (h + 1), :].T, V[k * h : k * (h + 1), :])
        to_add = np.dot(np.dot(C_h, S), np.dot(A_h, S))
        rhs += to_add
    return rhs + rhs.T


def rhs_sinkhorn(S, Q, K, V, d, H):  # Sinkhorn self-attention with epsilon = 1
    eigvals = np.linalg.eigvalsh(S)
    S_sqrt = scipy.linalg.sqrtm(S).real
    A = np.dot(K.T, Q)
    prod = np.dot(S_sqrt, A.T)
    C_S = 0.5 * (
        S_sqrt
        @ scipy.linalg.sqrtm(4 * (prod.dot(S)).dot(prod.T) + np.eye(d)).real
        @ np.linalg.pinv(S_sqrt)
        - np.eye(d)
    )
    rhs = V @ np.linalg.pinv(A.T) @ np.linalg.pinv(S) @ C_S @ S
    rhs = rhs + rhs.T
    return rhs


########## Projected Euler scheme


def psd_projection(S, tol=1e-14):
    # projects S to the set of positive semi-definite matrices
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals[eigvals < 0] = tol
    return eigvecs.dot(np.diag(eigvals).dot(eigvecs.T))


def evolution(
    S,
    Q,
    K,
    V,
    d=2,
    H=1,
    rhs=rhs_trad,
    max_iter=50000,
    step_size=0.01,
    tol_low=1e-5,
    tol_high=1e3,
    print_status=True,
):
    # computes the iterates for S up to convergence (norm of rhs < threshold_low) or explosion (norm of rhs > threshold_high) or max_iter
    converged = False
    S_list = []
    for i in range(max_iter):
        S_list.append(S.copy())
        rhs_val = rhs(S, Q, K, V, d, H)
        check_convergence = np.linalg.norm(rhs_val)
        S += step_size * rhs_val
        S = psd_projection(S)
        check_explosion = np.trace(S)
        if check_convergence < tol_low:
            if print_status:
                print(f"Converged in {i} iterations")
            converged = True
            break
        elif check_explosion > tol_high:
            if print_status:
                print(f"Diverged in {i} iterations")
            break
    if i == max_iter - 1:
        if print_status:
            print("Did not converge")
    return np.array(S_list), converged


########## 3D plot

sqrt2 = np.sqrt(2)


def f(x, y):
    return np.sqrt(x**2 + y**2)


def to_3d(S):
    a = S[:, 0, 0]
    b = S[:, 1, 1]
    c = S[:, 1, 0]
    return np.array([(a - b), 2 * c, (a + b)])


def to_2d(S):
    a = S[:, 0, 0]
    b = S[:, 1, 1]
    c = S[:, 1, 0]
    return np.array([(a - b), 2 * c])


def from_3d(X):
    u, w, v = X[0], X[1], X[2]
    a = (u + v) / 2
    b = (v - u) / 2
    c = w / 2
    return np.array([[a, c], [c, b]])


def plot_evolution(
    fig_name,
    S_lists,
    azim=-112,
    elev=19,
    rank_1=False,
    dir_mat_3d=np.array([0, 0, 0]),
    starting_point=False,
    cone=True,
    save=True,
):
    colormap = colormaps["viridis"]
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    scale = 0
    max_color = 0
    min_color = 1e10
    S_lists_3d = [to_3d(S_list) for S_list in S_lists]
    if starting_point:
        for S_list_3d in S_lists_3d:
            min_color = min(min_color, S_list_3d[1, 0])
            max_color = max(max_color, S_list_3d[1, 0])
    else:
        for S_list_3d in S_lists_3d:
            min_color = min(min_color, S_list_3d[2][-1])
            max_color = max(max_color, S_list_3d[2][-1])
    for S_list_3d in S_lists_3d:
        if starting_point:
            color = colormap((S_list_3d[1, 0] - min_color) / (max_color - min_color))
        else:
            color = colormap((S_list_3d[2][-1] - min_color) / (max_color - min_color))
        ax.plot(
            S_list_3d[0],
            S_list_3d[1],
            S_list_3d[2],
            color=color,
            alpha=0.6,
            linewidth=1,
        )
        ax.scatter(
            S_list_3d[0, -1],
            S_list_3d[1, -1],
            S_list_3d[2, -1],
            color=color,
            s=5,
            marker="o",
        )
        scale_i = np.max(np.abs(S_list_3d[2]))
        scale = max(scale, scale_i)
    if rank_1:
        # Plot the line of direction dir_mat
        c = np.linspace(0, scale * np.sqrt(2), 100)
        ax.plot(
            dir_mat_3d[0] * c,
            dir_mat_3d[1] * c,
            dir_mat_3d[2] * c,
            color="black",
            alpha=0.5,
            linewidth=1,
        )
    if cone:
        # plot of the cone
        u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 80j]
        x = scale * np.cos(u) * np.sin(v)
        y = scale * np.sin(u) * np.sin(v)
        z = f(x, y)
        ax.plot_surface(x, y, z, alpha=0.07, color="k")
    ax.view_init(azim=azim, elev=elev)
    plt.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(fig_name)
    plt.show()


def plot_evolution_2d(fig_name, S_lists, save=True):
    colormap = colormaps["viridis"]
    f, ax = plt.subplots()
    scale = 0
    max_color = 0
    min_color = 1e10
    S_lists_2d = [to_2d(S_list) for S_list in S_lists]
    for S_list_2d in S_lists_2d:
        min_color = min(min_color, S_list_2d[1, 0])
        max_color = max(max_color, S_list_2d[1, 0])
    for S_list_2d in S_lists_2d:
        color = colormap((S_list_2d[1, 0] - min_color) / (max_color - min_color))
        ax.plot(S_list_2d[0], S_list_2d[1], color=color, alpha=0.6, linewidth=1)
        ax.scatter(S_list_2d[0, -1], S_list_2d[1, -1], color=color, s=5, marker="o")
    # plot of the unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, color="k", alpha=0.5, linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.axis("off")
    if save:
        plt.savefig(fig_name)
    plt.show()


def init_grid(n_U, n_C, n_V, limits=[-0.5, 0.5, -0.5, 0.5]):
    U = np.linspace(limits[0], limits[1], n_U)
    C = np.linspace(limits[2], limits[3], n_C)
    V = np.linspace(1.0, 1, n_V)
    inits = []
    X = np.array(np.meshgrid(U, C, V)).T.reshape(-1, 3)
    for x in X:
        u, w, v = x[0], x[1], x[2]
        if v**2 - u**2 > w**2:
            inits.append(from_3d(x))
    return inits


def compute_dir_mat(q):
    dir = np.random.randn(1, 2)
    dir = dir / np.linalg.norm(dir)
    dir = dir - np.inner(dir, q) * q
    dir = dir / np.linalg.norm(dir)
    dir_mat = np.outer(dir, dir)
    dir_mat_3d = to_3d(dir_mat[None, :, :])
    return dir_mat_3d / np.linalg.norm(dir_mat_3d)


def plot_cone(fig_name, azim=-112, elev=20, circle=False):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # plot of the cone
    u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 80j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = f(x, y)
    ax.plot_surface(x, y, z, alpha=0.07, color="k")
    ax.view_init(azim=azim, elev=elev)
    if circle == True:
        theta = np.linspace(0, 2 * np.pi, 100)
        scale = 0.5
        x = scale * np.cos(theta)
        y = scale * np.sin(theta)
        z = f(x, y)
        ax.plot(x, y, z, color="k", alpha=0.5, linewidth=0.8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()


######## Histograms


def create_histograms(
    dimensions,
    Q_list,
    K_list,
    V_list,
    rhs,
    n_inits,
    n_per_dim,
    step_size_list,
    tol_evol=1e-3,
    max_iter_evol=100000,
    tol_rank=1e-1,
    n_max=1000,
    seed=41,
    H=1,
):
    rank_lists = []
    exceptions = []
    for j, d in enumerate(dimensions):
        np.random.seed(seed)
        print("dim =", d)
        stat_points = []
        i = 0
        rank_list = []
        for k in range(n_max):
            index = n_per_dim * j + i
            Q, K, V = Q_list[index], K_list[index], V_list[index]
            inits = [np.random.randn(d, d) for _ in range(n_inits)]
            inits = [S @ S.T for S in inits]
            for S_0 in inits:
                S = S_0.copy()
                S_list, converged = evolution(
                    S,
                    Q,
                    K,
                    V,
                    d,
                    H=H,
                    rhs=rhs,
                    max_iter=max_iter_evol,
                    step_size=step_size_list[j],
                    tol_low=tol_evol,
                )
                if converged:
                    S_last = S_list[-1]
                    rank = np.sum(np.linalg.eigvalsh(S_last) > tol_rank)
                    if rank <= (d + 1) // 2:
                        rank_list.append(rank)
                        stat_points.append(S_list[-1])
                    else:
                        print("Rank too high:", rank)
                        exceptions.append((S_0, Q, K, V, step_size_list[j]))
                    i += 1
            if i == n_per_dim:
                break
        rank_lists.append(rank_list)
        print(f"there were {len(exceptions)} exceptions")
    return rank_lists, exceptions
