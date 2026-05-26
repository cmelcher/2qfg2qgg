import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Problem setup: rank deficient quadratic convex-concave saddle point problem
# L(x,y) = 0.5||C1 x - b1||^2 + <A x, y> - 0.5||C2 y - b2||^2
# ============================================================

rng = np.random.default_rng(12345)

# Objectives and gradients for the problem
def L_val(x, y):
    return 0.5*np.linalg.norm(C1 @ x - b1)**2 + y @ (A @ x) - 0.5*np.linalg.norm(C2 @ y - b2)**2

def grad_x(x, y):
    return H1 @ x - C1.T @ b1 + A.T @ y

def grad_y(x, y):
    return A @ x - (H2 @ y - C2.T @ b2)  # = A x - C2^T(C2 y - b2)

def saddle_residual(x, y):
    m, n = x.shape[0], y.shape[0]
    gx = grad_x(x, y)
    gy = grad_y(x, y)
    return np.sqrt(np.linalg.norm(gx)**2 + np.linalg.norm(gy)**2)/np.sqrt(m+n)


# ============================================================
# GAPD algorithm
# ============================================================
def GAPD(A, C1, C2, b1, b2, iters, x0, y0, theta, vsigma):
    # Initialization
    x_prev = x0.copy()
    y_prev = y0.copy()
    x_curr = x0.copy()
    y_curr = y0.copy()   
    #Lipschitz terms
    L_xx = np.linalg.norm(C1,2)**2 
    L_yx = np.linalg.norm(A,2)
    L_xy = L_yx 
    L_yy = np.linalg.norm(C2,2)**2

    #mu calculation
    mu_x = mu_global
    mu_y = mu_global
    #helpers
    L_x = np.sqrt(L_xx**2 + L_yx**2)
    L_y = np.sqrt(L_yy**2 + L_xy**2)
    L_max = max(L_x,L_y)
    mu_min=min(mu_x,mu_y)

    #Parameter choices from paper (See Corollary 1)
    x_denom = L_x**2 + theta*L_xx + (((1-theta)**2)*(L_max**2))/(vsigma*mu_min)
    y_denom = L_y**2 + (((1-theta)**2)*L_max**2)/(vsigma*mu_min)
    alpha_x = 1- (vsigma*mu_x)/(x_denom)
    alpha_y = 1-(vsigma*mu_y)/(y_denom)
    alpha=max(alpha_x,alpha_y)

    #Optimal step sizes (See Corollary 1)
    tau = 2 * (1 - alpha) / (vsigma * alpha * mu_x) 
    sigma = 2 * (1 - alpha) / (vsigma * alpha * mu_y)
    #print(tau,sigma)
    beta = alpha * (1 - theta)

    res_hist = []
    res_hist.append(saddle_residual(x_curr, y_curr))
    #GAPD loop
    for _ in range(iters):
        # q_y^k
        gy_curr = grad_y(x_curr, y_curr)
        gy_prev = grad_y(x_prev, y_prev)
        q_y_k = gy_curr - gy_prev

        # y update
        y_next  = y_curr + sigma * (gy_curr + alpha * q_y_k)

        # q_x^k
        gx_curr_curr = grad_x(x_curr,y_curr)
        gx_curr_next = grad_x(x_curr, y_next)
        gx_prev_prev = grad_x(x_prev, y_prev)

        q_x_k = gx_curr_curr - gx_prev_prev

        # s_k
        s_k = (
            theta * gx_curr_next
            + (1 - theta) * grad_x(x_curr, y_curr)
            + beta * q_x_k
        )

        # x update
        x_next = x_curr - tau * s_k

        # logging (primal objective and gradient norm)
        res_hist.append(saddle_residual(x_next, y_next))

        # shift iterates for next loop
        x_prev, y_prev = x_curr, y_curr
        x_curr, y_curr = x_next, y_next

    return x_next, y_next, np.array(res_hist)


# ============================================================
# Gradient Descent–Ascent (GDA)
# x^{k+1} = x^k - eta*x grad_x L(x^k,y^k)
# y^{k+1} = y^k + eta*y grad_y L(x^k,y^k)
# stepsize from Zamani 2024
# ============================================================
def GDA(A, C1, C2, b1, b2, iters, x0, y0):
    #Lipschitz calculations for the problem
    L_xx = np.linalg.norm(C1,2)**2 
    L_yx = np.linalg.norm(A,2)
    L_xy = L_yx 
    L_yy = np.linalg.norm(C2,2)**2
    #Pull mu from global calculation 
    mu_x = mu_global
    mu_y = mu_global
    L = max(L_xx,L_yy)
    mu = min(mu_x,mu_y)
    
    #Stepsize from Zamani 2024
    #tstar = (mu)/(L*mu + 2*L_xy*np.sqrt(mu*(L-mu)+L_xy**2))
    tstar = (mu)/(L*mu + L_xy**2)

    #print(tstar)
    eta = tstar
    

    x = x0.copy()
    y = y0.copy()
    res_hist = []
    res_hist.append(saddle_residual(x, y))
    #GDA loop
    for k in range(iters):
        gx = grad_x(x, y)
        gy = grad_y(x, y)
        x = x - eta * gx
        y = y + eta * gy
        res_hist.append(saddle_residual(x, y))

    return x, y, np.array(res_hist)


#Extragradient baseline,  with stepsize from Zamani 2024.
#Lookaheads are computed as :
#     x_half = x^k - eta * grad_x L(x^k, y^k)
#     y_half = y^k + eta * grad_y L(x^k, y^k)
#then update:
#     x^{k+1} = x^k - eta * grad_x L(x_half, y_half)
#     y^{k+1} = y^k + eta * grad_y L(x_half, y_half)

def EG(A, C1, C2, b1, b2, iters, x0, y0):
    #Lipschitz calculation
    L_xx = np.linalg.norm(C1, 2)**2 
    L_yx = np.linalg.norm(A, 2)
    L_xy = L_yx 
    L_yy = np.linalg.norm(C2, 2)**2

    #Pull mu from global calculation
    mu_x = mu_global
    mu_y = mu_global

    L = max(L_xx, L_yy)
    mu = min(mu_x, mu_y)

    #Stepsize from Zamani 2024
    #eta = mu / (L * mu + 2 * L_xy * np.sqrt(mu * (L - mu) + L_xy**2))
    eta = (mu)/(L*mu + L_xy**2)

    x = x0.copy()
    y = y0.copy()

    res_hist = []
    res_hist.append(saddle_residual(x, y))
    #EG loop
    for k in range(iters):
        gx = grad_x(x, y)
        gy = grad_y(x, y)

        x_half = x - eta * gx
        y_half = y + eta * gy

        gx_half = grad_x(x_half, y_half)
        gy_half = grad_y(x_half, y_half)

        x = x - eta * gx_half
        y = y + eta * gy_half

        res_hist.append(saddle_residual(x, y))

    return x, y, np.array(res_hist)

# ============================================================
# Run the 3 methods and compare across 3 dimension sets
# ============================================================
dim_list = [
    (900, 720, 540, 432),
    (1500, 1200, 900, 720),
    (3000, 2400, 1800, 1440),
]

#run for 10,000 iterations
iters = 10000


fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

#run the 3 methods for each dimension choice
for plot_id, (n, m, p, q) in enumerate(dim_list):

    rng = np.random.default_rng(12345)

    #problem data generation

    #normalize C1, C2 by dimension
    C1 = rng.normal(size=(p, n)) 
    C1 = C1 / np.linalg.norm(C1, 2) * 10
    C2 = rng.normal(size=(q, m)) 
    C2 = C2 / np.linalg.norm(C2, 2) * 10

    #coupling constant
    rho = 1.0

    R = rng.normal(size=(q, p))
    R = R / np.linalg.norm(R, 2)

    A = rho * C2.T @ R @ C1

    b1 = rng.normal(size=p)
    b2 = rng.normal(size=q)

    #compute lipschitz constants for problem
    L_xx = np.linalg.norm(C1,2)**2 
    L_yx = np.linalg.norm(A,2)
    L_xy = L_yx 
    L_yy = np.linalg.norm(C2,2)**2

    
    print("\n====================================")
    print(f"Dimensions: n={n}, m={m}, p={p}, q={q}")
    print(f"L_xx = {L_xx:.6e}")
    print(f"L_xy = {L_xy:.6e}")
    print(f"L_yy = {L_yy:.6e}")

    H1 = C1.T @ C1
    H2 = C2.T @ C2

    #Hoffman constant calulation for unconstrained problem
    #(See Class of problems satisfying two-sided QFG/QGG section)
    D = np.block([
        [C1, np.zeros((p, m))],
        [A, np.zeros((m, m))],
        [np.zeros((q, n)), C2],
        [np.zeros((n, n)), A.T]
    ])

    sD = np.linalg.svd(D, compute_uv=False)
    rank_D = p + q
    sigma_D = sD[rank_D - 1]

    #kappas are 1 for this problem
    kappa_x = 1.0
    kappa_y = 1.0

    #nu calculations
    nu1 = rho**2 *np.linalg.norm(C2.T @ R, 2)**2
    nu2 = rho**2 *np.linalg.norm(R @ C1, 2)**2

    #mu calculation from the unconstrained family from the Hoffman Bound
    mu_global = (sigma_D**2)*min(kappa_x, kappa_y) / (max(1.0 + nu1, 1.0 + nu2))

    #print nu and mu calc for the problem
    print("nu1 =", nu1)
    print("nu2 =", nu2)
    print("mu =", mu_global)

    #ground KKT calculation for optimum
    KKT = np.block([[H1,         A.T],
                [A,    -H2        ]])
    rhs = np.concatenate([C1.T @ b1, -C2.T @ b2])
    sol = np.linalg.solve(KKT, rhs)
    x_star = sol[:n]
    y_star = sol[n:]

    #initial points
    x0 = np.zeros(n)
    y0 = np.zeros(m)

    #GDA and EG
    x_g, y_g, res_g = GDA(A, C1, C2, b1, b2, iters, x0, y0)
    x_eg, y_eg, res_eg = EG(A, C1, C2, b1, b2, iters, x0, y0)

    #GAPD for varying theta choices
    theta = 1.0
    vsigma = theta
    x_apd, y_apd, res_apd = GAPD(A, C1, C2, b1, b2, iters, x0, y0, theta, vsigma)

    theta = 0.0
    vsigma = 2 * (1 - theta)
    x_ogd, y_ogd, res_ogd = GAPD(A, C1, C2, b1, b2, iters, x0, y0, theta, vsigma)

    theta = 0.5
    vsigma = 2 * (1 - theta)
    x_p, y_p, res_p = GAPD(A, C1, C2, b1, b2, iters, x0, y0, theta, vsigma)

    theta = 0.8
    vsigma = 2 * (1 - theta)
    x_p2, y_p2, res_p2 = GAPD(A, C1, C2, b1, b2, iters, x0, y0, theta, vsigma)

    theta = 0.9
    vsigma = 2 * (1 - theta)
    x_p3, y_p3, res_p3 = GAPD(A, C1, C2, b1, b2, iters, x0, y0, theta, vsigma)

    print("Final residuals:")
    print(f"  GDA : {res_g[-1]:.3e}")
    print(f"  EG : {res_eg[-1]:.3e}")
    print(f"  GAPD-theta=0: {res_ogd[-1]:.3e}")
    print(f"  GAPD-theta=0.5: {res_p[-1]:.3e}")
    print(f"  GAPD-theta=0.8: {res_p2[-1]:.3e}")
    print(f"  GAPD-theta=0.9: {res_p3[-1]:.3e}")
    print(f"  GAPD-theta=1: {res_apd[-1]:.3e}")

    #print("Distances to (x*, y*):")
    #print(f"  ||x_g - x*||: {np.linalg.norm(x_g - x_star):.3e}   ||y_g - y*||: {np.linalg.norm(y_g - y_star):.3e}")
    #print(f"  ||x_eg - x*||: {np.linalg.norm(x_eg - x_star):.3e}   ||y_eg - y*||: {np.linalg.norm(y_eg - y_star):.3e}")
    #print(f"  ||x_p - x*||: {np.linalg.norm(x_p - x_star):.3e}   ||y_p - y*||: {np.linalg.norm(y_p - y_star):.3e}")

    ax = axes[plot_id]

    ax.semilogy(res_g / res_g[0], linestyle='-', lw=1.5, label=r"GDA")
    ax.semilogy(res_eg / res_eg[0], linestyle='--', lw=1.5, label=r"EG")
    ax.semilogy(res_ogd / res_ogd[0], linestyle='-.', lw=2.0, label=r"GAPD ($\theta=0$)")
    ax.semilogy(res_p / res_p[0], linestyle=':', lw=2.0, label=r"GAPD ($\theta=0.5$)")
    ax.semilogy(res_p2 / res_p2[0], linestyle='-', lw=1.5, label=r"GAPD ($\theta=0.8$)")
    ax.semilogy(res_p3 / res_p3[0], linestyle='-', lw=1.5, label=r"GAPD ($\theta=0.9$)")
    ax.semilogy(res_apd / res_apd[0], linestyle='--', lw=2.0, label=r"GAPD ($\theta=1$)")

    ax.set_title(rf"$n={n}, m={m}$", fontsize=14)
    ax.set_xlabel("Iteration", fontsize=14)
    ax.grid(True)

    if plot_id == 0:
        ax.set_ylabel(
            r"$\|F(z_k)\|/\|F(z_0)\|$",
            fontsize=14
        )

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=12)

plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.show()
