"""
Numerical experiment: Robust microgrid regulation under adversarial grid stress.

Problem:
    min_{x in [-R,R]^n} max_{y in [0,R]^n} f(x,y)
where
    f(x,y) = sum_i a_i rho(x_i) - sum_i b_i rho(y_i)
             + (1/2) sum_i eta_i x_i^2 log(1+y_i),
with rho(t) = t^2/2 - t^4/6 + t^6/30.

Properties:
  * Convex in x, concave in y, with non-bilinear coupling
    eta_i x_i^2 log(1+y_i) (quadratic in x, concave in y).
  * Unique saddle point z* = (0,0).
  * rho''(t) = (1-t^2)^2 vanishes at t = +/- 1, so f is NOT
    strongly convex-strongly concave on X x Y at R = 1.
  * Two-sided QGG (Euclidean): <F(z), z> >= c_R (a_min ||x||^2 + b_min ||y||^2),
    c_R = 4/9 at R > 1.
  * Two-sided QFG: f(x,0) - f(0,y) >= (7/24)(a_min ||x||^2 + b_min ||y||^2).

"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

R = 2.0


def rho(t):
    return 0.5 * t**2 - t**4 / 6.0 + t**6 / 30.0


def rho_prime(t):
    return t - 2.0 * t**3 / 3.0 + t**5 / 5.0


def grad_x(x, y, a, eta):
    return a * rho_prime(x) + eta * x * np.log1p(y)


def grad_y(x, y, b, eta):
    return -b * rho_prime(y) + 0.5 * eta * x**2 / (1.0 + y)


def proj_X(x):
    return np.clip(x, -R, R)


def proj_Y(y):
    return np.clip(y, 0.0, R)


def normalized_residual(x, y, x0, y0, eta):
    """ ||z - z*||^2/||z_0 - z*||^2 """
    #gx = grad_x(x, y, a, eta)
    #gy = grad_y(x, y, b, eta)
    return float((np.sum(x ** 2)+ np.sum(y ** 2)) / (np.sum(x0 ** 2)+ np.sum(y0 ** 2)))
            #float(np.sqrt(np.sum((x - proj_X(x - gx)) ** 2)
                 #        + np.sum((y - proj_Y(y + gy)) ** 2)))


def empirical_qgg_ratio(x, y, a, b, eta):
    """ <F(z), z - z*> / ||z - z*||^2  with z* = 0; should be bounded below
        away from zero by two-sided QGG. """
    z2 = float(np.sum(x * x) + np.sum(y * y))
    if z2 < 1e-30:
        return np.nan
    gx = grad_x(x, y, a, eta)
    gy = grad_y(x, y, b, eta)
    return float(np.sum(x * gx) + np.sum(y * (-gy))) / z2


# ---------------------------------------------------------------------------
# Lipschitz / QGG bounds
# ---------------------------------------------------------------------------

def lipschitz_and_qgg(a, b, eta):
    L_xx = float(np.max(a) + np.max(eta) * np.log1p(R))
    L_yy = float(np.max(b) + 0.5 * np.max(eta) * R * R)
    L_xy = float(np.max(eta) * R)
    L = max(L_xx, L_yy) + L_xy
    c_R = min(4.0 / 9.0, 7.0/24.0 * 2)
    mu = c_R * min(float(np.min(a)), float(np.min(b)))
    return L, L_xx, L_yy, L_xy, mu


# ---------------------------------------------------------------------------
# Algorithms (return both residual and QGG ratio along iterates)
# ---------------------------------------------------------------------------

def _alloc(num_iter):
    return np.empty(num_iter + 1), np.empty(num_iter + 1)


def run_gda(x0, y0, a, b, eta, tau, num_iter):
    x, y = x0.copy(), y0.copy()
    res, qg = _alloc(num_iter)
    res[0] = normalized_residual(x, y, x0, y0, eta)
    qg[0] = empirical_qgg_ratio(x, y, a, b, eta)
    for k in range(num_iter):
        gx = grad_x(x, y, a, eta)
        gy = grad_y(x, y, b, eta)
        x = proj_X(x - tau * gx)
        y = proj_Y(y + tau * gy)
        res[k + 1] = normalized_residual(x, y, x0, y0, eta)
        qg[k + 1] = empirical_qgg_ratio(x, y, a, b, eta)
    return res, qg


def run_eg(x0, y0, a, b, eta, tau, num_iter):
    x, y = x0.copy(), y0.copy()
    res, qg = _alloc(num_iter)
    res[0] = normalized_residual(x, y, x0, y0, eta)
    qg[0] = empirical_qgg_ratio(x, y, a, b, eta)
    for k in range(num_iter):
        gx = grad_x(x, y, a, eta)
        gy = grad_y(x, y, b, eta)
        x_half = proj_X(x - tau * gx)
        y_half = proj_Y(y + tau * gy)
        gx_h = grad_x(x_half, y_half, a, eta)
        gy_h = grad_y(x_half, y_half, b, eta)
        x = proj_X(x - tau * gx_h)
        y = proj_Y(y + tau * gy_h)
        res[k + 1] = normalized_residual(x, y, x0, y0, eta)
        qg[k + 1] = empirical_qgg_ratio(x, y, a, b, eta)
    return res, qg


def run_gapd(x0, y0, a, b, eta, tau, sigma, alpha, theta, num_iter):
    """
    Generalized Accelerated Primal-Dual with Euclidean Bregman.
      beta = alpha (1 - theta).
      theta = 0   -> OGDA-type single-call iteration.
      theta = 1   -> APD (beta = 0).
    """
    beta = alpha * (1.0 - theta)
    x, y = x0.copy(), y0.copy()
    x_prev, y_prev = x0.copy(), y0.copy()
    gx_prev = grad_x(x_prev, y_prev, a, eta)
    gy_prev = grad_y(x_prev, y_prev, b, eta)
    res, qg = _alloc(num_iter)
    res[0] = normalized_residual(x, y, x0, y0, eta)
    qg[0] = empirical_qgg_ratio(x, y, a, b, eta)
    for k in range(num_iter):
        gx_k = grad_x(x, y, a, eta)
        gy_k = grad_y(x, y, b, eta)
        q_y = gy_k - gy_prev
        y_new = proj_Y(y + sigma * (gy_k + alpha * q_y))
        q_x = gx_k - gx_prev
        gx_mixed = grad_x(x, y_new, a, eta)
        s = theta * gx_mixed + (1.0 - theta) * gx_k + beta * q_x
        x_new = proj_X(x - tau * s)
        x_prev, y_prev = x, y
        gx_prev, gy_prev = gx_k, gy_k
        x, y = x_new, y_new
        res[k + 1] = normalized_residual(x, y, x0, y0, eta)
        qg[k + 1] = empirical_qgg_ratio(x, y, a, b, eta)
    return res, qg


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------

def make_instance(n, seed):
    """
    Strong-coupling regime: weak diagonal curvature (a_i, b_i small),
    strong cross-coupling (eta_i large). This is where two-sided QFG/QGG
    is meaningful but strong convexity-strong concavity fails.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.1, 1.1, size=n)
    b = rng.uniform(0.1, 1.1, size=n)
    eta = rng.uniform(1.0, 10.0, size=n)
    x0 = rng.uniform(-R, R, size=n)
    y0 = rng.uniform(0.0, R, size=n)
    return a, b, eta, x0, y0


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_experiment(n, num_iter, seed):
    a, b, eta, x0, y0 = make_instance(n, seed)
    L, L_xx, L_yy, L_xy, mu = lipschitz_and_qgg(a, b, eta)

    tau_gda_safe = mu/(mu*max(L_xx,L_yy) + L_xy**2)    # Zamani-safe
    #tau_gda_safe = mu / (L * L)        
    tau_eg = tau_gda_safe #0.9 / L
    #tau_gapd = 0.9 / L
    #sigma_gapd = 0.9 / L
    #alpha_gapd = 1.0
    Lpsi = 1.0
    Lx2 = L_xx**2 + L_xy**2
    Ly2 = L_yy**2 + L_xy**2

    print(f"  n={n}: L_xy={L_xy:.3f}, mu(theory)={mu:.4f}, L/mu={L/mu:.1f}")
    print(f"        tau_GDA_safe={tau_gda_safe:.4e}, tau_EG={tau_eg:.4e}")

    results = {}
    qg_traj = {}
    res, qg = run_gda(x0, y0, a, b, eta, tau_gda_safe, num_iter)
    results["GDA"], qg_traj["GDA"] = res, qg
    res, qg = run_eg(x0, y0, a, b, eta, tau_eg, num_iter)
    results["EG"], qg_traj["EG"] = res, qg
    for theta in (0.0, 0.5, 0.8, 0.9, 1.0):
        varsigma = 2.0 * (1.0 - theta) if theta == 0.0 else theta
        alpha_x = 1.0 - (varsigma * mu) / (
            Lx2 + theta * L_xx
            + ((1.0 - theta)**2 * max(Lx2,Ly2) * Lpsi**2) / (varsigma * mu)
        )
        alpha_y = 1.0 - (varsigma * mu) / (
            Ly2 
            + ((1.0 - theta)**2 * max(Lx2,Ly2) * Lpsi**2) / (varsigma * mu)
        )
        
        alpha_gapd = min(alpha_x,alpha_y)
        tau_gapd = 2.0 * (1.0 - alpha_gapd) / (alpha_gapd * varsigma * mu)
        sigma_gapd = tau_gapd
        res, qg = run_gapd(x0, y0, a, b, eta, tau_gapd, sigma_gapd,
                           alpha_gapd, theta, num_iter)
        key = f"GAPD ($\\theta$={theta:g})"
        results[key] = res
        qg_traj[key] = qg
    meta = dict(L=L, mu=mu, tau_gda_safe=tau_gda_safe,
                tau_eg=tau_eg, tau_gapd=tau_gapd)
    return results, qg_traj, meta


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    "GDA":          dict(color="#888888", ls="--",  lw=1.6, marker="",  markevery=400),
    "EG":                  dict(color="#000000", ls="-.",  lw=1.4, marker="s", markevery=350, markersize=4),
    r"GAPD ($\theta$=0)":    dict(color="#1f77b4", ls="-",   lw=1.4, marker="o", markevery=350, markersize=4),
    r"GAPD ($\theta$=0.5)":  dict(color="#2ca02c", ls="-",   lw=1.4, marker="^", markevery=370, markersize=4),
    r"GAPD ($\theta$=0.8)": dict(color="#ff7f0e", ls="-.",   lw=1.4, marker="", markevery=380, markersize=4),
    r"GAPD ($\theta$=0.9)": dict(color="#d62728", ls="-",   lw=1.8, marker="D", markevery=390, markersize=4),
    r"GAPD ($\theta$=1)":    dict(color="#9467bd", ls=(0, (1, 1)), lw=1.8, marker="x", markevery=410, markersize=5),
}
METHOD_ORDER = [
    "GDA", "EG",
    r"GAPD ($\theta$=0)", r"GAPD ($\theta$=0.5)",r"GAPD ($\theta$=0.8)",
    r"GAPD ($\theta$=0.9)", r"GAPD ($\theta$=1)",
]


def plot_convergence(all_results, dims, num_iter, savedir):
    fig, axes = plt.subplots(1, len(dims), figsize=(4.7 * len(dims), 4.5), sharey=True)
    if len(dims) == 1:
        axes = [axes]
    iters = np.arange(num_iter + 1)
    for ax, n, results in zip(axes, dims, all_results):
        for m in METHOD_ORDER:
            ax.semilogy(iters, np.maximum(results[m], 1e-16),
                        label=m, **METHOD_STYLES[m])
        ax.set_title(f"$n = {n}$")
        ax.set_xlabel("iteration $k$")
        ax.grid(True, which="both", alpha=0.3)
        #ax.set_ylim(1e-12, 5e1)
    axes[0].set_ylabel("$\\|z_k - z^*\\|^2/\\|z_0 - z^*\\|^2$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.01), fontsize=9)
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    pdf = os.path.join(savedir, "microgrid_convergence.pdf")
    png = os.path.join(savedir, "microgrid_convergence.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    print(f"Saved {pdf} and {png}")
    plt.close(fig)


def plot_qgg_verification(qg_trajs_per_dim, dims, num_iter, mu_theory_per_dim, savedir):
    """
    Show <F(z_k), z_k> / ||z_k||^2 along iterates of GAPD (theta=0.9) for
    each dimension. Stays bounded below by mu_QGG > 0 - direct numerical
    evidence that two-sided QGG holds along trajectories.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    iters = np.arange(num_iter + 1)
    chosen = r"GAPD ($\theta$=0.9)"
    for n, qg, mu, c in zip(dims, qg_trajs_per_dim, mu_theory_per_dim, colors):
        q = qg[chosen]
        ax.plot(iters[1:], q[1:], color=c, lw=1.4, label=f"$n={n}$")
        ax.axhline(mu, color=c, lw=1.0, ls=":", alpha=0.7)
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$\langle F(z_k),\, z_k\rangle\, /\, \|z_k\|^2$")
    ax.set_title("Empirical two-sided QGG ratio along iterates "
                 r"(dotted: $\mu$ from theory)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    pdf = os.path.join(savedir, "microgrid_qgg_verification.pdf")
    png = os.path.join(savedir, "microgrid_qgg_verification.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    print(f"Saved {pdf} and {png}")
    plt.close(fig)


def iteration_csv(all_results, dims, thresholds):
    lines = []
    header = ["method"] + [f"n={n},eps={eps:g}" for n in dims for eps in thresholds]
    lines.append(",".join(header))
    methods = list(next(iter(all_results)).keys())
    for m in methods:
        row = [m]
        for results in all_results:
            r = results[m]
            for eps in thresholds:
                idx = np.where(r <= eps)[0]
                row.append(str(int(idx[0])) if idx.size else "—")
        lines.append(",".join(row))
    return "\n".join(lines)


def pretty_table(all_results, dims, thresholds):
    lines = []
    methods = list(next(iter(all_results)).keys())
    lines.append("")
    lines.append("Iterations to reach residual thresholds (— = not reached)")
    lines.append("=" * 110)
    hdr = ["method".ljust(24)]
    for n in dims:
        for eps in thresholds:
            hdr.append(f"n={n}, e={eps:.0e}".rjust(13))
    lines.append("".join(hdr))
    lines.append("-" * 110)
    for m in methods:
        cells = [m.ljust(24)]
        for results in all_results:
            r = results[m]
            for eps in thresholds:
                idx = np.where(r <= eps)[0]
                cells.append((str(int(idx[0])) if idx.size else "—").rjust(13))
        lines.append("".join(cells))
    lines.append("=" * 110)
    return "\n".join(lines)


def print_pretty(all_results, dims, thresholds):
    print(pretty_table(all_results, dims, thresholds))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    savedir = os.path.join(script_dir, "microgrid_outputs")
    os.makedirs(savedir, exist_ok=True)

    dims = [1000, 2000, 5000]
    num_iter = 5000
    seeds = [11, 22, 33]

    all_results, all_qg, all_meta = [], [], []
    for n, s in zip(dims, seeds):
        print(f"--- Running n={n} (seed={s}) ---")
        res, qg, meta = run_experiment(n, num_iter, s)
        all_results.append(res)
        all_qg.append(qg)
        all_meta.append(meta)

    plot_convergence(all_results, dims, num_iter, savedir)
    plot_qgg_verification(all_qg, dims, num_iter,
                          [m["mu"] for m in all_meta], savedir)

    thresholds = [1e-2, 1e-4, 1e-6, 1e-9]
    table = pretty_table(all_results, dims, thresholds)
    print(table)
    with open(os.path.join(savedir, "iterations_table.txt"), "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    csv = iteration_csv(all_results, dims, thresholds)
    with open(os.path.join(savedir, "iterations_table.csv"), "w", encoding="utf-8") as fh:
        fh.write(csv + "\n")
    print(f"\nSaved plots and tables in {savedir}")
