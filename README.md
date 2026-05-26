# 2QFG and 2QGG Saddle Point Experiments

## Overview

This repository contains Python code for the two main numerical experiments in Section 6 of **"Linear Convergence of a Unified Primal-Dual Algorithm for Convex-Concave Saddle Point Problems with Quadratic Growth"**. Both experiments compare several variants of the Generalized Accelerated Primal-Dual method (GAPD) with the baseline methods Gradient Descent-Ascent (GDA) and Extragradient (EG) on convex-concave saddle point problems satisfying two-sided quadratic functional growth (2QFG) and two-sided quadratic gradient growth (2QGG).

The repository contains two main files:

```text
Bilinear_2QFG2QGG_Final.py
NonBilinear_2QFG2QGG_Final.py
```

## Bilinear Experiment

The file `Bilinear_2QFG2QGG_Final.py` reproduces the bilinear saddle point experiment from Section 6.1 of the paper. It generates a rank-deficient quadratic convex-concave saddle point problem of the form

```math
\min_{x\in\mathbb{R}^n}\max_{y\in\mathbb{R}^m}
\frac{1}{2}\|C_1x-b_1\|^2
+
\langle Ax,y\rangle
-
\frac{1}{2}\|C_2y-b_2\|^2.
```

See the paper for more details on the problem construction. The code compares GDA, EG, and several GAPD variants for different choices of the parameter $\theta$. The methods are run for 10,000 iterations across three large problem dimensions:

```text
(900, 720, 540, 432)
(1500, 1200, 900, 720)
(3000, 2400, 1800, 1440)
```

The output includes normalized residual plots for GDA, EG, and the GAPD variants.

## Nonbilinear Experiment

The file `NonBilinear_2QFG2QGG_Final.py` reproduces the nonbilinear saddle point experiment from Section 6.2 of the paper. The problem is inspired by robust power control models in wireless networks and has the form

```math
\min_{x \in [-R,R]^n}
\max_{y \in [0,R]^n}
\sum_{i=1}^n a_i \rho(x_i)
+
\frac{1}{2}\sum_{i=1}^n \eta_i x_i^2 \log(1+y_i)
-
\sum_{i=1}^n b_i \rho(y_i),
```

where $\rho$ is a convex polynomial function. See the paper for more details on the problem. The code compares GDA, EG, and several GAPD variants for different choices of the parameter $\theta$. The methods are run for 5,000 iterations across three problem dimensions:

```text
n = 1000
n = 2000
n = 5000
```

The output includes normalized residual plots for GDA, EG, and the GAPD variants.

## Requirements

The code uses Python 3 and the following packages:

```text
numpy
scipy
matplotlib
```

## License

This code is released under the MIT License. See `LICENSE` for details.
