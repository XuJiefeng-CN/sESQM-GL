# sESQM
A MATLAB package implementing the smoothing extended sequential quadratic method (sESQM) [[1]] for solving group Lasso regularized minimization problems with polynomial constraints (PCGL):

$$\eqalign{
\min_{x\in \mathbb{R}^{n}} & \frac{1}{2}x^{\top} Q x + q^{\top}x + \tau \sum_{J \in 𝒥} |x_{J}|_{2}\\ 
\text{s.t.} & c_{i}(x) \coloneqq \sum_{j_1\in [n], \ldots, j_d \in [n]} a_{j_1, \ldots, j_d}^{(i)} x_{j_1}\cdots x_{j_d} + x^{\top} q^{i} - b_i \le 0 \quad \forall i \in [m],\\
&  | x_{J} |_{2} \le M \ \ \forall J\in 𝒥,
}$$

where $Q\in \mathbb{R}^{n \times n}$, $q \in \mathbb{R}^{n}$, $\tau > 0$, 
	$𝒥$ is a partition of $[n]$, 
	$x_{J}$ is the subvector of $x\in \mathbb{R}^{n}$ indexed by $J \in 𝒥$, 
	$q^{i} \in \mathbb{R}^{n}$, $A^{(i)} = (a_{j_1, \ldots, j_d}^{(i)})$ is a real-valued array with size $n \times \cdots \times n$ of order $d \ge 1$ for each $i\in [m]$, $b \in \mathbb{R}^{m}_{++}$, 
    and $M>0$.
For further details, please refer to our paper in [[1]].

## MATLAB Source Files

| File | Description |
|---|---|
| `demo_conv.m` | Reproduces the numerical experiments from [[1]] examining the effect of different decay rates of $\lbrace \mu_k \rbrace$ for the **convex** setting. |
| `demo_nonconv.m` | Reproduces the numerical experiments from [[1]] examining the effect of different decay rates of $\lbrace \mu_k \rbrace$  for the **nonconvex** setting. |
| `sESQM_GL.m` | Core implementation of sESQM for solving PCGL. |
| `genProblem.m` | Generates problem data for PCGL instances. |
| `sym_array.m` | Symmetrizes a given array. |
| `P1_fun.m` | Evaluates the group Lasso regularization term $P_1(x) = \tau \sum_{J \in 𝒥} \|x_{J}\|_{2}$. |
| `f_fun.m` | Evaluates the quadratic function $f(x) = \frac{1}{2}x^{\top} Q x + q^{\top}x$. |
| `df_fun.m` | Evaluates the gradient of $f$. |

# References
[1]: https://arxiv.org/abs/2606.13343 "J. Xu, T. K. Pong and Y. L. Zhang. A smoothing extended sequential quadratic method for difference-of-convex optimization over a convex composite inequality constraint. Preprint (2026)."
\[1\] [J. Xu, T. K. Pong and Y. L. Zhang. A smoothing extended sequential quadratic method for difference-of-convex optimization over a convex composite inequality constraint. Preprint (2026).](https://arxiv.org/abs/2606.13343)





