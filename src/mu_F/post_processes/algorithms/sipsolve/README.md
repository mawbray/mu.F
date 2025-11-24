
# Algorithm Overview

In this repository, we have implementation for an approximation algorithm for solving semi-infinite programs (SIP) of the form:

$$ \min_{\textbf{x} \in \mathcal{X}} \ f(\textbf{x}) $$
$$    \quad \text{s.t.} \quad g(\textbf{x}, \textbf{y})\leq \mathbf{0}, \quad \forall \textbf{y} \in \mathcal{Y} $$


with $\mathcal{X}\subseteq \mathbb{R}^{n_x}$ and $\mathcal{Y} \subseteq \mathbb{R}^{n_y}$. 

The challenge here is the constraint index set is assumed to be of infinite cardinality, and the constraint system is therefore infinite dimensional. This formulation describes a wide range of problems, including robust and bilevel programs. Constrained optimization algorithms are reliant on having access to a finite dimensional system of equations to solve. Given the constraint set is infinite dimensional, how best to proceed?

The algorithm follows the lead of Blankenship and Falk (1976), who propose a relaxation approach. 

They a) discretise the constraint set, to produce $\mathcal{Y}_k \subset \mathcal{Y}$ with finite cardinality, b) solve the problem above given $\mathcal{Y}_k$, which gives a lower bound to the optimal objective value under the solution $\textbf{x}_k$, c) they then solve the following feasibility problem

$$ \max_{\textbf{y} \in \mathcal{Y}} \ g(\textbf{x}_k, \textbf{y}) $$

and if the value function of this problem is non-positive we can assume that b) has yielded the optimum of the original problem and we can terminate. 

If not then we define $\mathcal{Y}_{k+1} = \mathcal{Y}_k \bigcup \\{\textbf{y}_k\\}$ and iterate $k \leftarrow k +1$ continuing with b) and c) iteration until convergence.

#### NOTE
 Adding this element to the discretised constraint index set is equivalent to adding a nonlinear cut to our discretised problem which reduces the feasible region.  In principle we should observe convergence in the optimal value functions of the discretised problem b) through iterations. This would require us to solve subproblems b) and c) to global optimality, which could be expensive. In this implementation, we simply use local nonlinear programming solvers and so i) are not guaranteed convergence and ii) may not recover an exact solution. However, we avoid the requirement for global optimization, which means we can be fast. To ensure that we at least identify a feasible solution, we have implemented validation procedures to interrogate the solution identified. 

```
@article{blankenship1976infinitely,
  title={Infinitely constrained optimization problems},
  author={Blankenship, Jerry W and Falk, James E},
  journal={Journal of Optimization Theory and Applications},
  volume={19},
  number={2},
  pages={261--281},
  year={1976},
  publisher={Springer}
}
```

### Installation

Simply navigate to /dist and run `pip install sipsolve-0.0.1.tar.gz`