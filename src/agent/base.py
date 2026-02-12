"""
Base agent class implementation
"""

import os
import pickle
import jax.numpy as jnp
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Optional, List
from functools import partial
from omegaconf import OmegaConf
from collections import deque
from contextlib import contextmanager, redirect_stdout, redirect_stderr

from constraints.evaluator import current_q_evaluator as Q_Network
from constraints.evaluator import current_constraint_evaluator as Constraint_Surrogate
from visualisation.methods import add_policy, reconstruction_plot, plotting_format
from graph.graph_assembly import build_graph_structure

# --- Global Constamts ---
OUTPUTS_DIR = str(Path(__file__).parent.parent) + "/outputs/{solve_date}/{solve_id}/"
PICKLE_FILE = "graph_{mode}_iterate_{i}_node_{n}.pickle"
HYDRA_CONFIG_FILE = ".hydra/config.yaml"
XSLX_file = "inside_samples_{mode}_iterate_{i}.xlsx"
POOL = "mp-ms"
CFG_UPDATE = {
    "parallelised": False,
    "n_starts": 20,
    "n_rejects": 100,
    "rejection_margin": 100,
    "casadi_ipopt_options": {
        "maxiter": 2000,
        "verbose": False,
        "tol": 1e-4,
        "options": {
            "verbose": 0,
            "maxiter": 2000,
            "disp": False,
        },
    },
}


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


class Agent:
    _node = 0
    _actions = deque()

    def __init__(self, solve_date: str, solve_id: str):
        self._node = 0
        self._actions = deque()
        self._out_dir = OUTPUTS_DIR.format(solve_date=solve_date, solve_id=solve_id)
        self.cfg = OmegaConf.load(self._out_dir + HYDRA_CONFIG_FILE)
        self._update_cfg()
        self.graph = self._load_pickle()
        self.q_network = partial(Q_Network, cfg=self.cfg, graph=self.graph, pool=POOL)
        self.constraint_surrogate = partial(
            Constraint_Surrogate, cfg=self.cfg, graph=self.graph, pool=POOL
        )

    # ---- Public methods ---- #
    def act(self, u: jnp.ndarray):
        """Take an action based on the current state u"""

        if self._node == 0:
            u = jnp.empty((u.shape[0], u.shape[1], 0))

        with suppress_output():
            # First try the Q-network
            v, status = self.q_network(node=self._node)(u[jnp.newaxis, :], None)
            cons = False

            # If Q-network optimisation fails for this node, fall back to constraint surrogate
            if not bool(jnp.all(status)):
                v, status = self.constraint_surrogate(node=self._node)(u[jnp.newaxis, :], None)
                cons = True

        print(f"[INFO]: Action taken at node {self._node}: {v.T[0]}. Using {'value function' if not cons else 'constraint surrogate'}")

        self._actions.append((self._node, v.T))
        self._node += 1

        return v.T

    def add_policy_to_plot(self):
        data_frame = pd.read_excel(
            self._out_dir
            + XSLX_file.format(
                mode=self.cfg.case_study.mode[0],
                i=0,
            ),
            index_col=0,
        )
        data_frame = data_frame[self.cfg.case_study.design_space_dimensions]
        vis = reconstruction_plot(
            self.cfg,
            self.graph,
            data_frame,
            save=False,
            path=self._out_dir + "reconstructed_with_policy",
            include_decomposition=False,
        )

        cols = list(self.cfg.case_study.design_space_dimensions)
        col_to_idx = {name: i for i, name in enumerate(cols)}
        policy_vec = np.full(len(cols), np.nan, dtype=float)

        for node, action in self._actions:
            action_vec = np.ravel(action)
            node_dims = list(self.cfg.case_study.process_space_names[node])

            for dim_idx, value in enumerate(action_vec):
                if dim_idx >= len(node_dims):
                    break

                base_name = node_dims[dim_idx]
                target_name = f"{base_name}_{node}"
                col_idx = col_to_idx.get(target_name, col_to_idx.get(base_name))
                if col_idx is not None:
                    policy_vec[col_idx] = float(value)

        vis = add_policy(vis, policy_vec, cfg=self.cfg, color="r", marker="o", size=60)
        vis.savefig(self._out_dir + "reconstructed_with_policy.svg", dpi=300)
        return None

    def plot_trajectories(self, action_names: Optional[List] = None):

        import matplotlib.pyplot as plt

        plotting_format()

        if len(self._actions) == 0:
            raise ValueError("No actions recorded. Call `act` before plotting trajectories.")

        action_dim = np.ravel(np.asarray(self._actions[0][1])).shape[0]

        y_labels = list(self.cfg.case_study.process_space_names[0])[:action_dim]
        if len(y_labels) < action_dim:
            y_labels += [f"action_{i}" for i in range(len(y_labels), action_dim)]

        if action_names is not None:
            assert (
                len(action_names) == action_dim
            ), "Length of action namespace must equal environment action dimension."
        else:
            action_names = list(self.cfg.case_study.process_space_names[0])[:action_dim]
            if len(action_names) < action_dim:
                action_names += [f"action_{i}" for i in range(len(action_names), action_dim)]

        trajectories = []
        action_labels = []

        for node, action in self._actions:
            action_vec = np.ravel(np.asarray(action, dtype=float))
            if action_vec.shape[0] != action_dim:
                raise ValueError(
                    f"Inconsistent action size at node {node}. "
                    f"Expected {action_dim}, got {action_vec.shape[0]}."
                )
            trajectories.append(action_vec)
            action_labels.append(f"t{node}")

        lower_bounds = np.full(action_dim, np.nan, dtype=float)
        upper_bounds = np.full(action_dim, np.nan, dtype=float)
        for node_bounds in self.cfg.case_study.KS_bounds.design_args:
            for i, bounds in enumerate(node_bounds[:action_dim]):
                lb, ub = bounds
                if lb != "None" and ub != "None":
                    lb = float(lb)
                    ub = float(ub)
                    if np.isnan(lower_bounds[i]) or lb < lower_bounds[i]:
                        lower_bounds[i] = lb
                    if np.isnan(upper_bounds[i]) or ub > upper_bounds[i]:
                        upper_bounds[i] = ub

        actions = np.vstack(trajectories)
        fig, axs = plt.subplots(action_dim, 1, figsize=(12, 3.5 * action_dim), squeeze=False)
        axs = axs.ravel()

        steps = np.arange(actions.shape[0])
        for i, ax in enumerate(axs):
            ax.plot(steps, actions[:, i], marker="o", linewidth=3, color='r')
            ax.set_xticks(steps)
            ax.set_xticklabels(action_labels, rotation=45, ha="right")
            ax.set_ylabel(str(y_labels[i]))
            ax.set_title(str(action_names[i]))
            if not np.isnan(lower_bounds[i]) and not np.isnan(upper_bounds[i]):
                ax.set_ylim(lower_bounds[i], upper_bounds[i])
                ax.axhline(y=lower_bounds[i], ls="--", linewidth=2, c="black", alpha=0.7)
                ax.axhline(y=upper_bounds[i], ls="--", linewidth=2, c="black", alpha=0.7)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()

        return fig

    # ---- Private methods ---- #

    def _load_pickle(self):

        target_file = PICKLE_FILE.format(
            case_study=self.cfg.case_study,
            i=0,
            n=0,
            mode=self.cfg.case_study.mode[0],
        )

        graph = pickle.load(open(Path(self._out_dir) / target_file, "rb"))

        return graph

    def _update_cfg(self):
        """Update the cfg with any new values provided at call time. This is useful for updating the cfg with new surrogate parameters after training."""

        self.cfg = build_graph_structure(self.cfg)
        self.cfg.solvers.evaluation_mode.reward = POOL

        for key, value in CFG_UPDATE.items():
            setattr(self.cfg.solvers.forward_coupling, key, value)
        return None
