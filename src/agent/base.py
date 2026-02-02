"""
Base agent class implementation
"""

import os
import pickle
import jax.numpy as jnp
import pandas as pd
import numpy as np

from pathlib import Path
from functools import partial
from omegaconf import OmegaConf
from collections import deque
from contextlib import contextmanager, redirect_stdout, redirect_stderr

from constraints.evaluator import current_q_evaluator as Q_Network
from visualisation.visualiser import visualiser
from visualisation.methods import add_policy, reconstruction_plot
from graph.graph_assembly import build_graph_structure

# --- Global Constamts ---
OUTPUTS_DIR = str(Path(__file__).parent.parent) + "/outputs/{solve_date}/{solve_id}/"
PICKLE_FILE = "graph_{mode}_iterate_{i}_node_{n}.pickle"
HYDRA_CONFIG_FILE = ".hydra/config.yaml"
XSLX_file = "inside_samples_{mode}_iterate_{i}.xlsx"
POOL = "mp-ms"


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
        self._out_dir = OUTPUTS_DIR.format(solve_date=solve_date, solve_id=solve_id)
        self.cfg = OmegaConf.load(self._out_dir + HYDRA_CONFIG_FILE)
        self.cfg = build_graph_structure(self.cfg)
        self.cfg.solvers.evaluation_mode.reward = POOL
        self.graph = self._load_pickle()
        self.q_network = partial(Q_Network, cfg=self.cfg, graph=self.graph, pool=POOL)

    # ---- Public methods ---- #
    def act(self, u: jnp.ndarray):
        """Take an action based on the current state u"""

        if self._node == 0:
            u = jnp.empty((u.shape[0], 0))

        with suppress_output():
            v = self.q_network(node=self._node)(u[jnp.newaxis, :], None)

        self._actions.append((self._node, v))
        self._node += 1

        return v

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

        actions = dict(self._actions)
        cols = list(self.cfg.case_study.design_space_dimensions)
        policy_vec = []
        for idx in range(len(cols)):
            if idx in actions:
                policy_vec.append(float(np.ravel(actions[idx])[0]))
            else:
                policy_vec.append(np.nan)
        vis = add_policy(vis, policy_vec, cfg=self.cfg, color="r", marker="o", size=60)
        vis.savefig(self._out_dir + "reconstructed_with_policy.svg", dpi=300)
        return None

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
