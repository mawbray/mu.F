<h1 align="center">
  <a href="https://github.com/mawbray/mu.F/blob/main/png/mu_.png">
    <img src="https://github.com/mawbray/mu.F/blob/main/png/mu_.png" width="200" height="200" /></a><br>
  <b>Multi-unit feasibility and flexibility</b><br>
</h1>
<p align="center">
      <a href="https://www.python.org/doc/versions/">
        <img src="https://img.shields.io/badge/python-3.10-blue.svg" /></a>  
      <a href="https://opensource.org/license/mit">
        <img src="https://img.shields.io/badge/license-MIT-orange" /></a>
</p>

# Overview 

- This code base provides a general sampling-based solver for numerical constraint satisfaction problems defined on directed acyclic graphs representative of function composition.
- The code is agnostic to the sampling scheme used, which may be customised by the user. By default a nested sampling scheme (DEUS) is used.
- The code exploits graph structure to improve the efficiency of solution identification. 

# Installation

### Source installation

#### Poetry installations

This project uses [Poetry](https://python-poetry.org/docs) to
manage dependencies in a local virtual environment. To install Poetry, [follow the
instructions in the Poetry documentation](https://python-poetry.org/docs/#installation).

To install this project in editable mode, run the commands below from the root directory of the `sipsolve` repository.

```bash
poetry install
```

Poetry `install` command creates a virtual environment for this project
in a hidden `.venv` directory under the root directory. 

This will make all bar two currrent dependencies avaialable. To configure default algorithms for sampling and solving semi-infinite programs we use pip. 

#### Pip installations

Two additional packages are managed by pip. The first is the DEUS sampler. To install, simply navigate to mu_F/samplers/algorithms/deus and follow the instructions within the README.md file. 

The second installation is the sipsolve repository. Again simply navigate to mu_F/post_processes/algorithms/sipsolve and follow the instructions within the README.md file.

# Algorithmic details

For details please find the pre-print:
```
@misc{mowbray2025decompositionapproachsolvingnumerical,
      title={A Decomposition Approach to Solving Numerical Constraint Satisfaction Problems on Directed Acyclic Graphs}, 
      author={Max Mowbray and Nilay Shah and Benoît Chachuat},
      year={2025},
      eprint={2511.10426},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2511.10426}, 
}
```

