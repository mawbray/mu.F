import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


def print_visible(message):
    """Print message"""
    msg = f"\n{'='*60}\nSUBMODULE SETUP: {message}\n{'='*60}\n"
    print(msg, file=sys.stdout)
    sys.stdout.flush()

def init_submodules():
    """Initialize git submodules if they haven't been initialized yet."""
    try:
        setup_dir = os.path.dirname(os.path.abspath(__file__))
        gitmodules_path = os.path.join(setup_dir, ".gitmodules")
        
        submodule_paths = []
        with open(gitmodules_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("path = "):
                    path = line.split("path = ")[1]
                    submodule_paths.append(path)
                    
        # Check if any submodules are uninitialized (empty directories)
        needs_init = False
        uninit_modules = []
        for path in submodule_paths:
            full_path = os.path.join(setup_dir, path)
            if os.path.exists(full_path) and not os.listdir(full_path):
                needs_init = True
                uninit_modules.append(path)
            elif not os.path.exists(full_path):
                needs_init = True
                uninit_modules.append(path)

        if needs_init:
            print_visible(
                f"Initializing uninitialized submodules: {', '.join(uninit_modules)}"
            )

            # Initialize submodules
            print("Running: git submodule init", file=sys.stderr)
            sys.stderr.flush()
            result = subprocess.run(
                ["git", "submodule", "init"], cwd=setup_dir, text=True
            )

            if result.returncode != 0:
                print_visible(
                    f"Failed to initialize submodules (exit code: {result.returncode})"
                )
                return

            # Update submodules
            print("Running: git submodule update --recursive", file=sys.stderr)
            sys.stderr.flush()
            result = subprocess.run(
                ["git", "submodule", "update", "--recursive"], cwd=setup_dir, text=True
            )

            if result.returncode != 0:
                print_visible(
                    f"Failed to update submodules (exit code: {result.returncode})"
                )
                return

            print_visible("Submodules initialized successfully!")
        else:
            print_visible("All submodules already initialized")

    except Exception as e:
        print_visible(f"Could not initialize submodules: {e}")
        print(
            "Auto init failed: Run: 'git submodule update --init --recursive' manually",
            file=sys.stderr,
        )
        sys.stderr.flush()


class PostInstallCommand(install):
    """Custom install command that initializes submodules after installation."""

    def run(self):
        init_submodules()
        install.run(self)


class PostDevelopCommand(develop):
    """Custom develop command that initializes submodules after installation."""

    def run(self):
        init_submodules()
        develop.run(self)

# Defining GPU-Specific Packages
_GPU_PACKAGES = [
    "jax-cuda12-pjrt==0.4.23",
    "jax-cuda12-plugin==0.4.23",
    "nvidia-cuda-cupti-cu12==12.8.90",
    "nvidia-cuda-nvcc-cu12==12.9.86",
    "nvidia-cuda-nvrtc-cu12==12.8.93",
    "nvidia-cuda-runtime-cu12==12.8.90",
] 

# Resolve the relative path at build time
DEUS_PATH = (
    Path(__file__).parent
    / "src/samplers/algorithms/deus/src"
).as_uri()

_WEATHER_PACKAGE = "meteor_py @ git+https://github.com/cja119/meteor_py.git"

setup(
    name="mu.F",
    version="0.1",
    description="Multi-Unit Feasibility",
    author="Max Mowbray",
    author_email="maxmowbray@msn.com",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "scipy==1.11.4",
        "pandas>=2.2.0",

        # JAX Ecosystem
        "jax==0.4.23",
        "jaxlib==0.4.23",
        "jaxopt==0.8.2",
        "diffrax==0.3.1",
        "flax==0.8.3",
        "optax==0.1.7",
        "openpyxl==3.1.2",

        # Machine Learning
        "scikit-learn>=1.3.0",
        "torch>=2.4.0",
        "gpytorch>=1.13.0",

        # Optimization & Distributed Computing
        "casadi>=3.6.0",
        "ray>=2.20.0",

        # Configuration Management
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",

        # Visualization
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",

        # Graph Operations
        "networkx>=3.3.0",

        # Specialized Tools
        "sobol-seq>=0.2.0",
        "tensorflow==2.19.0",
        f"deus @ {DEUS_PATH}",  # Dynamic relative path resolution!
    ],
    extras_require={
        "gpu": _GPU_PACKAGES,
        "weather": [_WEATHER_PACKAGE],
    },
    cmdclass={
        "install": PostInstallCommand,
        "develop": PostDevelopCommand,
    },
)
