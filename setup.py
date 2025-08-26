import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


def print_visible(message):
    """Print message that's visible during pip install."""
    # Try multiple approaches to ensure visibility
    msg = f"\n{'='*60}\nSUBMODULE SETUP: {message}\n{'='*60}\n"

    # Method 1: Print to stderr (less likely to be captured)
    print(msg, file=sys.stderr)
    sys.stderr.flush()

    # Method 2: Also print to stdout
    print(msg, file=sys.stdout)
    sys.stdout.flush()

    # Method 3: Write directly to terminal if available
    try:
        with open("/dev/tty", "w") as tty:
            tty.write(msg)
            tty.flush()
    except:
        pass  # Not available on all systems


def init_submodules():
    """Initialize git submodules if they haven't been initialized yet."""
    try:
        # Check if we're in a git repository
        if not os.path.exists(".git"):
            print_visible("Not in a git repository, skipping submodule initialization")
            return

        # Get the directory where setup.py is located
        setup_dir = os.path.dirname(os.path.abspath(__file__))

        # Check if submodules exist and are empty
        gitmodules_path = os.path.join(setup_dir, ".gitmodules")
        if not os.path.exists(gitmodules_path):
            print_visible(
                "No .gitmodules file found, skipping submodule initialization"
            )
            return

        # Parse .gitmodules to find submodule paths
        submodule_paths = []
        with open(gitmodules_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("path = "):
                    path = line.split("path = ")[1]
                    submodule_paths.append(path)

        print_visible(
            f"Found {len(submodule_paths)} submodule(s): {', '.join(submodule_paths)}"
        )

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
            "You may need to run 'git submodule update --init --recursive' manually",
            file=sys.stderr,
        )
        sys.stderr.flush()


class PostInstallCommand(install):
    """Custom install command that initializes submodules after installation."""

    def run(self):
        # Initialize submodules before installation to ensure dependencies are available
        init_submodules()
        install.run(self)


class PostDevelopCommand(develop):
    """Custom develop command that initializes submodules after installation."""

    def run(self):
        # Initialize submodules before installation to ensure dependencies are available
        init_submodules()
        develop.run(self)


# Resolve the relative path at build time
deus_path = (
    Path(__file__).parent
    / "src/samplers/algorithms/deus/src"
).as_uri()

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
        "scipy>=1.11.0",
        "pandas>=2.2.0",

        # JAX Ecosystem
        "jax==0.4.23",
        "jaxlib==0.4.23",
        "jaxopt==0.8.2",
        "diffrax==0.3.1",
        "flax==0.8.3",
        "optax==0.1.7",

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
        "tensorflow>=2.16.0",
        f"deus @ {deus_path}",  # Dynamic relative path resolution!
    ],
    cmdclass={
        "install": PostInstallCommand,
        "develop": PostDevelopCommand,
    },
)
