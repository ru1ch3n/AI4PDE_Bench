from .diffusionpde_runner import DiffusionPDERunnerConfig, run_diffusionpde
from .supervised import SupervisedTrainConfig, train as train_supervised, evaluate as eval_supervised
from .pinn import PINNConfig, solve_pinn_instance
