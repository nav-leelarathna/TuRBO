from bo.models.gp_bo import GP_BO
from emukit.examples.gp_bayesian_optimization.enums import AcquisitionType
from bo.turbo import TurboM, Turbo1

class ModelFactory:
    def __init__(self, f,  # Handle to objective function
    lb,# Numpy array specifying lower bounds
    ub,  # Numpy array specifying upper bounds
    n_init=5, # Number of initial bounds from an Latin hypercube design
    accquisition_function=AcquisitionType.EI,
    verbose=True,  # Print information from each batch
    ):
        self.f = f
        self.lb = lb 
        self.ub = ub 
        self.n_init = n_init 
        self.acquisition_function = accquisition_function
        self.verbose = verbose


    def getGP_BO(self, batch_size, max_evals):
        return GP_BO(f=self.f, lb=self.lb, ub=self.ub, n_init=self.n_init, acquisition_type=self.acquisition_function, max_evals=max_evals, batch_size=batch_size, noiseless=True)
    
    def getTurbo1(self, batch_size, max_evals, use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64"):
        return Turbo1(
        f=self.f,  # Handle to objective function
        lb=self.lb,  # Numpy array specifying lower bounds
        ub=self.ub,  # Numpy array specifying upper bounds
        n_init=self.n_init,  # Number of initial bounds from an Latin hypercube design
        max_evals = max_evals,  # Maximum number of evaluations
        batch_size=batch_size,  # How large batch size TuRBO uses
        verbose=self.verbose,  # Print information from each batch
            use_ard=use_ard,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=max_cholesky_size,  # When we switch from Cholesky to Lanczos
        n_training_steps=n_training_steps,  # Number of steps of ADAM to learn the hypers
        min_cuda=min_cuda,  # Run on the CPU for small datasets
        device=device,  # "cpu" or "cuda"
        dtype=dtype
    )

    def getTurboM(self, batch_size, max_evals,trust_regions=5, use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64"):
        return TurboM(
        f=self.f,  # Handle to objective function
        lb=self.lb,  # Numpy array specifying lower bounds
        ub=self.ub,  # Numpy array specifying upper bounds
        n_init=self.n_init,  # Number of initial bounds from an Latin hypercube design
        max_evals = max_evals,  # Maximum number of evaluations
        trust_regions = trust_regions,
        batch_size=batch_size,  # How large batch size TuRBO uses
        verbose=self.verbose,  # Print information from each batch
            use_ard=use_ard,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=max_cholesky_size,  # When we switch from Cholesky to Lanczos
        n_training_steps=n_training_steps,  # Number of steps of ADAM to learn the hypers
        min_cuda=min_cuda,  # Run on the CPU for small datasets
        device=device,  # "cpu" or "cuda"
        dtype=dtype
    )