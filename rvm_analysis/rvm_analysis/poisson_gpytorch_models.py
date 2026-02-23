import gpytorch
import pyro.distributions
import torch
import pyro
from rvm_analysis.simulations import ConwayMaxwellPoisson, GeneralizedPoisson

class PVGPRegressionModel(gpytorch.models.ApproximateGP):
    """
    A class for fitting a Poisson Likelihood GPytorch model
    using variational inference.
    """
    def __init__(self, num_inducing=64, T_max=1, name_prefix="mixture_gp"):
        self.name_prefix = name_prefix

        # Define all the variational stuff
        inducing_points = torch.linspace(0, T_max, num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points,
            gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        )

        # Standard initializtation
        super().__init__(variational_strategy)

        # Mean, covar, likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module.constant = torch.log(torch.tensor(30))
        rbf_kernel = gpytorch.kernels.RBFKernel()
        periodic_kernel = gpytorch.kernels.PeriodicKernel()
        periodic_kernel.period_length = 300
        self.covar_module = gpytorch.kernels.ScaleKernel(periodic_kernel) # RBFKernel()
        # self.nu = gpytorch.priors.

    def forward(self, x):
        """ Computes the GP mean and covariance at the supplied times."""
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, x, y):
        """ Defines the approximate GP posterior."""
        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            pyro.sample(self.name_prefix + ".f(x)", function_dist)

    def model(self, x, y):
        """
        Needs to:
        - Compute the GP prior at x.
        - Convert GP function samples into
        scale function samples using the link function.
        - Sample from the observed distribution p(y|f). This
        is in place of the `Likelihood` at the high level interface.
        """
        pyro.module(self.name_prefix + ".gp", self)

        # Get p(f) - prior distribution of latent function
        function_dist = self.pyro_model(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)

            # Use the link function to convert GP samples into scale samples
            scale_samples = function_samples.exp()

            # Sample from observed distribution
            return pyro.sample(
                self.name_prefix + ".y",
                pyro.distributions.Poisson(scale_samples), #torch.tensor(1.1)
                obs=y
            )

class PEXPVGPRegressionModel(gpytorch.models.ApproximateGP):
    """
    A class for fitting a Poisson Likelihood GPytorch model
    using variational inference.
    """
    def __init__(self, num_inducing=64, T_max=1, name_prefix="mixture_gp"):
        self.name_prefix = name_prefix

        # Define all the variational stuff
        inducing_points = torch.linspace(0, T_max, num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points,
            gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        )

        # Standard initializtation
        super().__init__(variational_strategy)

        # Mean, covar, likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module.constant = torch.log(torch.tensor(30))
        rbf_kernel = gpytorch.kernels.RBFKernel()
        rbf_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.GreaterThan(torch.tensor(10000.0)))
        periodic_kernel = gpytorch.kernels.PeriodicKernel()
        # matern_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        # constant_kernel = gpytorch.kernels.ConstantKernel()
        periodic_kernel.period_length = 300
        # periodic_kernel.register_constraint("raw_lengthscale",gpytorch.constraints.GreaterThan(torch.tensor(0.1)))
        # periodic_kernel.register_constraint("raw_lengthscale")
        self.covar_module = gpytorch.kernels.ScaleKernel(
            periodic_kernel*rbf_kernel)# + constant_kernel#*gpytorch.kernels.MaternKernel(nu=1.5)#+ matern_kernel
            # outputscale_constraint=gpytorch.constraints.GreaterThan(torch.tensor(0.1))) # RBFKernel()
        
        # self.nu = gpytorch.priors.
        # Trainable diagonal noise
        # self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.tensor(0.1)))
        # self.register_constraint("raw_noise", gpytorch.constraints.Positive())

    # @property
    # def noise(self):
    #     return self.raw_noise_constraint.transform(self.raw_noise)
    def forward(self, x):
        """ Computes the GP mean and covariance at the supplied times."""
        mean = self.mean_module(x)
        covar = self.covar_module(x)#.add_diag(self.noise)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, x, y):
        """ Defines the approximate GP posterior."""
        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            pyro.sample(self.name_prefix + ".f(x)", function_dist)

    def model(self, x, y):
        """
        Needs to:
        - Compute the GP prior at x.
        - Convert GP function samples into
        scale function samples using the link function.
        - Sample from the observed distribution p(y|f). This
        is in place of the `Likelihood` at the high level interface.
        """
        pyro.module(self.name_prefix + ".gp", self)

        # Get p(f) - prior distribution of latent function
        function_dist = self.pyro_model(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)

            # Use the link function to convert GP samples into scale samples
            scale_samples = function_samples.exp()

            # Sample from observed distribution
            return pyro.sample(
                self.name_prefix + ".y",
                pyro.distributions.Poisson(scale_samples), #torch.tensor(1.1)
                obs=y
            )


class PDispVGPRegressionModel(gpytorch.models.ApproximateGP):
    """
    A class for fitting a Poisson Likelihood GPytorch model
    using variational inference.
    """
    def __init__(self, num_inducing=64, T_max=1, name_prefix="mixture_gp"):
        self.name_prefix = name_prefix

        # Define all the variational stuff
        inducing_points = torch.linspace(0, T_max, num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points,
            gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        )

        # Standard initializtation
        super().__init__(variational_strategy)

        # Mean, covar, likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module.constant = torch.log(torch.tensor(30))
        # rbf_kernel = gpytorch.kernels.RBFKernel()
        # rbf_kernel.lengthscale = 10
        periodic_kernel = gpytorch.kernels.PeriodicKernel()
        periodic_kernel.period_length = 300
        self.covar_module = gpytorch.kernels.ScaleKernel(
            periodic_kernel
            ) # RBFKernel()
        # self.nu = gpytorch.priors.
        # Trainable diagonal noise
        self.register_parameter(name="raw_dispersion", parameter=torch.nn.Parameter(torch.tensor(1.0)))
        self.register_constraint("raw_dispersion", gpytorch.constraints.Positive())

    @property
    def dispersion(self):
        return self.raw_dispersion_constraint.transform(self.raw_dispersion)
    def forward(self, x):
        """ Computes the GP mean and covariance at the supplied times."""
        mean = self.mean_module(x)
        covar = self.covar_module(x)#.add_diag(self.noise)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, x, y):
        """ Defines the approximate GP posterior."""
        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            pyro.sample(self.name_prefix + ".f(x)", function_dist)

    def model(self, x, y):
        """
        Needs to:
        - Compute the GP prior at x.
        - Convert GP function samples into
        scale function samples using the link function.
        - Sample from the observed distribution p(y|f). This
        is in place of the `Likelihood` at the high level interface.
        """
        pyro.module(self.name_prefix + ".gp", self)

        # Get p(f) - prior distribution of latent function
        function_dist = self.pyro_model(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)

            # Use the link function to convert GP samples into scale samples
            scale_samples = function_samples.exp()

            # Total count = dispersion, probs computed from mean & dispersion
            r = 1 / self.dispersion
            probs = r / (r + scale_samples)

            # Sample from observed distribution
            return pyro.sample(
                self.name_prefix + ".y",
                # GeneralizedPoisson(scale_samples,dispersion=self.dispersion), #torch.tensor(1.1)
                pyro.distributions.NegativeBinomial(total_count=r,probs=probs), #torch.tensor(1.1)
                obs=y
            )




def percentiles_from_samples(samples, percentiles=[0.05, 0.5, 0.95]):
    """
    Takes pyro samples and returns percentiles from them.
    Useful for plotting posterior fits to latent functions!!
    """
    num_samples = samples.size(0)
    samples = samples.sort(dim=0)[0]

    # Get samples corresponding to percentile
    percentile_samples = [samples[int(num_samples * percentile)] for percentile in percentiles]

    # Smooth the samples
    kernel = torch.full((1, 1, 5), fill_value=0.2)
    percentiles_samples = [
        torch.nn.functional.conv1d(percentile_sample.view(1, 1, -1), kernel, padding=2).view(-1)
        for percentile_sample in percentile_samples
    ]

    return percentile_samples