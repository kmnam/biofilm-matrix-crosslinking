"""
A simple implementation of a finite von Mises mixture model. 

Authors:
    Kee-Myoung Nam

Last updated:
    9/30/2025
"""
import numpy as np
from scipy.stats import vonmises
from scipy.special import i0, i0e, i1e, softmax
from scipy.optimize import root_scalar

########################################################################
class VonMisesMixture:
    """
    A finite von Mises mixture model. 
    """
    def __init__(self, n_components=2, rng=None):
        """
        Simple constructor. 

        Parameters
        ----------
        n_components : int
            Number of components in the mixture model. 
        rng : `numpy.random.Generator`
            Random number generator for initializing mixture components. 
        """
        self.n_components = n_components
        if rng is None:
            rng = np.random.default_rng(1234567890)

        # Randomly distribute means and initialize all concentrations to 1 
        self.means = rng.uniform(-np.pi, np.pi, size=n_components)
        self.kappa = np.ones(n_components, dtype=np.float64)
        self.weights = np.ones(n_components, dtype=np.float64) / n_components

    ####################################################################
    def log_likelihood(self, X, Z, weights=None, means=None, kappa=None,
                       hard=False):
        """
        Compute the complete-data log-likelihood for angles X and component
        assignments Z.

        Parameters
        ----------
        X : `numpy.ndarray`
            Observed angles. 
        Z : `numpy.ndarray`
            Component assignments for the angles in `X`.
        weights : `numpy.ndarray`
            Weights to use instead of `self.weights`.
        means : `numpy.ndarray`
            Means to use instead of `self.means`.
        kappa : `numpy.ndarray`
            Concentrations to use instead of `self.kappa`.
        hard : bool
            Use hard component assignments for log-likelihood calculation.

        Returns
        -------
        Complete-data log-likelihood. 
        """
        # Flatten X and Z
        X = X.reshape(-1)
        Z = Z.reshape(-1)
        n = X.shape[0]

        # If the weights, means, and concentrations were not specified, 
        # use the stored values 
        if weights is None:
            weights = self.weights
        if means is None:
            means = self.means
        if kappa is None:
            kappa = self.kappa

        L = 0
        if hard:    # Use hard assignments, if desired
            for i in range(n):
                j = Z[i]
                L += np.log(weights[j])
                L -= np.log(2 * np.pi * i0(kappa[j]))
                L += kappa[j] * np.cos(X[i] - means[j])
        else:       # Otherwise, use soft assignments
            gamma = self.Estep(X)
            for i in range(n):
                for j in range(self.n_components):
                    L += gamma[i, j] * (
                        np.log(weights[j])
                        - np.log(2 * np.pi * i0(kappa[j]))
                        + kappa[j] * np.cos(X[i] - means[j])
                    )

        return L

    ####################################################################
    def Estep(self, X):
        """
        Perform the E-step. 

        Parameters
        ----------
        X : `numpy.ndarray`
            Observed angles. 

        Returns
        -------
        Array of probabilities, `gamma[i, j]`, that `X[i]` belongs to 
        component `j`. 
        """
        n = X.shape[0]
        gamma = np.zeros((n, self.n_components), dtype=np.float64)
        for i in range(n):
            # Calculate the pdf in log space using softmax 
            factors = np.zeros(self.n_components, dtype=np.float64)
            for j in range(self.n_components):
                factors[j] = np.log(self.weights[j])
                factors[j] -= np.log(2 * np.pi * i0(self.kappa[j]))
                factors[j] += self.kappa[j] * np.cos(X[i] - self.means[j])
            gamma[i, :] = softmax(factors - np.max(factors))

        return gamma

    ####################################################################
    def Mstep(self, X, Z, gamma, max_kappa=1000.0):
        """
        Perform the M-step. 

        Parameters
        ----------
        X : `numpy.ndarray`
            Observed angles. 
        Z : `numpy.ndarray`
            Component assignments for the angles in `X`.
        gamma : `numpy.ndarray`
            Component assignment probabilities calculated in the E-step. 
        max_kappa : float
            Maximum concentration value. 

        Returns
        -------
        Weights, means, and concentrations that maximize the expected log-
        likelihood function calculated in the E-step. 
        """
        n = X.shape[0]
        new_weights = np.zeros(self.n_components, dtype=np.float64)
        for j in range(self.n_components):
            new_weights[j] = gamma[:, j].mean()

        # Estimate the mean angle and concentration parameter of each component
        # as the weighted MLE
        new_means = np.zeros(self.n_components, dtype=np.float64)
        new_kappa = np.zeros(self.n_components, dtype=np.float64)
        for j in range(self.n_components):
            cos_sum = (gamma[:, j] * np.cos(X)).sum()
            sin_sum = (gamma[:, j] * np.sin(X)).sum()
            new_means[j] = np.arctan2(sin_sum, cos_sum)
            R = np.sqrt(cos_sum ** 2 + sin_sum ** 2) / gamma[:, j].sum()
            if R > 1 - 1e-6:
                new_kappa[j] = max_kappa
            else:
                result = root_scalar(
                    lambda kappa: R - i1e(kappa) / i0e(kappa), method='newton',
                    x0=R * (2 - R * R) / (1 - R * R)
                )
                new_kappa[j] = result.root

        return new_weights, new_means, new_kappa

    ####################################################################
    def fit(self, X, rng, tol=1e-5, init_weights=None, init_means=None,
            init_kappa=None, verbose=False):
        """
        Run the EM algorithm to fit the mixture model to the given set of 
        angles. 

        Parameters
        ----------
        X : `numpy.ndarray`
            Observed angles. 
        rng : `numpy.random.Generator`
            Random number generator for initializing mixture components. 
        tol : float
            Terminate when the change in the log-likelihood is below this
            value.
        init_weights : `numpy.ndarray`
            Initial weights. 
        init_means : `numpy.ndarray`
            Initial means. 
        init_kappa : `numpy.ndarray`
            Initial concentrations.
        verbose : bool
            If True, print intermittent output. 
        """
        # Initialize parameters 
        n = X.shape[0]
        if init_weights is None:
            self.weights = np.ones(self.n_components, dtype=np.float64) / self.n_components
        else:
            self.weights = np.array(init_weights)
        if init_means is None:
            self.means = rng.uniform(-np.pi, np.pi, size=self.n_components)
        else:
            self.means = np.array(init_means)
        if init_kappa is None:
            self.kappa = np.ones(self.n_components, dtype=np.float64)
        else:
            self.kappa = np.array(init_kappa)

        # Initialize component assignments 
        Z = np.zeros(n, dtype=np.int64)
        for i in range(n):
            delta = X[i] - self.means
            Z[i] = np.argmin(np.abs((delta + np.pi) % (2 * np.pi) - np.pi))

        curr_log_likelihood = self.log_likelihood(X, Z)
        update = np.inf
        while update > tol:
            # Perform E and M steps 
            gamma = self.Estep(X)
            new_weights, new_means, new_kappa = self.Mstep(X, Z, gamma)

            # Compute new log-likelihood
            new_log_likelihood = self.log_likelihood(
                X, Z, weights=new_weights, means=new_means, kappa=new_kappa
            )
            if verbose:
                print('Log-likelihood = {:.10f}'.format(new_log_likelihood))
            update = new_log_likelihood - curr_log_likelihood

            # Update model parameters and component assignments
            curr_log_likelihood = new_log_likelihood
            self.weights = new_weights
            self.means = new_means
            self.kappa = new_kappa
            for i in range(n):
                delta = X[i] - self.means
                Z[i] = np.argmin(np.abs((delta + np.pi) % (2 * np.pi) - np.pi))

    ####################################################################
    def plot(self, ax):
        """
        Plot the PDF corresponding to the mixture model. 

        Parameters
        ----------
        ax : `matplotlib.pyplot.Axes`
            Input axes. 

        Returns
        -------
        Updated axes. 
        """
        x = np.linspace(-np.pi, np.pi, 100)
        ax.plot(
            x,
            sum(
                self.weights[j] * vonmises.pdf(x, loc=self.means[j], kappa=self.kappa[j])
                for j in range(self.n_components)
            )
        )

        return ax

    ####################################################################
    def predict(self, X):
        """
        Assign the most probable mixture component corresponding to each angle
        in the given array. 

        Parameters
        ----------
        X : `numpy.ndarray`
            Observed angles. 

        Returns
        -------
        Array indicating the most probable mixture component for each angle.
        """
        return np.argmax(self.Estep(X), axis=1)

