import numpy as np

class GP:
    def __init__(self, x, y, kern, noise_var=1e-2, seed=0):
        """Gaussian Process
        
        Parameters
        ----------
        x : 2d-ndarray
            Input data X
        y : 1d-ndarray
            Output data y
        kern : kernel class object
            Kernel class (see kern.py)
        noise_var : float, optional
            Observation noise's variance, by default is 1e-2
        seed : int, optional
            Random number generator's seed, by default is 0
        """

        self.noise_var = noise_var

        self.x = x
        self.y = y
        self.n_data = x.shape[0]
        self.n_dim = x.shape[1]
        self.kern = kern
        
        self.K = self.kern.K(x, x)
        self.K_y = self.K + noise_var * np.eye(self.n_dim)
        self.inv_K = None 
        self.inv_K_y = None

        L = np.linalg.cholesky(self.K)
        inv_L = np.linalg.inv(L)
        self.inv_K = np.matmul(inv_L.T, inv_L)
        
        L = np.linalg.cholesky(self.K + self.noise_var * np.eye(x.shape[0]))
        inv_L = np.linalg.inv(L)
        self.inv_K_y = np.matmul(inv_L.T, inv_L)
        self.rand_gen = np.random.RandomState(seed)
    
    def prior_sampling(self, x):
        mu = np.zeros(x.shape[0])
        cov = self.kern.K(x, x)
        return self.rand_gen.multivariate_normal(mu, cov)

    def predict_f(self, x, full_var=False):
        """Return predict mean and variance of unobserved f
        
        Parameters
        ----------
        x : 2d-ndarray
            Input data
        full_var : bool, optional
            If true, return full covariance matrix, by default is False
        
        Returns
        -------
        (mean, variance)
            Predict mean and variance of f
        """
        k = self.kern.K(x, self.x)
        mean = np.matmul(np.matmul(k, self.inv_K), self.y[:, np.newaxis]).reshape([-1])
        if full_var:
            var = self.kern.K(x, x) - np.matmul(np.matmul(k, self.inv_K_y), k.T)
        else:
            var = self.kern.K(x, x, diag=True) - np.sum(np.matmul(k, self.inv_K_y) * k, axis=1)
        return mean, var
    
    def predict_fvar(self, x, full_var=False):
        """Return predict variance of unobserved f
        
        Parameters
        ----------
        x : 2d-ndarray
            Input data
        full_var : bool, optional
            If true, return full covariance matrix, by default is False 
        
        Returns
        -------
        1d or 2d-ndarary
            Predict mean and variance of f
        """
        if full_var:
            k = self.kern.K(x, self.x)
            return self.kern.K(x, x) - np.matmul(np.matmul(k, self.inv_K_y), k.T)
        else:
            k = self.kern.K(x, self.x)
            return self.kern.K(x, x, diag=True) - np.sum(np.matmul(k, self.inv_K_y) * k, axis=1)
    
    def predict(self, x, full_var=False):
        """Return predict mean and variance
        
        Parameters
        ----------
        x : 2d-ndarray
            Input data
        full_var : bool, optional
            If True, return full covariance matrix, by default is False
        Returns
        -------
        (mean, varaince)
            Predict mean and variance
        """
        k = self.kern.K(x, self.x)
        mean = np.matmul(np.matmul(k, self.inv_K), self.y[:, np.newaxis]).reshape([-1])
        
        if full_var:
            var = self.kern.K(x, x) - np.matmul(np.matmul(k, self.inv_K_y), k.T) + np.eye(x.shape[0]) * self.noise_var
        else:
            var = self.kern.K(x, x, diag=True) - np.sum(np.matmul(k, self.inv_K_y) * k, axis=1) + self.noise_var
        return mean, var

    def predict_mean(self, x):
        """Return predict mean
        
        Parameters
        ----------
        x : 2d-ndarray
            Input data
        
        Returns
        -------
        1d-ndarray
            Predict mean
        """
        k = self.kern.K(x, self.x)
        return np.matmul(np.matmul(k, self.inv_K_y), self.y[:, np.newaxis]).reshape([-1])

    def predict_var(self, x, full_var=False):
        """Return predict variance
        
        Parameters
        ----------
        x : 2d-ndarray
            Input data
        full_var : bool, optional
            If True, return full covariance matrix, by default False
        
        Returns
        -------
        1d or 2d-ndarray
            predict variance
        """
        if full_var:
            k = self.kern.K(x, self.x)
            return self.kern.K(x, x) - np.matmul(np.matmul(k, self.inv_K_y), k.T) + np.eye(x.shape[0]) * self.noise_var
        else:
            k = self.kern.K(x, self.x)
            return self.kern.K(x, x, diag=True) - np.sum(np.matmul(k, self.inv_K_y) * k, axis=1) + self.noise_var
    
    def predict_cov(self, x1, x2):
        """Return predict covariance
        
        Parameters
        ----------
        x1 : 2d-ndarray
            Input data 1
        x2 : 2d-ndarray
            Input data 2
        
        
        Returns
        -------
        2d-ndarray
            predict covariance
        """
        k1 = self.kern.K(x1, self.x)
        k2 = self.kern.K(self.x, x2)
        return self.kern.K(x1, x2) - np.matmul(np.matmul(k1, self.inv_K_y), k2)