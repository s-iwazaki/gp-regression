import numpy as np

class Rbf:
    def __init__(self, xdim, length_scale=1.0, variance=1.0):
        self.xdim = xdim
        self.length_scale = length_scale
        self.variance = variance

    def K(self, x1, x2, diag=False):
        """calculate kernel matrix
        
        Parameters
        ----------
        x1 : 2d-ndarray
            Input data 1
        x2 : 2d-ndarray
            Input data 2
        diag : bool, optional
            Caluculate only diagonal elements of kernel matrix
        
        Returns
        -------
        1d or 2d-ndarray  
            Kernel matrix(or its diagonal elements)
        """
        if diag:
            return self.variance * np.exp(-np.sum((x1 - 
                                    x2)**2, axis=1) / (2 * self.length_scale ** 2))
        else:
            return self.variance * np.exp(-np.sum((x1[:, np.newaxis, :] - 
                                    x2[np.newaxis, :, :])**2, axis=2) / (2 * self.length_scale ** 2))

class Matern32:
    def __init__(self, xdim, length_scale=1.0, variance=1.0):
        self.xdim = xdim
        self.length_scale = length_scale
        self.variance = variance
    
    def K(self, x1, x2, diag=False):
        """calculate kernel matrix
        
        Parameters
        ----------
        x1 : 2d-ndarray
            Input data 1
        x2 : 2d-ndarray
            Input data 2
        diag : bool, optional
            Caluculate only diagonal elements of kernel matrix
        
        Returns
        -------
        1d or 2d-ndarray  
            Kernel matrix(or its diagonal elements)
        """
        if diag:
            dist = np.sqrt(np.sum((x1 - x2)**2, axis=1))
            return self.variance * (1 + np.sqrt(3)*dist / self.length_scale) * np.exp(-np.sqrt(3) * dist / self.length_scale)
        else:
            dist = np.sqrt(np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :])**2, axis=2))
            return self.variance * (1 + np.sqrt(3)*dist / self.length_scale) * np.exp(-np.sqrt(3) * dist / self.length_scale)
        
class Matern52:
    def __init__(self, xdim, length_scale=1.0, variance=1.0):
        self.xdim = xdim
        self.length_scale = length_scale
        self.variance = variance
    
    def K(self, x1, x2, diag=False):
        """calculate kernel matrix
        
        Parameters
        ----------
        x1 : 2d-ndarray
            Input data 1
        x2 : 2d-ndarray
            Input data 2
        diag : bool, optional
            Caluculate only diagonal elements of kernel matrix
        
        Returns
        -------
        1d or 2d-ndarray  
            Kernel matrix(or its diagonal elements)
        """
        if diag:
            dist = np.sqrt(np.sum((x1 - x2)**2, axis=1))
            return self.variance * (1 + np.sqrt(5)*dist / self.length_scale + 5 * dist ** 2 / (3 * self.length_scale ** 2)) \
                     * np.exp(-np.sqrt(5) * dist / self.length_scale)
        else:
            dist = np.sqrt(np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :])**2, axis=2))
            return self.variance * (1 + np.sqrt(5)*dist / self.length_scale + 5 * dist ** 2 / (3 * self.length_scale ** 2)) \
                     * np.exp(-np.sqrt(5) * dist / self.length_scale)

    class Ard_se:
        def __init__(self, xdim, variance=1.0):
            self.xdim = xdim
            self.length_scale = np.empty([xdim])
            self.variance = variance

        def K(self, x1, x2, diag=False):
            """calculate kernel matrix
            
            Parameters
            ----------
            x1 : 2d-ndarray
                Input data 1
            x2 : 2d-ndarray
                Input data 2
            diag : bool, optional
                Caluculate only diagonal elements of kernel matrix
            
            Returns
            -------
            1d or 2d-ndarray  
                Kernel matrix(or its diagonal elements)
            """
            if diag:
                return self.variance * np.exp(-np.sum((x1 - 
                                        x2)**2 / (2 * self.length_scale ** 2), axis=1))
            else:
                return self.variance * np.exp(-np.sum((x1[:, np.newaxis, :] - 
                                        x2[np.newaxis, :, :])**2 / (2 * self.length_scale[np.newaxis, np.newaxis, :] ** 2), axis=2))

