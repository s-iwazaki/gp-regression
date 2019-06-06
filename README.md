# gp-regression
Gaussian process regression experiment
- Compare following kernel
    - Gauss(Squared Exponential)
    - Matern3/2
    - Matern5/2
- Compare piror sample
- Compare predictive mean and variance

## Prior sampling experiment
- Kernel hyperparameter
    - lengthscale = 0.4
    - variance = 16
![prior sample(gauss)](https://github.com/s-iwazaki/image/gp-prior-gauss.pdf)
![prior sample(matern3/2)](https://github.com/s-iwazaki/image/gp-prior-matern32.pdf)
![prior sample(matern5/2)](https://github.com/s-iwazaki/image/gp-prior-matern52.pdf)

## Posterior experiment
### Predictive mean and variance
![posterior (gauss)](https://github.com/s-iwazaki/image/gp-posterior-gauss.pdf)
![posterior (matern3/2)](https://github.com/s-iwazaki/image/gp-posterior-matern32.pdf)
![posterior (matern5/2)](https://github.com/s-iwazaki/image/gp-posterior-matern52.pdf)