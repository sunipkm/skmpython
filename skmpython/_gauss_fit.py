from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from . import staticvars

class GaussFit:
    """Fit N Gaussians with a 5 degree polynomial background.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, *p0, plot: bool = True, **kwargs):
        """Initialize a Gauss fit object.

        Args:
            x (np.ndarray): X data.
            y (np.ndarray): Y data.
            p0 (list, np.ndarray): Fit parameters initial guess. p0 must have the length
            of 5 + N * 3, where N is the number of Gaussians to be fit. The first 5
            parameters are the polynomial coefficients of the background function,
            the Gaussian parameters are ordered as [center, amplitude, width].
            plot (bool, optional): Plot progress of the fit. Defaults to True.
            kwargs: Additional arguments passed to the internal plt.subplots() call.
        """
        if len(p0) == 1:
            p0 = p0[0]
        if (plot):
            fig, ax = plt.subplots(1, 1, **kwargs)
            self._fig = fig
            self._ax = ax
            self._orig, = self._ax.plot(x, y, color = 'r')
            self._bck, = self._ax.plot(x, GaussFit.background(x, p0), color = 'k')
            self._sig = []
            for i in range((len(p0) - 5) // 3):
                sig, = self._ax.plot(x, GaussFit.gaussian_w_bg(x, i, p0))
                self._sig.append(sig)
            self._ax.set_xlim(x.min(), x.max())
        self._plot = plot
        self._iteration = 0
        self._param = p0
        self._x = x
        self._y = y
        self._last_res = None
        self._n_gaussians = (len(p0) - 5) // 3
        self._plot_every = 0
        if plot:
            plt.ion()
            plt.show()

    def _fit_func(self, x, *params):
        y = GaussFit.full_field(x, params)
        self._update(x, *params)
        return y
    
    def _update(self, x, *params):
        if self._plot:
            if self._plot_every > 0 and (self._iteration % self._plot_every):
                pass
            else:
                self._bck.set_ydata(GaussFit.background(x, params))
                for idx, sig in enumerate(self._sig):
                    sig.set_ydata(GaussFit.gaussian_w_bg(x, idx, params))
                data = GaussFit.full_field(x, params)
                self._ax.set_ylim(np.min([data.min(), self._y.min()]), np.max([data.max(), self._y.max()]))    
                res = np.sqrt(np.sum((self._y - data)**2))
                if self._last_res is None:
                    self._last_res = res
                dres = res - self._last_res
                self._last_res = res
                self._fig.suptitle('Gaussians: %d, Iteration: %d, Residual: %.3e, Improvement: %.3e'%(self._n_gaussians, self._iteration, res, dres))
                self._fig.canvas.draw()
                self._fig.canvas.flush_events()
        self._iteration += 1

        if self._interval > 0 and self._plot:
            plt.pause(self._interval)

    def run(self, interval: float=1/24, plot_every: int = 0, **kwargs)->tuple(np.ndarray, np.ndarray):
        """Run the fit routine. Read documentation for scipy.optimize.curve_fit to see what additional named parameters can be passed through kwargs (x, y, p0 are internally passed).

        Args:
            interval (float, optional): Interval between plot frame updates in seconds. Defaults to 1/24.
            plot_every (int, optional): Plot every Nth frame (0 to plot every frame, default.)
            kwargs: Named arguments passed to scipy.optimize.curve_fit. DO NOT pass f, xdata, ydata, p0.

        Returns:
            tuple(np.ndarray, np.ndarray): popt, pcov from scipy.optimize.curve_fit.
        """
        self._interval = interval
        self._plot_every = plot_every
        popt, pcov = curve_fit(self._fit_func, self._x, self._y, p0=self._param, **kwargs)
        if self._plot:
            data = GaussFit.full_field(self._x, popt)
            res = np.sqrt(np.sum((self._y - data)**2))
            self._fig.suptitle('Gaussians: %d, Iteration: %d, Residual: %.3e\nOptimization complete.'%(self._n_gaussians, self._iteration, res))
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.ioff()
        return popt, pcov

    @staticmethod
    def full_field(x: np.ndarray, *params)->np.ndarray:
        """Evaluate background + gaussians from the guess/fit parameters.

        Args:
            x (np.ndarray): X data.
            params (list): Background + Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        y = np.zeros_like(np.asarray(x, dtype = float))
        if len(params) == 1:
            params = params[0]
        # print(params)
        for i in range(5): # 5 degree polynomial
            # print((params[i] * (x**i)).shape)
            y += float(params[i]) * (x**i)
        # print(y)
        for i in range(5, len(params), 3):
            ctr = params[i]
            amp = params[i + 1]
            wid = params[i + 2]
            dy = (amp * np.exp(-((x - ctr)/wid)**2))
            y += dy
        return y

    @staticmethod
    def background(x: np.ndarray, *params)->np.ndarray:
        """Evaluate background from the guess/fit parameters.

        Args:
            x (np.ndarray): X data.
            params (list): Background + Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        y = np.zeros_like(x)
        if len(params) == 1:
            params = params[0]
        for i in range(5):
            y += float(params[i]) * (x ** i)
        return y

    @staticmethod
    def gaussian_w_bg(x: np.ndarray, id: int, *params)->np.ndarray:
        """Evaluate a gaussian, with the background, from the guess/fit parameters.

        Args:
            x (np.ndarray): X data.
            id (int): ID of the Gaussian. Must be between [0, (len(params) - 5) // 3].
            params (list): Background + Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        y = np.zeros_like(x)
        if len(params) == 1:
            params = params[0]
        for i in range(5):
            y += float(params[i]) * (x ** i)
        base = 5 + (3 * id)
        ctr = params[base]
        amp = params[base + 1]
        wid = params[base + 2]
        y += (amp * np.exp(-((x - ctr)/wid)**2))
        return y

    @staticmethod
    def gaussian_wo_bg(x: np.ndarray, id: int, *params)->np.ndarray:
        """Evaluate a gaussian, without the background, from the guess/fit parameters.

        Args:
            x (np.ndarray): X data.
            id (int): ID of the Gaussian. Must be between [0, (len(params) - 5) // 3].
            params (list): Background + Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        y = np.zeros_like(x)
        base = 5 + (3 * id)
        ctr = params[base]
        amp = params[base + 1]
        wid = params[base + 2]
        y += (amp * np.exp(-((x - ctr)/wid)**2))
        return y

    @staticmethod
    def gaussian(x: np.ndarray, H: float, A: float, x0: float, sigma: float)->np.ndarray:
        """Evaluate a Gaussian.

        Args:
            x (np.ndarray): X data.
            H (float): Y Offset (constant).
            A (float): Amplitude.
            x0 (float): Mean.
            sigma (float): Width (2*sigma^2)

        Returns:
            np.ndarray: Evaluated Gaussian.
        """
        return H + A * np.exp(-((x - x0)/sigma)**2)