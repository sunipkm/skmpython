from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod
import gc

from warnings import warn

warn('The module GaussFit is deprecated.', DeprecationWarning, stacklevel=2)

class BaseGaussFuncs:
    """Abstract class listing methods that a base class for GaussFit must provide.
    """
    @staticmethod
    @abstractmethod
    def full_field(x: np.ndarray, *params) -> np.ndarray:
        """Calculate full fit result (background + gaussians)

        Args:
            x (np.ndarray): X data.
            params (list): Background and Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        pass

    @staticmethod
    @abstractmethod
    def background(x: np.ndarray, *params) -> np.ndarray:
        """Calculate background only.

        Args:
            x (np.ndarray): X data.
            params (list): Background and Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        pass

    @staticmethod
    @abstractmethod
    def gaussian_w_bg(x: np.ndarray, id: int, *params) -> np.ndarray:
        """Calculate fit result (background + single gaussian)

        Args:
            x (np.ndarray): X data.
            id (int): ID of the Gaussian [[0, N), where N is the total number of Gaussians.]
            params (list): Background and Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        pass

    @staticmethod
    @abstractmethod
    def gaussian_wo_bg(x: np.ndarray, id: int, *params) -> np.ndarray:
        """Calculate single Gaussian (without background)

        Args:
            x (np.ndarray): X data.
            id (int): ID of the Gaussian [[0, N), where N is the total number of Gaussians.]
            params (list): Background and Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        pass

    @staticmethod
    def gaussian(x: np.ndarray, H: float, A: float, x0: float, sigma: float) -> np.ndarray:
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

    @staticmethod
    @abstractmethod
    def n_gaussians(*params) -> int:
        """Get number of Gaussians inferred from the parameters.

        Returns:
            int: Number of Gaussians.
        """
        pass

    @staticmethod
    @abstractmethod
    def n_background_params() -> int:
        """Get number of background parameters.

        Returns:
            int: Number of background parameters.
        """
        pass


class GaussFuncsBasic(BaseGaussFuncs):
    """This class accepts parameters of length 5 + 3 * N, where
    N is the number of Gaussians to be fit. first 5 parameters are 
    a0 to a4. The background is evaluated as: 
    background = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4.
    For each Gaussian, the parameters to be supplied are [C, A, W]. 
    The Gaussians are evaluated as A*exp(-((x-C)/W)^2).

    WARNING: Very basic fit function. With large (or small) values of x,
    the polynomial evaluated can be very large. Use if x is around 0.
    GaussFuncs is the preferred class.
    """

    def __init__(self):
        warn('GaussFuncsBasic class is deprecated. Please use GenericFitFunc and GenericFitManager.', category=DeprecationWarning, stacklevel=2)
        pass

    @staticmethod
    def full_field(x: np.ndarray, *params) -> np.ndarray:
        """Evaluate background + gaussians from the guess/fit parameters.

        Args:
            x (np.ndarray): X data.
            params (list): Background + Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        if len(params) == 1:
            params = params[0]
        y = GaussFuncsBasic.background(x, params)
        for i in range(5, len(params), 3):
            ctr = params[i]
            amp = params[i + 1]
            wid = params[i + 2]
            dy = (amp * np.exp(-((x - ctr)/wid)**2))
            y += dy
        return y

    @staticmethod
    def background(x: np.ndarray, *params) -> np.ndarray:
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
    def gaussian_w_bg(x: np.ndarray, id: int, *params) -> np.ndarray:
        """Evaluate a gaussian, with the background, from the guess/fit parameters.

        Args:
            x (np.ndarray): X data.
            id (int): ID of the Gaussian. Must be between [0, (len(params) - 5) // 3].
            params (list): Background + Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        if len(params) == 1:
            params = params[0]
        y = GaussFuncsBasic.background(x, params)
        base = 5 + (3 * id)
        ctr = params[base]
        amp = params[base + 1]
        wid = params[base + 2]
        y += (amp * np.exp(-((x - ctr)/wid)**2))
        return y

    @staticmethod
    def gaussian_wo_bg(x: np.ndarray, id: int, *params) -> np.ndarray:
        """Evaluate a gaussian, without the background, from the guess/fit parameters.

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
        base = 5 + (3 * id)
        ctr = params[base]
        amp = params[base + 1]
        wid = params[base + 2]
        y += (amp * np.exp(-((x - ctr)/wid)**2))
        return y

    @staticmethod
    def n_gaussians(*params):
        """Get number of Gaussians inferred from the parameters.

        Returns:
            int: Number of Gaussians.
        """
        if len(params) == 1:
            params = params[0]
        return (len(params) - 5) // 3

    @staticmethod
    def n_background_params() -> int:
        """Get number of background parameters.

        Returns:
            int: Number of background parameters.
        """
        return 5


class GaussFuncs(BaseGaussFuncs):
    """This class accepts parameters of length 6 + 3 * N, where
    N is the number of Gaussians to be fit. The first parameter
    is x0, and the next 5 parameters are a0 to a4.
    The background is evaluated as: 
    background = a0 + a1(x-x0) + a2(x-x0)^2 + a3(x-x0)^3 + a4(x-x0)^4.
    For each Gaussian, the parameters to be supplied are [C, A, W]. 
    The Gaussians are evaluated as A*exp(-((x-C)/W)^2).

    This is the preferred function suite used to fit Gaussians.
    """
    @staticmethod
    def full_field(x: np.ndarray, *params) -> np.ndarray:
        """Evaluate background + gaussians from the guess/fit parameters.

        Args:
            x (np.ndarray): X data.
            params (list): Background + Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        if len(params) == 1:
            params = params[0]
        y = GaussFuncs.background(x, params)
        for i in range(6, len(params), 3):
            ctr = params[i]
            amp = params[i + 1]
            wid = params[i + 2]
            dy = (amp * np.exp(-((x - ctr)/wid)**2))
            y += dy
        return y

    @staticmethod
    def background(x: np.ndarray, *params) -> np.ndarray:
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
        x0 = params[0]
        for i in range(5):
            y += float(params[i+1])*((x-x0)**i)
        return y

    @staticmethod
    def gaussian_w_bg(x: np.ndarray, id: int, *params) -> np.ndarray:
        """Evaluate a gaussian, with the background, from the guess/fit parameters.

        Args:
            x (np.ndarray): X data.
            id (int): ID of the Gaussian. Must be between [0, (len(params) - 5) // 3].
            params (list): Background + Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        if len(params) == 1:
            params = params[0]
        y = GaussFuncs.background(x, params)
        base = 6 + (3 * id)
        ctr = params[base]
        amp = params[base + 1]
        wid = params[base + 2]
        y += (amp * np.exp(-((x - ctr)/wid)**2))
        return y

    @staticmethod
    def gaussian_wo_bg(x: np.ndarray, id: int, *params) -> np.ndarray:
        """Evaluate a gaussian, without the background, from the guess/fit parameters.

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
        base = 6 + (3 * id)
        ctr = params[base]
        amp = params[base + 1]
        wid = params[base + 2]
        y += (amp * np.exp(-((x - ctr)/wid)**2))
        return y

    @staticmethod
    def n_gaussians(*params):
        """Get number of Gaussians inferred from the parameters.

        Returns:
            int: Number of Gaussians.
        """
        if len(params) == 1:
            params = params[0]
        return (len(params) - 6) // 3

    @staticmethod
    def n_background_params() -> int:
        """Get number of background parameters.

        Returns:
            int: Number of background parameters.
        """
        return 6


class GaussFuncsExtS(BaseGaussFuncs):
    """This class accepts parameters of length 9 + 3 * N, where
    N is the number of Gaussians to be fit. The first 4 parameters
    are the x1 to x4, and the next 5 parameters are a0 to a4.
    The background is evaluated as: 
    background = a0 + a1(x-x1) + a2(x-x2)^2 + a3(x-x3)^3 + a4(x-x4)^4.
    For each Gaussian, the parameters to be supplied are [C, A, W]. 
    The Gaussians are evaluated as A*exp(-((x-C)/W)^2).

    WARNING: Use this class only in case of very stubborn fits.
    """
    @staticmethod
    def full_field(x: np.ndarray, *params) -> np.ndarray:
        """Evaluate background + gaussians from the guess/fit parameters.

        Args:
            x (np.ndarray): X data.
            params (list): Background + Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        if len(params) == 1:
            params = params[0]
        y = GaussFuncsExtS.background(x, params)
        for i in range(9, len(params), 3):
            ctr = params[i]
            amp = params[i + 1]
            wid = params[i + 2]
            dy = (amp * np.exp(-((x - ctr)/wid)**2))
            y += dy
        return y

    @staticmethod
    def background(x: np.ndarray, *params) -> np.ndarray:
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
        x1 = params[0]
        x2 = params[1]
        x3 = params[2]
        x4 = params[3]
        y += float(params[4])
        y += float(params[5])*(x-x1)
        y += float(params[6])*(x-x2)**2
        y += float(params[7])*(x-x3)**3
        y += float(params[8])*(x-x4)**4
        return y

    @staticmethod
    def gaussian_w_bg(x: np.ndarray, id: int, *params) -> np.ndarray:
        """Evaluate a gaussian, with the background, from the guess/fit parameters.

        Args:
            x (np.ndarray): X data.
            id (int): ID of the Gaussian. Must be between [0, (len(params) - 5) // 3].
            params (list): Background + Gaussian parameters.

        Returns:
            np.ndarray: Y data.
        """
        if len(params) == 1:
            params = params[0]
        y = GaussFuncsExtS.background(x, params)
        base = 9 + (3 * id)
        ctr = params[base]
        amp = params[base + 1]
        wid = params[base + 2]
        y += (amp * np.exp(-((x - ctr)/wid)**2))
        return y

    @staticmethod
    def gaussian_wo_bg(x: np.ndarray, id: int, *params) -> np.ndarray:
        """Evaluate a gaussian, without the background, from the guess/fit parameters.

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
        base = 9 + (3 * id)
        ctr = params[base]
        amp = params[base + 1]
        wid = params[base + 2]
        y += (amp * np.exp(-((x - ctr)/wid)**2))
        return y

    @staticmethod
    def n_gaussians(*params):
        """Get number of Gaussians inferred from the parameters.

        Returns:
            int: Number of Gaussians.
        """
        if len(params) == 1:
            params = params[0]
        return (len(params) - 9) // 3

    @staticmethod
    def n_background_params() -> int:
        """Get number of background parameters.

        Returns:
            int: Number of background parameters.
        """
        return 9


class GaussFit:
    """Fit N Gaussians with a polynomial background using the GaussFuncs* class.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, p0: tuple | list | np.ndarray, *, baseclass: BaseGaussFuncs = GaussFuncs(), plot: bool = True, figure_title: str = None, window_title: str = None, **kwargs):
        """Initialize a Gauss fit object.

        Args:
            x (np.ndarray): X data.
            y (np.ndarray): Y data.
            p0 (tuple | list | np.ndarray): Fit parameters initial guess. p0 must have the length
            compatible with the GaussFuncs* class.
            baseclass (GaussFuncs*(BaseGaussFuncs), optional): Function set to evaluate background and gaussians. Defaults to GaussFuncs().
            plot (bool, optional): Plot progress of the fit. Defaults to True.
            figure_title (str, optional): Progress figure title. Defaults to None.
            window_title (str, optional): Progress window title. Defaults to None.
            kwargs: Additional arguments passed to the internal plt.subplots() call.

        Raises:
            TypeError: Base function class is not of GaussFuncs* type.
        """
        warn('GaussFit class is deprecated. Please use GenericFitFunc and GenericFitManager.', category=DeprecationWarning, stacklevel=2)
        if not isinstance(baseclass, BaseGaussFuncs):
            raise TypeError('Base class must be of GaussFuncs* type.')
        self._baseclass = baseclass
        self._n_gaussians = self._baseclass.n_gaussians(p0)
        if (plot):
            fig, ax = plt.subplots(1, 1, **kwargs)
            self._fig = fig
            if figure_title is not None and len(figure_title) > 0:
                self._fig.suptitle(figure_title)
            if window_title is not None and len(window_title) > 0:
                self._fig.canvas.manager.set_window_title(window_title)
            self._ax = ax
            self._orig, = self._ax.plot(x, y, color='r')
            self._bck, = self._ax.plot(
                x, self._baseclass.background(x, p0), color='k')
            self._sig = []
            for i in range(self._n_gaussians):
                sig, = self._ax.plot(
                    x, self._baseclass.gaussian_w_bg(x, i, p0))
                self._sig.append(sig)
            self._ax.set_xlim(x.min(), x.max())
        self._plot = plot
        self._iteration = 0
        self._param = p0
        self._x = x
        self._y = y
        self._last_res = None
        self._plot_every = 0
        if plot:
            plt.ion()
            plt.show()

    @property
    def param(self) -> tuple:
        """Get the latest fit parameters.

        Returns:
            tuple: Fit parameters.
        """
        return tuple(self._param)

    def _fit_func(self, x, *params):
        y = self._baseclass.full_field(x, params)
        self._update(x, *params)
        self._param = params
        return y

    def _update(self, x, *params):
        if self._plot:
            if self._plot_every > 0 and (self._iteration % self._plot_every):
                pass
            else:
                bck = self._baseclass.background(x, params)
                self._bck.set_ydata(bck)
                for idx, sig in enumerate(self._sig):
                    gaus = self._baseclass.gaussian_wo_bg(x, idx, params)
                    gaus += bck
                    sig.set_ydata(gaus)
                data = self._baseclass.full_field(x, params)
                self._ax.set_ylim(np.min([data.min(), self._y.min()]), np.max(
                    [data.max(), self._y.max()]))
                res = np.sqrt(np.sum((self._y - data)**2))
                if self._last_res is None:
                    self._last_res = res
                dres = res - self._last_res
                self._last_res = res
                self._ax.set_title('Gaussians: %d, Iteration: %d, Residual: %.3e, Delta: %.3e' % (
                    self._n_gaussians, self._iteration, res, dres))
                self._fig.canvas.draw()
                self._fig.canvas.flush_events()
        self._iteration += 1

        if self._interval > 0 and self._plot:
            plt.pause(self._interval)

    def run(self, *, plot_every: int = 0, interval: float = 1/24, ioff: bool = True, close_after: bool = False, **kwargs) -> tuple:
        """Run the fit routine. Read documentation for scipy.optimize.curve_fit to see what additional named parameters can be passed through kwargs (x, y, p0 are internally passed).

        Args:
            plot_every (int, optional): Plot every Nth frame (0 to plot every frame, default.) Setting this
            to a positive value causes run() to ignore the value of interval.
            interval (float, optional): Interval between plot frame updates in seconds. Defaults to 1/24.
            Interval is used only when plot_every is non-positive.
            ioff (bool, optional): Execute plt.ioff() after run() to revert the system to non-interactive state. Defaults to True.
            close_after (bool, optional): Execute plt.close() after run() to close the active window. Defaults to False.
            kwargs: Named arguments passed to scipy.optimize.curve_fit. DO NOT pass f, xdata, ydata, p0.

        Returns:
            tuple: popt, pcov (and infodict, mesg, ier if full_output=True) from scipy.optimize.curve_fit.
        """
        if plot_every < 0:
            plot_every = 0
        self._plot_every = plot_every
        if plot_every > 0:
            self._interval = 0
        else:
            self._interval = interval
        output = curve_fit(self._fit_func, self._x, self._y,
                           p0=self._param, **kwargs)
        if self._plot:
            data = self._baseclass.full_field(self._x, output[0])
            res = np.sqrt(np.sum((self._y - data)**2))
            self._ax.set_title('Gaussians: %d, Iteration: %d, Residual: %.3e\nOptimization complete.' % (
                self._n_gaussians, self._iteration, res))
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            if ioff:
                plt.ioff()
            if close_after:
                plt.cla()
                plt.close(self._fig)
                self._fig = None
                gc.collect()
        return output

    def plot(self, *, x: list | np.ndarray = None, y : list | np.ndarray = None, ax: plt.Axes = None, figure_title: str = '', window_title: str = '', show: bool = True, **kwargs) -> plt.Figure:
        """Plot the data and the fit.

        Args:
            x (list | np.ndarray, optional): X data. Defaults to None for internal X data.
            y (list | np.ndarray, optional): Y data. Defaults to None for internal Y data.
            ax (plt.Axes, optional): External axis object. Defaults to None, creates new figure and axis.
            figure_title (str, optional): Figure title, used if ax is not supplied. Defaults to ''.
            window_title (str, optional): Plot window title, used if ax is not supplied. Defaults to ''.
            show (bool, optional): Execute plt.show() at the end. Defaults to True.

        Raises:
            ValueError: _description_

        Returns:
            plt.Figure: _description_
        """
        ax_is_internal = False
        if ax is None:
            # create fig, ax
            ax_is_internal = True
            fig, ax = plt.subplots(1, 1, **kwargs)
            if figure_title is not None and len(figure_title) > 0:
                fig.suptitle(figure_title)
            if window_title is not None and len(window_title) > 0:
                fig.canvas.manager.set_window_title(window_title)
        if x is None:
            x = self._x
        if y is None:
            y = self._y
        if len(x) != len(y):
            raise ValueError('X and Y arrays must have the same length.')
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        # plot everything here
        ax.plot(x, y, label='Data')
        ax.plot(x, self.background(x), label='Background')
        for i in range(self.n_gaussians()):
            ax.plot(x, self.gaussian_w_bg(id=i), label='Gaussian %d'%(i))
        if ax_is_internal:
            ax.set_xlim(x.min(), x.max())
            ax.legend()
        if ax_is_internal and show:
            plt.show()
        return ax.get_figure()

    def full_field(self, x: np.ndarray = None, params: tuple | list | np.ndarray = None) -> np.ndarray:
        """Calculate full fit result (background + gaussians)

        Args:
            x (np.ndarray, optional): X data. Leave empty or None to use the X coordinates used to initialize the GaussFit() instance.
            params (tuple | list | np.ndarray, optional): Background and Gaussian parameters. Set to None to use the inferred parameters.

        Returns:
            np.ndarray: Y data.
        """
        if x is None:
            x = self._x
        if params is None:
            params = self._param
        return self._baseclass.full_field(x, params)

    def background(self, x: np.ndarray = None, params: tuple | list | np.ndarray = None) -> np.ndarray:
        """Calculate background only.

        Args:
            x (np.ndarray, optional): X data. Leave empty or None to use the X coordinates used to initialize the GaussFit() instance.
            params (tuple | list | np.ndarray, optional): Background and Gaussian parameters. Set to None to use the inferred parameters.

        Returns:
            np.ndarray: Y data.
        """
        if x is None:
            x = self._x
        if params is None:
            params = self._param
        return self._baseclass.background(x, params)

    def gaussian_w_bg(self, x: np.ndarray = None, id: int = 0, params: tuple | list | np.ndarray = None) -> np.ndarray:
        """Calculate fit result (background + single gaussian)

        Args:
            x (np.ndarray, optional): X data. Leave empty or None to use the X coordinates used to initialize the GaussFit() instance.
            id (int, optional): ID of the Gaussian [[0, N), where N is the total number of Gaussians.] Defaults to 0.
            params (tuple | list | np.ndarray, optional): Background and Gaussian parameters. Set to None to use the inferred parameters.

        Returns:
            np.ndarray: Y data.
        """
        if x is None:
            x = self._x
        if params is None:
            params = self._param
        return self._baseclass.gaussian_w_bg(x, id, params)

    def gaussian_wo_bg(self, x: np.ndarray = None, id: int = 0, params: tuple | list | np.ndarray = None) -> np.ndarray:
        """Calculate single Gaussian (without background)

        Args:
            x (np.ndarray, optional): X data. Leave empty or None to use the X coordinates used to initialize the GaussFit() instance.
            id (int, optional): ID of the Gaussian [[0, N), where N is the total number of Gaussians.] Defaults to 0.
            params (tuple | list | np.ndarray, optional): Background and Gaussian parameters. Set to None to use the inferred parameters.

        Returns:
            np.ndarray: Y data.
        """
        if x is None:
            x = self._x
        if params is None:
            params = self._param
        return self._baseclass.gaussian_wo_bg(x, id, params)

    @staticmethod
    def gaussian(x: np.ndarray, H: float, A: float, x0: float, sigma: float) -> np.ndarray:
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

    def n_gaussians(self, params: tuple | list | np.ndarray = None) -> int:
        """Get number of Gaussians inferred from the parameters.

        Returns:
            int: Number of Gaussians.
        """
        if params is None:
            params = self._param
        return self._baseclass.n_gaussians(params)

    def n_background_params(self) -> int:
        """Get number of background parameters.

        Returns:
            int: Number of background parameters.
        """
        return self._baseclass.n_background_params()

    def gaussfunc_name(self) -> str:
        """Get name of the underlying Gaussian function class.

        Returns:
            str: ame of the underlying Gaussian function class
        """
        return type(self._baseclass).__name__


if __name__ == '__main__':
    x = np.linspace(-2, 2, 100)
    p_def = (0, 10, 0.2, -0.5, 0, 0, 0.2, 10, 0.1)
    y = 10 + 0.2 * x - 0.5 * x**2 + 10 * np.exp(-((x - 0.2) / 0.1)**2)
    p0 = (0, 10, 0.1, -0.1, 0, 0, 0, 8, 0.5)
    p_low = (-2, 0, -np.inf, -np.inf, -1e-8, -1e-8, -2, 0, 0)
    p_high = (2, np.inf, np.inf, 0, 1e-8, 1e-8, 2, np.inf, np.inf)
    gfit = GaussFit(x, y, p0)
    gfit.run(ioff=True, close_after=False, bounds=(p_low, p_high))
    fparam = list(gfit.param)
    fparam[-1] = abs(fparam[-1])
    fparam = tuple(fparam)
    print('Original:', p_def)
    print('Derived:', fparam)
    print('Error:', ((y - gfit.full_field())**2).sum())
    gfit.plot()