from __future__ import annotations
from typing import Tuple
from weakref import proxy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, OptimizeResult
from abc import abstractmethod
import gc


class MinFuncBase:
    """Abstract class methods that a base class for GenericMinimizeManager must provide.
    """
    @abstractmethod
    def get_x(self) -> np.ndarray:
        """Get the X axis (for plot).

        Returns:
            np.ndarray: X coordinates.
        """
        pass

    @abstractmethod
    def get_y0(self) -> np.ndarray:
        """Get the Y0 reference (for plot).

        Returns:
            np.ndarray: Y0 values.
        """
        pass

    @abstractmethod
    def get_y(self) -> np.ndarray:
        """Get the Y values (for plot).

        Returns:
            np.ndarray: Y values.
        """
        pass

    @abstractmethod
    def get_metric(self) -> float:
        """Evaluate the minimization metric.

        Returns:
            float: Minimization metric.
        """
        pass

    @abstractmethod
    def update_guess(self, p0: tuple | list | np.ndarray) -> None:
        """Update the independent parameters.
        """
        pass

    @abstractmethod
    def num_params(self) -> int:
        """Get numbers of supported parameters.

        Return:
            int: Number of supported parameters.
        """
        pass


class GenericMinimizeManager:
    """Minimize a metric iteratively using *MinimizeFunc class, using scipy.optimize.least_squares.
    """

    def __init__(self, *, p0: tuple | list | np.ndarray, baseclass: MinFuncBase, plot: bool = True, figure_title: str = None, window_title: str = None, hist_len: int = 50, **kwargs):
        """Initialize a GenericMinimizeManager object.

        Args:
            p0 (tuple | list | np.ndarray): Fit parameters initial guess. p0 must have the length
            compatible with the *MinimizeFunc class.
            baseclass (*MinimizeFunc(MinFuncBase)): Function set to evaluate background and features. Must be an instance of a class derived from `MinFuncBase`.
            plot (bool, optional): Plot progress of the fit. Defaults to True.
            figure_title (str, optional): Progress figure title. Defaults to None.
            window_title (str, optional): Progress window title. Defaults to None.
            kwargs: Additional arguments passed to the internal plt.subplots() call.

        Raises:
            TypeError: Base function class is not of *MinimizeFunc type.
            ValueError: Initial guess parameter list incompatible with base function class.
        """
        if not isinstance(baseclass, MinFuncBase):
            raise TypeError('Base class must be of *MinimizeFunc type.')
        if len(p0) != baseclass.num_params():
            raise ValueError('Length of initial guess parameters list does not match the number of parameters accepted by %s.' % (
                self.baseclass_name))
        self._baseclass = baseclass
        self._baseclass.update_guess(p0)  # evaluate X, Y, Y0
        self._plot = plot
        self._iteration = 0
        self._param = p0
        self._last_res = self._baseclass.get_metric()
        self._dres = None
        self._plot_every = 0
        self._output: OptimizeResult = None
        self._metric_hist = [np.nan] * hist_len
        if (plot):
            x = self._baseclass.get_x()
            y = self._baseclass.get_y()
            y0 = self._baseclass.get_y0()
            cost = self._baseclass.get_metric()
            fig, ax = plt.subplots(
                2, 1, gridspec_kw={'height_ratios': (3, 1)}, **kwargs)
            ax_s = ax[1]
            ax = ax[0]
            self._fig = fig
            if figure_title is not None and len(figure_title) > 0:
                self._fig.suptitle(figure_title)
            if window_title is not None and len(window_title) > 0:
                self._fig.canvas.manager.set_window_title(window_title)
            self._ax = ax
            self._ax_s = ax_s
            self._orig, = self._ax.plot(x, y0, color='r')
            self._sig, = self._ax.plot(
                x, y, color='k')
            self._hist, = self._ax_s.plot(np.arange(self._iteration - len(
                self._metric_hist), self._iteration), self._metric_hist, color='b', ls='--', marker='*')
            self._ax_s.set_xlabel('Iteration')
            self._ax_s.set_ylabel('Metric')
            self._ax_s.set_yscale('log')
            xmin = x.min()
            xmax = x.max()
            if xmax == xmin:
                xmin -= 0.5
                xmax += 0.5
            self._ax.set_xlim(xmin, xmax)
            self._ax.set_title('Iteration: %d, Residual: %.3e, Delta: %s' % (
                self._iteration, cost, 'Not Available'))
            plt.ion()
            plt.show()

    @property
    def param(self) -> tuple:
        """Get the latest fit parameters.

        Returns:
            tuple: Fit parameters.
        """
        return tuple(self._param)

    @property
    def meansq_error(self) -> float:
        """Get the mean squared error.

        Returns:
            float: Mean squared error.
        """
        return self._baseclass.get_metric()

    def _fit_func(self, *params):
        self._baseclass.update_guess(params)
        res = self._baseclass.get_metric()
        if res < 0:
            raise RuntimeError(
                'Residual can not be negative. Please check implementation.')
        self._metric_hist.pop(0)
        self._metric_hist.append(res)
        self._update(self._baseclass.get_x(),
                     self._baseclass.get_y(), self._baseclass.get_y0(), res)
        self._param = params
        return res

    def _update(self, x: np.ndarray, y: np.ndarray, y0: np.ndarray, res: float):
        if self._plot:
            if self._plot_every > 0 and (self._iteration % self._plot_every):
                pass
            else:
                self._orig.set_data(x, y0)
                self._sig.set_data(x, y)

                ymin = min(y.min(), y0.min())
                ymax = min(y.max(), y0.max())

                if ymin == ymax:
                    ymin -= 0.5
                    ymax += 0.5

                xmin = x.min()
                xmax = x.max()

                if xmin == xmax:
                    xmin -= 0.5
                    xmax += 0.5

                self._ax.set_xlim(xmin, xmax)
                self._ax.set_ylim(ymin, ymax)

                if self._last_res is None:
                    self._last_res = res
                dres = res - self._last_res

                self._ax.set_title('Iteration: %d, Residual: %.3e, Delta: %s' % (
                    self._iteration, res, 'None' if self._dres is None else ('%.3e' % (dres))))

                self._dres = dres

                if self._iteration:
                    iter_x = np.arange(self._iteration -
                                       len(self._metric_hist), self._iteration)
                    iter_y = np.asarray(self._metric_hist)
                    xmin = iter_x.min()
                    xmax = iter_x.max()
                    ymin = np.nanmin(iter_y)
                    ymax = np.nanmax(iter_y)
                    self._ax_s.set_xlim(xmin, xmax)
                    self._ax_s.set_ylim(ymin, ymax)
                    self._hist.set_data(iter_x, iter_y)

                self._fig.canvas.draw()
                self._fig.canvas.flush_events()
        self._iteration += 1

        if self._interval > 0 and self._plot:
            plt.pause(self._interval)

    def run(self, *, plot_every: int = 0, interval: float = 1/24, ioff: bool = True, close_after: bool = False, **kwargs) -> OptimizeResult:
        """Run the minimization routine. Read documentation for scipy.optimize.least_squares to see what additional named parameters can be passed through kwargs (f, p0 are internally passed).

        Args:
            plot_every (int, optional): Plot every Nth frame (0 to plot every frame, default.) Setting this
            to a positive value causes run() to ignore the value of interval.
            interval (float, optional): Interval between plot frame updates in seconds. Defaults to 1/24.
            Interval is used only when plot_every is non-positive.
            ioff (bool, optional): Execute plt.ioff() after run() to revert the system to non-interactive state. Defaults to True.
            close_after (bool, optional): Execute plt.close() after run() to close the active window. Defaults to False.
            kwargs: Named arguments passed to scipy.optimize.least_squares. DO NOT pass fun, x0, args and callback.

        Raises:
            RuntimeError: If residual/metric is negative, RuntimeError is raised.

        Returns:
            OptimizeResult: Output of scipy.optimize.least_squares. Attribute x: np.ndarray is the solution of the optimization.
        """
        unsupported_args = ['fun', 'x0', 'args']
        for key in unsupported_args:
            if key in dict(kwargs).keys():
                kwargs.pop(key)
        if plot_every < 0:
            plot_every = 0
        self._plot_every = plot_every
        if plot_every > 0:
            self._interval = 0
        else:
            self._interval = interval
        self._output = output = least_squares(self._fit_func,
                                              x0=self._param, **kwargs)
        if self._plot:
            p0 = tuple(output.x)
            self._fit_func(*p0)
            self._iteration -= 1
            self._ax.set_title('Iteration: %d, Residual: %.3e\nOptimization %s.' % (
                self._iteration, self._baseclass.get_metric(), 'Success' if output.success else 'Failed'))
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

    @property
    def baseclass_name(self) -> str:
        """Get name of the underlying *MinimizeFunc function class.

        Returns:
            str: Name of the underlying *MinimizeFunc function class
        """
        return type(self._baseclass).__name__


if __name__ == '__main__':
    pass
