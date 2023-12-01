from __future__ import annotations
from typing import SupportsAbs, Tuple
from weakref import proxy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from abc import abstractmethod
import gc


class FitFuncBase:
    """Abstract class listing methods that a base class for GenericFitManager must provide.
    """
    @abstractmethod
    def full_field(self, x: np.ndarray, params: tuple | list | np.ndarray) -> np.ndarray:
        """Calculate full fit result (background + features)

        Args:
            x (np.ndarray): X coordinates.
            params (tuple | list | np.ndarray): Fit parameters.

        Raises:
            RuntimeError: Expected number of parameters not equal to number of supplied parameters.

        Returns:
            np.ndarray: Y coordinates.
        """
        pass

    @staticmethod
    @abstractmethod
    def background(self, x: np.ndarray, params: tuple | list | np.ndarray) -> np.ndarray:
        """Calculate background only.

        Args:
            x (np.ndarray): X coordinates.
            params (tuple | list | np.ndarray): Flat list of function parameters.

        Raises:
            RuntimeError: Expected number of parameters not equal to number of supplied parameters.

        Returns:
            np.ndarray: Y coordinates of the background.
        """
        pass

    @staticmethod
    @abstractmethod
    def feature(self, x: np.ndarray, params: tuple | list | np.ndarray, *, id: int, with_background: bool = True) -> np.ndarray:
        """Calculate single feature.

        Args:
            x (np.ndarray): X coordinates.
            params (tuple | list | np.ndarray): Flat list of function parameters.
            id (int): ID of the feature. Supports negative indexing.
            with_background (bool): Add background to feature. Defaults to True.

        Raises:
            RuntimeError: Expected number of parameters not equal to number of supplied parameters.
            RuntimeError: Feature ID exceeds number of available features.

        Returns:
            np.ndarray: Y coordinates.
        """
        pass

    @abstractmethod
    def num_features(self) -> int:
        """Get number of features.

        Returns:
            int: Number of features.
        """
        pass

    @abstractmethod
    def num_feature_params(self, id: int) -> Tuple(int, int):
        """Get number of parameters for the feature ID.

        Args:
            id (int): Feature ID. Supports negative indexing.

        Raises:
            RuntimeError: Feature ID exceeds number of available features.

        Returns:
            tuple(int, int): Start index of parameters, number of parameters.
        """
        pass

    @abstractmethod
    def num_background_params(self) -> int:
        """Get number of background parameters.

        Returns:
            int: Number of background parameters.
        """
        pass

    @abstractmethod
    def num_params(self) -> int:
        """Get total number of parameters.

        Returns:
            int: Number of parameters.
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize the instance, i.e. mark the function class immutable for execution.
        Read the GenericFitFunc() class documentation for the need of such a function.
        GenericFitManager() automatically calls this function on init, so this must
        be provided.
        """
        pass

    @abstractmethod
    def isfinalized(self) -> bool:
        """Check if the instance is finalized using the finalize() method.

        Returns:
            bool: True if finalized, False otherwise.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Get the representation of this function.

        Returns:
            str: Representation string.
        """
        return object.__repr__(self)


class GenericFitFunc(FitFuncBase):
    """This class accepts generic functions as backgrounds and features, along with number of parameters.
    """
    @staticmethod
    def _dummy(param) -> np.ndarray:
        return np.zeros((1), dtype=float)

    def __init__(self, *, background_fcn: object, num_background_params: int):
        """Initialize a generic fit function backend.

        Args:
            background_fcn (object): A callable function that takes X coordinates and N parameters to calculate the background.
            num_background_params (int): Number of parameters (N).

        Raises:
            ValueError: background_fcn object can not be None.
            TypeError: background_fcn has to be callable.
            ValueError: num_background_params must be non-negative.
        """
        if background_fcn is None:
            raise ValueError('Function object can not be None.')
        if not callable(background_fcn):
            raise TypeError('Object %s is not callable.' %
                            (background_fcn.__name__))
        if num_background_params < 0:
            raise ValueError(
                'Number of background function parameters can not be negative.')
        self._tot_params = num_background_params
        self._bckFcn = background_fcn  # background function
        # no parameter space for background functions
        self._bckParams = num_background_params
        self._featureFcns = []  # empty list of feature function
        self._featureFcnParamN = []  # empty list of feature function parameter numbers
        self._finalized = False
        self._output = None

    def register_feature_fcn(self, *, fcn: object, num_params: int) -> None:
        """Register features.

        Args:
            fcn (object): A callable function that takes X coordinates and N parameters to calculate a feature. Must be a callable.
            num_params (int): Number of parameters (N).

        Raises:
            RuntimeError: This instance of GenericFitFunc() has been finalized.
            TypeError: fcn must be callable.
            ValueError: Number of parameters can not be non-negative.
        """
        if isinstance(self._featureFcns, tuple):
            raise RuntimeError(
                'Can not register any more features to this instance.')
        elif fcn is None or not callable(fcn):
            raise TypeError('Object %s is not callable.' % (fcn.__name__))
        if num_params < 0:
            raise ValueError(
                'Number of background function parameters can not be negative.')

        self._featureFcns.append(fcn)
        self._featureFcnParamN.append(num_params)
        self._tot_params += num_params

    def finalize(self):
        """Finalize this instance of the GenericFitFunc() class. Finalization renders the internal
        feature function structures immutable, and prevents addition of further features. This is
        crucial to the generic functionality of this class. GenericFitManager() automatically
        calls this function on init.
        """
        if isinstance(self._featureFcns, tuple):
            return
        # convert to tuple, immutable
        self._featureFcns = tuple(self._featureFcns)
        featureFcnParamN = np.cumsum(
            [self._bckParams] + self._featureFcnParamN, axis=0, dtype=int)  # convert to slice indices
        self._featureFcnParamN = tuple(
            featureFcnParamN)  # convert to immutable
        self._finalized = True
        return

    def isfinalized(self) -> bool:
        return self._finalized

    def background(self, x: np.ndarray, params: tuple | list | np.ndarray) -> np.ndarray:
        if len(params) != self._tot_params:
            raise RuntimeError('Expected number of parameters %d, received %d' % (
                self._tot_params, len(params)))
        y = np.zeros(x.shape, dtype=float)
        params = tuple(params[:self._bckParams])
        y += self._bckFcn(x, *params)
        return y

    def feature(self, x: np.ndarray, params: tuple | list | np.ndarray, *, id: int, with_background: bool = True) -> np.ndarray:
        if len(params) != self._tot_params:
            raise RuntimeError('Expected number of parameters %d, received %d' % (
                self._tot_params, len(params)))
        if not -len(self._featureFcns) <= id < len(self._featureFcns):
            raise RuntimeError(
                'Feature ID exceeds number of available features.')
        if id < 0:
            id += len(self._featureFcns)
        y = np.zeros(x.shape, dtype=float)
        _params = tuple(params[self._featureFcnParamN[id]
                        :self._featureFcnParamN[id+1]])
        y += self._featureFcns[id](x, *_params)
        if with_background:
            y += self.background(x, params)
        return y

    def full_field(self, x: np.ndarray, params: tuple | list | np.ndarray) -> np.ndarray:
        if len(params) != self._tot_params:
            raise RuntimeError('Expected number of parameters %d, received %d' % (
                self._tot_params, len(params)))

        y = self.background(x, params)
        for i in range(self.num_features()):
            y += self.feature(x, params, id=i, with_background=False)

        return y

    def num_features(self) -> int:
        return len(self._featureFcns)

    def num_feature_params(self, id: int) -> Tuple(int, int):
        if not -len(self._featureFcns) <= id < len(self._featureFcns):
            raise RuntimeError(
                'Feature ID exceeds number of available features.')
        if id < 0:
            id += len(self._featureFcns)
        idx = self._featureFcnParamN[id]
        n = self._featureFcnParamN[id + 1] - idx
        return (idx, n)

    def num_background_params(self) -> int:
        return self._bckParams

    def num_params(self) -> int:
        out = self.num_background_params()
        for i in range(self.num_features()):
            out += self.num_feature_params(i)[1]
        return out

    def __repr__(self) -> str:
        _name = type(self).__name__
        ret = '%s object with %d parameters (%d background parameters). %d feature%s; Feature parameters (index, length): ' % (
            _name, self.num_params(), self.num_background_params(), self.num_features(), 's' if self.num_features() > 1 else '')
        for i in range(self.num_features()):
            ret += str(self.num_feature_params(i)) + ', '
        ret = ret.rstrip(', ')
        ret += '.'
        return ret


class GenericFitManager:
    """Fit data to background + features using the *FitFunc class.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, *, p0: tuple | list | np.ndarray, baseclass: FitFuncBase, plot: bool = True, figure_title: str = None, window_title: str = None, **kwargs):
        """Initialize a GenericFitManager object.

        Args:
            x (np.ndarray): X coordinates.
            y (np.ndarray): Y coordinates.
            p0 (tuple | list | np.ndarray): Fit parameters initial guess. p0 must have the length
            compatible with the *FitFunc class.
            baseclass (*FitFunc(FitFuncBase)): Function set to evaluate background and features. Must be an instance of a class (e.g. `GenericFitFunc`) derived from `FitFuncBase`.
            plot (bool, optional): Plot progress of the fit. Defaults to True.
            figure_title (str, optional): Progress figure title. Defaults to None.
            window_title (str, optional): Progress window title. Defaults to None.
            kwargs: Additional arguments passed to the internal plt.subplots() call.

        Raises:
            TypeError: Base function class is not of *FitFunc type.
            ValueError: Initial guess parameter list incompatible with base function class.
        """
        if not isinstance(baseclass, FitFuncBase):
            raise TypeError('Base class must be of *FitFunc type.')
        if len(p0) != baseclass.num_params():
            raise ValueError('Length of initial guess parameters list does not match the number of parameters accepted by %s.' % (
                self.baseclass_name))
        self._baseclass = baseclass
        self._baseclass.finalize()
        self._n_features = self._baseclass.num_features()
        self._figtit = ''
        self._wintit = ''
        if (plot):
            fig, ax = plt.subplots(1, 1, **kwargs)
            self._fig = fig
            if figure_title is not None and len(figure_title) > 0:
                self._fig.suptitle(figure_title)
                self._figtit = figure_title
            if window_title is not None and len(window_title) > 0:
                self._fig.canvas.manager.set_window_title(window_title)
                self._wintit = window_title
            self._ax = ax
            self._orig, = self._ax.plot(x, y, color='r')
            self._bck, = self._ax.plot(
                x, self._baseclass.background(x, p0), color='k')
            self._sig = []
            for i in range(self._n_features):
                sig, = self._ax.plot(
                    x, self._baseclass.feature(x, p0, id=i))
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

    @property
    def iterations(self) -> int:
        """Get the total number of iterations performed.

        Returns:
            int: Number of iterations for fit convergence.
        """
        return self._iteration

    @property
    def meansq_error(self) -> float:
        """Get the mean squared error.

        Returns:
            float: Mean squared error.
        """
        return ((self._y - self.full_field())**2).sum()

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
                data = bck.copy()
                for idx, sig in enumerate(self._sig):
                    gaus = self._baseclass.feature(
                        x, params, id=idx, with_background=False)
                    data += gaus
                    gaus += bck
                    sig.set_ydata(gaus)
                self._ax.set_ylim(np.min([data.min(), self._y.min()]), np.max(
                    [data.max(), self._y.max()]))
                res = np.sqrt(np.sum((self._y - data)**2))
                if self._last_res is None:
                    self._last_res = res
                dres = res - self._last_res
                self._last_res = res
                self._ax.set_title('Features: %d, Iteration: %d, Residual: %.3e, Delta: %.3e' % (
                    self._n_features, self._iteration, res, dres))
                self._fig.canvas.draw()
                self._fig.canvas.flush_events()
        self._iteration += 1

        if self._interval > 0 and self._plot:
            plt.pause(self._interval)

    def run(self, *, plot_every: int = 0, interval: float = 1/24, ioff: bool = True, close_after: bool = False, throw_error: bool = False, **kwargs) -> tuple:
        """Run the fit routine. Read documentation for scipy.optimize.curve_fit to see what additional named parameters can be passed through kwargs (x, y, p0 are internally passed).

        Args:
            plot_every (int, optional): Plot every Nth frame (0 to plot every frame, default.) Setting this
            to a positive value causes run() to ignore the value of interval.
            interval (float, optional): Interval between plot frame updates in seconds. Defaults to 1/24.
            Interval is used only when plot_every is non-positive.
            ioff (bool, optional): Execute plt.ioff() after run() to revert the system to non-interactive state. Defaults to True.
            close_after (bool, optional): Execute plt.close() after run() to close the active window. Defaults to False.
            throw_error (bool, optional): Throw RuntimeError if the least-square minimization fails. Defaults to False.
            kwargs: Named arguments passed to scipy.optimize.curve_fit. DO NOT pass f, xdata, ydata, p0.

        Returns:
            tuple: popt, pcov (and infodict, mesg, ier if full_output=True) from scipy.optimize.curve_fit.

        Raises:
            RuntimeError: if the least-square minimization fails.
            OptimizeWarning: if covariance of the parameters can not be estimated.
        """
        unsupported_args = ['f', 'xdata', 'ydata', 'p0']
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
        errored = False
        if throw_error:
            self._output = output = curve_fit(self._fit_func, self._x, self._y,
                                          p0=self._param, **kwargs)
        else:
            try:
                self._output = output = curve_fit(self._fit_func, self._x, self._y,
                                          p0=self._param, **kwargs)
            except RuntimeError:
                errored = True
                output = self._param # last call value
        if self._plot:
            data = self._baseclass.full_field(self._x, output[0])
            res = np.sqrt(np.sum((self._y - data)**2))
            tot_pow = np.trapz(self._y, self._x)
            fit_pow = np.trapz(self._baseclass.background(self._x, output[0]), self._x)
            for fidx in range(self._n_features):
                fit_pow += np.trapz(self._baseclass.feature(self._x, output[0], id=fidx, with_background=False), self._x)
            self._ax.set_title('Features: %d, Iteration: %d, Residual: %.3e\nIntegrated sum: Data: %.3e, Fit: %.3e, Diff: %.3e' % (
                self._n_features, self._iteration, res, tot_pow, fit_pow, tot_pow - fit_pow))
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            if ioff:
                plt.ioff()
            if close_after:
                plt.cla()
                plt.close(self._fig)
                self._fig = None
                gc.collect()
        if errored:
            output = self._output = (None, None)
        return output

    def plot(self, *, x: list | np.ndarray = None, y: list | np.ndarray = None, ax: plt.Axes = None, figure_title: str = None, window_title: str = None, show: bool = True, **kwargs) -> Tuple(plt.Figure, plt.Axes):
        """Plot the data and the fit.

        Args:
            x (list | np.ndarray, optional): X coordinates. Defaults to None for internal X coordinates.
            y (list | np.ndarray, optional): Y coordinates. Defaults to None for internal Y coordinates.
            ax (plt.Axes, optional): External axis object. Defaults to None, creates new figure and axis.
            figure_title (str, optional): Figure title, used if ax is not supplied. Defaults to None.
            window_title (str, optional): Plot window title, used if ax is not supplied. Defaults to None.
            show (bool, optional): Execute plt.show() at the end. Defaults to True.

        Raises:
            ValueError: X and Y arrays are not of the same dimension.

        Returns:
            tuple(plt.Figure, plt.Axes): Matplotlib figure and axes objects.
        """
        ax_is_internal = False
        if ax is None:
            # create fig, ax
            ax_is_internal = True
            fig, ax = plt.subplots(1, 1, **kwargs)
            if figure_title is not None and len(figure_title) > 0:
                fig.suptitle(figure_title)
            elif len(self._figtit) > 0:
                fig.suptitle(self._figtit)
            if window_title is not None and len(window_title) > 0:
                fig.canvas.manager.set_window_title(window_title)
            elif len(self._wintit) > 0:
                fig.canvas.manager.set_window_title(self._wintit)

        if x is None:
            x = self._x
        if y is None:
            y = self._y
        if len(x) != len(y):
            raise ValueError('X and Y arrays must have the same length.')
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        # plot everything here
        ax.plot(x, y, color='k', label='Data')
        res_fmt = '%.3f'
        if abs(y.mean()) < 0.001:
            res_fmt = '%.3e'
        res_fmt = 'Residual + ' + res_fmt
        ax.plot(x, y - self.full_field(x) +
                y.mean(), label=res_fmt % (y.mean()))
        for i in range(self.num_features()):
            ax.plot(x, self.feature(id=i), label='Feature %d' % (i))
        ax.plot(x, self.background(x), label='Background')
        if ax_is_internal:
            ax.set_xlim(x.min(), x.max())
            ax.legend()
        if ax_is_internal and show:
            plt.show()
        return (ax.get_figure(), ax)
    
    def integrated_err(self, as_fraction: bool = False) -> SupportsAbs:
        """Calculate the integrated error between the data and the fit.
        
        Args:
            as_fraction (bool, optional): Return the integrated error as a fraction of the total power. Defaults to False.
            
        Returns:
            SupportsAbs: Integrated error.
        """
        tot_pow = np.trapz(self._y, self._x)
        fit_pow = np.trapz(self._baseclass.background(self._x, self._output[0]), self._x)
        for fidx in range(self._n_features):
            fit_pow += np.trapz(self._baseclass.feature(self._x, self._output[0], id=fidx, with_background=False), self._x)
        res = tot_pow - fit_pow
        if as_fraction:
            res /= tot_pow
        return res

    def full_field(self, x: np.ndarray = None, params: tuple | list | np.ndarray = None) -> np.ndarray:
        """Calculate full fit result (background + features)

        Args:
            x (np.ndarray, optional): X coordinates. Leave empty or None to use the X coordinates used to initialize the GenericFitManager() instance.
            params (tuple | list | np.ndarray, optional): Background and feature parameters. Set to None to use the inferred parameters.

        Returns:
            np.ndarray: Y coordinates.
        """
        if x is None:
            x = self._x
        if params is None:
            params = self._param
        return self._baseclass.full_field(x, params)

    def background(self, x: np.ndarray = None, params: tuple | list | np.ndarray = None) -> np.ndarray:
        """Calculate background only.

        Args:
            x (np.ndarray, optional): X coordinates. Leave empty or None to use the X coordinates used to initialize the GenericFitManager() instance.
            params (tuple | list | np.ndarray, optional): Background and feature parameters. Set to None to use the inferred parameters.

        Returns:
            np.ndarray: Y coordinates.
        """
        if x is None:
            x = self._x
        if params is None:
            params = self._param
        return self._baseclass.background(x, params)

    def feature(self, x: np.ndarray = None, params: tuple | list | np.ndarray = None, *, id: int = 0, with_background: bool = True) -> np.ndarray:
        """Calculate single feature.

        Args:
            x (np.ndarray, optional): X coordinates. Leave empty or None to use the X coordinates used to initialize the GenericFitManager() instance.
            params (tuple | list | np.ndarray, optional): Background and feature parameters. Set to None to use the inferred parameters.
            id (int, optional): ID of the Feature [[0, N), where N is the total number of features.] Defaults to 0.
            with_background (bool, optional): Return feature + background. Defaults to True.

        Returns:
            np.ndarray: Y coordinates.
        """
        if x is None:
            x = self._x
        if params is None:
            params = self._param
        return self._baseclass.feature(x, params, id=id, with_background=with_background)

    def num_features(self) -> int:
        """Get number of features fit with this instance.

        Returns:
            int: Number of features.
        """
        return self._baseclass.num_features()

    def num_background_params(self) -> int:
        """Get number of background parameters.

        Returns:
            int: Number of background parameters.
        """
        return self._baseclass.num_background_params()

    def num_params(self) -> int:
        """Get number of function parameters.

        Returns:
            int: Number of function parameters
        """
        return self._baseclass.num_params()

    def get_backend(self) -> FitFuncBase:
        """Get the instance of the fit function class used to fit the data.

        Returns:
            FitFuncBase: Instance of fit function class used to fit the data.
        """
        return self._baseclass

    def wrap_results(self) -> dict:
        """Get the result of the fit (if GenericFitManager().run() has been executed) and the base function class.

        Returns:
            dict: Key 'output': Output of scipy.optimize.curve_fit(), key 'backend': Fit function class instance.
        """
        return {'output': self._output, 'backend': self._baseclass}

    @property
    def baseclass_name(self) -> str:
        """Get name of the underlying *FitFunc function class.

        Returns:
            str: Name of the underlying *FitFunc function class
        """
        return type(self._baseclass).__name__


if __name__ == '__main__':
    x = np.linspace(-2, 2, 100)
    # parameters to generate the data to fit
    p_def = np.asarray([0, 10, 0.2, -0.5, 0.2, 10, 0.1], dtype=float)
    p0 = (0, 10, 0.1, -0.1, 0, 8, 0.5)  # initial guess parameters
    p_low = (-2, 0, -np.inf, -np.inf, -2, 0, 0)  # initial guess lower bounds
    # inital guess upper bounds
    p_high = (2, np.inf, np.inf, 0, 2, np.inf, np.inf)
    def background_fcn(x, x0, a0, a1, a2): return a0 + \
        a1 * (x-x0) + a2*(x-x0)**2
    # create the generic fit function and register the background
    bfuncs = GenericFitFunc(
        background_fcn=background_fcn, num_background_params=4)

    def feature_fcn(x, c, a, w): return a*np.exp(-((x-c)/w)**2)
    bfuncs.register_feature_fcn(
        fcn=feature_fcn, num_params=3)  # register the feature
    bfuncs.finalize()  # finalize the registration
    y = bfuncs.full_field(x, p_def)  # calculate the data to fit
    noise = np.random.uniform(-0.1, 0.1, size=y.shape)  # inject noise
    y += noise  # inject noise
    # create the fit manager
    gfit = GenericFitManager(x, y, p0=p0, baseclass=bfuncs, window_title='Test')
    print('GenericFitManager using %s backend.' %
          (gfit.baseclass_name))  # print the base class
    print('Original:', p_def)  # print the original parameters
    gfit.run(ioff=True, close_after=False, bounds=(
        p_low, p_high), p0=p0)  # run the fit
    print('Derived:', gfit.param)  # print the derived parameters
    # print the mean squared error and noise
    print('Error:', gfit.meansq_error, ', Noise:', noise.std(), ', Initial guess error:', gfit.integrated_err())
    print(gfit.wrap_results())
    print('Number of iterations:', gfit.iterations)
    gfit.plot()  # plot the fit result
