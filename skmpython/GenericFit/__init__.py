"""
GenericFit
=============

Provides
  1. A wrapper around scipy.optimize.curve_fit in order to fit generic functions, called GenericFitManager.
  2. A generic class to register functions as backgrounds and features, fed to GenericFitManager to fit the data to these functions.
"""

from ._generic_fit import FitFuncBase, GenericFitFunc, GenericFitManager

__all__ = ['FitFuncBase', 'GenericFitFunc', 'GenericFitManager']