"""
GenericLeastSq
=============

Provides a wrapper around scipy.optimize.minimize for generic minimization, called GenericMinimizeManager.
"""

from ._generic_min import GenericMinimizeManager, MinFuncBase

__all__ = ['GenericMinimizeManager', 'MinFuncBase']