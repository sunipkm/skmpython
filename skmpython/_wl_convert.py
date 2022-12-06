from __future__ import annotations
import numpy as np
from collections.abc import Iterable

def vac2air(wl_nm: float | np.ndarray)->float | np.ndarray:
    """Convert vacuum wavelength in nm to wavelength in air.
    Ref: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    Air-to-vacuum conversion, Nikolai Piskunov

    Args:
        wl_nm (float | np.ndarray): Input vacuum wavelength (nm).

    Returns:
        float | np.ndarray: Output air wavelength (nm).
    """
    s2 = 1e6 / (wl_nm * wl_nm)
    return (1 + 0.0000834254 + (0.02406147 / (130 - s2)) + (0.00015998 / (38.9 - s2)))

def air2vac(wl_nm: float | np.ndarray)->float | np.ndarray:
    """Convert air wavelength in nm to wavelength in vacuum.
    Ref: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    Air-to-vacuum conversion, Nikolai Piskunov

    Args:
        wl_nm (float | np.ndarray): Input air wavelength (nm).

    Returns:
        float | np.ndarray: Output vacuum wavelength (nm).
    """
    s2 = 1e6 / (wl_nm * wl_nm)
    return (1 + 0.00008336624212083 + (0.02408926869968 / (130.1065924522 - s2)) + (0.0001599740894897 / (38.92568793293 - s2)))
