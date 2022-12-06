from __future__ import annotations
import numpy as np
from collections.abc import Iterable

def vac2air(wl_nm: float | np.ndarray)->float | np.ndarray:
    """Convert vacuum wavelength in nm to wavelength in air (valid > 200 nm).
    Ref: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    Air-to-vacuum conversion, Nikolai Piskunov

    Blows up around 87.7 nm and 160.33 nm.

    Args:
        wl_nm (float | np.ndarray): Input vacuum wavelength (nm).

    Returns:
        float | np.ndarray: Output air wavelength (nm).
    """
    s2 = 1e6 / (wl_nm * wl_nm)
    return wl_nm / (1 + 0.0000834254 + (0.02406147 / (130 - s2)) + (0.00015998 / (38.9 - s2)))

def air2vac(wl_nm: float | np.ndarray)->float | np.ndarray:
    """Convert air wavelength in nm to wavelength in vacuum (valid > 200 nm)
    Ref: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    Air-to-vacuum conversion, Nikolai Piskunov

    Blows up around 87.7 nm and 160.33 nm.

    Args:
        wl_nm (float | np.ndarray): Input air wavelength (nm).

    Returns:
        float | np.ndarray: Output vacuum wavelength (nm).
    """
    s2 = 1e6 / (wl_nm * wl_nm)
    return wl_nm * (1 + 0.00008336624212083 + (0.02408926869968 / (130.1065924522 - s2)) + (0.0001599740894897 / (38.92568793293 - s2)))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    wl = np.arange(200, 10000) # wavelength from 50 nm to 10 um
    wl_ = air2vac(vac2air(wl)) # transform and inverse
    fig, ax = plt.subplots(1, 1, squeeze=True, tight_layout=True, figsize=(6.4, 4.8))
    fig.suptitle('Vacuum to Air inversion error over valid range')
    ax.set_xscale('log')
    tax = ax.twinx()
    l1, = ax.plot(wl, (wl - wl_), color='k')
    ax.set_xlim(wl.min(), wl.max())
    tax.set_xlim(wl.min(), wl.max())
    l2, = tax.plot(wl, (wl - wl_) / wl * 1e12, color = 'r')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax.set_xlabel(r'Wavelength (nm)')
    ax.set_ylabel(r'$\Delta\lambda$ = vac2air(air2vac($\lambda$))')
    tax.set_ylabel(r'$10^{12}\times \Delta\lambda/\lambda$')
    ax.legend([l1, l2], [r'$\Delta\lambda$', r'$10^{12} \times \Delta\lambda/\lambda$'])
    tax.yaxis.label.set_color('r')
    ax.set_xticks([200, 400, 600, 1000, 2000, 4000, 10000])
    ax.set_xticklabels(map(str, [200, 400, 600, 1000, 2000, 4000, 10000]))
    tax.spines['right'].set_edgecolor('r')
    tax.tick_params(axis='y', colors='r')
    ax.grid()
    plt.show()
