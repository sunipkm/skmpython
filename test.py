from skmpython import PerfTimer, format_tdiff
import time

pt = None
N = 10
for i in range(N):
    if pt is None:
        pt = PerfTimer(N)
    time.sleep(1)
    pt.update(i + 1)
    print('ETA: %s\tElapsed: %s' %
          (format_tdiff(pt.eta, '%S.%f'), format_tdiff(pt.elapsed, '%S.%f')))
