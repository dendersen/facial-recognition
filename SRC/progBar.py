from typing import Union
import sys
from time import time

class progBar:
  def __init__(self, total: Union[float, int], prefix: str = 'Progress:', length: int = 50, fill: str = '█'):
    self.total = total - 1
    self.startTime = time()
    self.prefix = prefix
    self.length = length
    self.fill = fill

  def print(self, iteration, suffix=""):
    elapsed_time = time() - self.startTime
    progress = iteration / float(self.total)
    remaining_time = (elapsed_time / progress) - elapsed_time

    percent = f"{100 * progress:.1f}"
    filled_length = int(self.length * iteration / self.total)
    bar = self.fill * filled_length + '-' * (self.length - filled_length)
    time_str = f"Remaining: {remaining_time:.1f}s"
    count_str = f"{iteration}/{self.total}"

    sys.stdout.write(f'\r{self.prefix} |{bar}| {percent}% {count_str} {time_str} {suffix}')
    sys.stdout.flush()


def printProgressBar(iteration, total, startTime, prefix='Progress:', suffix='', length=50, fill='█'):
  total -= 1
  elapsed_time = time() - startTime
  progress = iteration / float(total)
  remaining_time = (elapsed_time / progress) - elapsed_time

  percent = ('{0:.1f}').format(100 * progress)
  filled_length = int(length * iteration // total)
  bar = fill * filled_length + '-' * (length - filled_length)
  time_str = 'Remaining: {0:.1f}s'.format(remaining_time)
  count_str = '{}/{}'.format(iteration, total)


  sys.stdout.write('\r%s |%s| %s%% %s %s %s' % (prefix, bar, percent, count_str, time_str, suffix))
  sys.stdout.flush()