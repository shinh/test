import os
import sys

class _DisableStderr(object):

  def __enter__(self):
    self._dev_null_fd = os.open('/dev/null', os.O_WRONLY)
    self._stderr_fd = os.dup(sys.stderr.fileno())
    os.dup2(self._dev_null_fd, sys.stderr.fileno())

  def __exit__(self, *args):
    os.dup2(self._stderr_fd, sys.stderr.fileno())
    os.close(self._stderr_fd)
    os.close(self._dev_null_fd)

sys.stderr.write('hoge\n')
with _DisableStderr():
  sys.stderr.write('fuga\n')
sys.stderr.write('hige\n')
