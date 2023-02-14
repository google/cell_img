"""Setup file for Cell Imaging package."""

import logging
import subprocess
import setuptools

# It is recommended to import setuptools prior to importing distutils to avoid
# using legacy behavior from distutils.
# https://setuptools.readthedocs.io/en/latest/history.html#v48-0-0
from distutils.command.build import build as _build  # isort:skip


# This class handles the pip install mechanism.
class build(_build):  # pylint: disable=invalid-name
  """A build command class that will be invoked during package install.

  The package built using the current setup.py will be staged and later
  installed in the worker using `pip install package'. This class will be
  instantiated during install for this specific scenario and will trigger
  running the custom commands specified.
  """
  sub_commands = _build.sub_commands + [('CustomCommands', None)]


CUSTOM_COMMANDS = [
    ['apt-get', 'update'],
    ['apt-get', '--assume-yes', 'install', 'git'],
    [
        'python3', '-m', 'pip', 'install',
        'git+https://github.com/google/cell_img#egg=cell_img',
    ]
]


class CustomCommands(setuptools.Command):
  """A setuptools Command class able to run arbitrary commands."""

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def RunCustomCommand(self, command_list):
    logging.info('Running command: %s', command_list)
    p = subprocess.Popen(
        command_list,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    # Can use communicate(input='y\n'.encode()) if the command run requires
    # some confirmation.
    stdout_data, _ = p.communicate()
    print('Command output: %s' % stdout_data)
    if p.returncode != 0:
      raise RuntimeError('Command %s failed: exit code: %s' %
                         (command_list, p.returncode))

  def run(self):
    for command in CUSTOM_COMMANDS:
      self.RunCustomCommand(command)

PY_MODULES = [
    'data_utils',
]

setuptools.setup(
    name='cell_img',
    version='0.0.1',
    description='Cell Imaging package.',
    install_requires=[
        'absl-py>=0.11.0',
        'apache-beam[gcp]>=2.34.0',
        'fsspec[gcs]',
        'keras>=2.11.0',
        'jaxlib>=0.3.25',
        'lightgbm==2.2.3',
        'numpy>=1.19.5',
        'Pillow>=8.1.0',
        'pandas>=1.2.1',
        'scikit-image>=0.18.3',
        'tensorstore>=0.1.10',
        'tensorflow-hub>=0.12.0',
        'tensorflow>=2.11.0',
    ],
    py_modules=PY_MODULES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    cmdclass={
        # Command class instantiated and run during pip install scenarios.
        'build': build,
        'CustomCommands': CustomCommands,
    }
)

