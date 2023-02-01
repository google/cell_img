"""Utilities to record binary flag info to aid data provenance.

The primary goal of this package is to create a text file.
"""

from absl import flags
from absl import logging

from cell_img.common import io_lib

FLAGS = flags.FLAGS


def write_all_flags_to_text_file(output_filename: str, module_name: str = ''):
  """Write all flags and their values to a text file.

  If module_name is given, only the desired module flags are written down, if
  not output will contain all the flags and their values involved in the
  pipeline.
  Args:
    output_filename: string (output-file CNS location).
    module_name: name of the Python module used for getting the defined flags.

  Returns:
    None.
  """

  logging.info('Write flags to:\n%s', output_filename)
  if module_name:
    module_flags_list = [
        flags for module, flags in FLAGS.flags_by_module_dict().items()
        if module_name in module
    ]
  else:
    module_flags = [
        flags for module, flags in FLAGS.flags_by_module_dict().items()
    ]

  module_flags = []
  for sublist in module_flags_list:
    module_flags.extend(sublist)

  io_lib.write_json_file(
      ['%s: %s \n' % (flag.name, flag.value) for flag in module_flags],
      output_filename)

