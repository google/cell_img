"""Utility methods for working with plates of wells."""

import re
import string


_WELL_RE = re.compile(r'^\.*(?P<row>[A-Z]+)0*(?P<col>[1-9]\d*)$')


def _get_row_str_raw(well: str) -> str:
  match = _WELL_RE.match(well)
  if match is None:
    raise ValueError('Invalid well representation: %s' % well)
  return match.group('row')


def _get_col_str_raw(well: str) -> str:
  match = _WELL_RE.match(well)
  if match is None:
    raise ValueError('Invalid well representation: %s' % well)
  return match.group('col')


def get_row_str(well: str, min_width: int = 1, strict: bool = True) -> str:
  """Returns the string representation of the row from the well.

  Args:
    well: A string well id.
    min_width: The minimum width of the resulting row.
    strict: (bool) Whether to raise ValueError if the content overflows.

  Returns:
    The string representation of the row. If the row string is natively shorter
    than min_width, it is padded on the left by the "." character.

  Raises:
    ValueError: If either:
      1) The input well is not a valid well representation.
      2) The min_width is insufficiently large and the row overflowed.
  """
  result = _get_row_str_raw(well).rjust(min_width, '.')
  if strict and len(result) != min_width:
    raise ValueError('The row of well string %s could not be padded to exactly '
                     '%d characters.' % (well, min_width))
  return result


def get_column_str(well: str, min_width: int, strict: bool = True) -> str:
  """Returns the string representation of the column from the well.

  Args:
    well: A string well id.
    min_width: The minimum width of the resulting column.
    strict: (bool) Whether to raise ValueError if the content overflows.

  Returns:
    The string representation of the column. If the column string is natively
    shorter than min_width, it is padded on the left by the "0" character.

  Raises:
    ValueError: If either:
      1) The input well is not a valid well representation.
      2) The min_width is insufficiently large and the column overflowed.
  """
  result = _get_col_str_raw(well).zfill(min_width)
  if strict and len(result) != min_width:
    raise ValueError('The column of well string %s could not be padded to '
                     'exactly %d characters.' % (well, min_width))
  return result


def get_row_int(well: str) -> int:
  """Return the integer representation of the row from the well.

  Args:
    well: A string well id.

  Returns:
    The one-based integer representation of the row.
  """
  row_str = get_row_str(well, strict=False)
  retval = 0
  for i, letter in enumerate(reversed(row_str)):
    retval += int(
        (string.ascii_uppercase.index(letter) + 1) *
        len(string.ascii_uppercase)**i)
  return retval


def get_column_int(well: str) -> int:
  """Return the integer representation of the column from the well.

  Args:
    well: A string well id.

  Returns:
    The one-based integer representation of the column.
  """
  return int(get_column_str(well, 1, strict=False))
