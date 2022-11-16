"""Tests for plate_utils."""

import string

from absl.testing import absltest
from absl.testing import parameterized
from cell_img.common import plate_utils


class PlateUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('A3', 1, True, 'A'),
      ('A13', 1, True, 'A'),
      ('K24', 2, True, '.K'),
      ('K24', 3, True, '..K'),
      ('Z2', 3, True, '..Z'),
      ('AA1', 1, False, 'AA'),
      ('AA001', 1, False, 'AA'),
      ('AA001', 2, True, 'AA'),
      ('AA011', 3, True, '.AA'),
      ('.AA011', 1, False, 'AA'),
      ('..AA011', 2, True, 'AA'),
      ('...AA011', 3, True, '.AA'),
      ('...AA011', 4, True, '..AA'),)
  def test_row_str(self, well, min_width, strict, expected):
    actual = plate_utils.get_row_str(well, min_width, strict=strict)
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      # No row.
      ('3',),
      ('005',),
      # No column.
      ('A',),
      ('AZ',),
      # Bad row.
      ('A1Z13',),
      ('a13',),
      ('!024',),
      # Bad column.
      ('A01a',),
      ('K0',),
      ('K00',),
      ('M13.',),
      ('P-35',),
      # Both bad.
      ('',),
      ('@#<$',),)
  def test_bad_well(self, badwell):
    with self.assertRaisesRegex(ValueError, 'Invalid well representation'):
      plate_utils.get_row_str(badwell)
    with self.assertRaisesRegex(ValueError, 'Invalid well representation'):
      plate_utils.get_column_str(badwell, 2)

  @parameterized.parameters(
      ('A1', 1, '1'),
      ('A1', 2, '01'),
      ('A1', 3, '001'),
      ('M14', 2, '14'),
      ('M014', 2, '14'),
      ('M0010', 2, '10'),
      ('M0010', 3, '010'),
      ('.ZM0010', 5, '00010'),)
  def test_column_str(self, well, min_width, expected):
    actual = plate_utils.get_column_str(well, min_width)
    self.assertEqual(actual, expected)

  def test_row_int(self):
    for ix, letter in enumerate(string.ascii_uppercase, start=1):
      actual1 = plate_utils.get_row_int('%s05' % letter)
      self.assertEqual(actual1, ix)
      actual2 = plate_utils.get_row_int('.%s06' % letter)
      self.assertEqual(actual2, ix)
      actual3 = plate_utils.get_row_int('..%s7' % letter)
      self.assertEqual(actual3, ix)

  @parameterized.parameters(
      ('AA3', 27),
      ('.AZ3', 52),
      ('BA5', 53),
      ('ZZ013', 702),
      ('..AAA24', 703),
  )
  def test_multirow_int(self, well, expected):
    actual = plate_utils.get_row_int(well)
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      ('A1', 1),
      ('A10', 10),
      ('K024', 24),
      ('ZZ016', 16),)
  def test_column_int(self, well, expected):
    actual = plate_utils.get_column_int(well)
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
