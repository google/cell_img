"""Tests for idot csv generation."""

import re
from absl.testing import absltest

import numpy as np
import pandas as pd

from cell_img.idot import idot

ROWS_384 = tuple('ABCDEFGHIJKLMNOP')
COLS_384 = tuple(range(1, 25))

EVERY_OTHER_ROW = ['A1', 'C1', 'E1', 'G1', 'I1', 'K1', 'M1', 'O1', 'B1', 'D1',
                   'F1', 'H1', 'J1', 'L1', 'N1', 'P1',
                   'A2', 'C2', 'E2', 'G2', 'I2', 'K2', 'M2', 'O2', 'B2', 'D2',
                   'F2', 'H2', 'J2']


EVERY_ROW_96 = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'A2', 'B2']


class IdotPlateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    plates, wells = idot.zip_plates_and_wells(range(6), ROWS_384, COLS_384)
    self.plates_df = pd.DataFrame(data={idot.TARGET_PLATE: plates,
                                        idot.TARGET_WELL: wells})

    self.plates_df[idot.DRUG] = 'drug'

  def test_duplicate_drug_per_plate(self):
    drug_split_df = idot.duplicate_drug_per_plate(self.plates_df, 'drug', 3)

    split_df_dict = {drug: df for drug, df in
                     drug_split_df.groupby(idot.DRUG)[idot.TARGET_PLATE]}

    self.assertEqual(set(split_df_dict['drug0'].unique()),
                     set(['0', '1', '2']))
    self.assertEqual(set(split_df_dict['drug1'].unique()),
                     set(['3', '4', '5']))

  def test_split_target_df_by_plates(self):
    splits = idot.split_target_df_by_plates(self.plates_df, 2)
    self.assertLen(splits, 3)

    for i, split in enumerate(splits):
      self.assertEqual(set(split[idot.TARGET_PLATE].unique()),
                       set((np.arange(2) + 2 * i).astype(str)))

  def test_sort_96_with_384_wells(self):
    shuffle = self.plates_df.sample(frac=1.)
    sort_df = idot.sort_96_with_384_wells(shuffle)

    self.assertEqual(list(sort_df[idot.TARGET_PLATE].unique()),
                     ['0', '1', '2', '3', '4', '5'])
    self.assertSequenceStartsWith(
        EVERY_OTHER_ROW, list(sort_df[idot.TARGET_WELL].values))

  def test_prepend_source_name(self):
    plates_df = pd.DataFrame({idot.SOURCE_PLATE: range(10)})
    prefix = 'prefix'

    orig_source_names = plates_df[idot.SOURCE_PLATE].values.copy()
    idot.prepend_source_name(plates_df, prefix)

    prepend_dict = dict(zip(orig_source_names,
                            plates_df[idot.SOURCE_PLATE].values))
    golden_dict = {p: f'{prefix}{p}' for p in orig_source_names}

    self.assertDictEqual(prepend_dict, golden_dict)


class IdotMultiDrugPlateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    plates, wells = idot.zip_plates_and_wells(range(6), ROWS_384, COLS_384)
    self.plates_df = pd.DataFrame(data={idot.TARGET_PLATE: plates,
                                        idot.TARGET_WELL: wells})

    self.plates_df[idot.DRUG] = [f'drug{i}' for i in range(len(self.plates_df))]
    self.plates_df[idot.UL] = 0.123
    self.plates_df[idot.LIQUID_NAME] = 'DMSO'

    self.source_plates, self.target_plates = idot.make_source_and_target_plates(
        [self.plates_df])

    self.map_df = idot.map_source_to_target(self.source_plates,
                                            self.target_plates)

  def test_map_plates(self):

    pd.testing.assert_series_equal(self.source_plates[idot.DRUG],
                                   self.target_plates[idot.DRUG])

    self.assertSequenceStartsWith(
        EVERY_OTHER_ROW, list(self.target_plates[idot.TARGET_WELL].values))

    self.assertSequenceStartsWith(
        EVERY_ROW_96, list(self.source_plates[idot.SOURCE_WELL].values))

  def test_map_source_to_target(self):

    self.assertSequenceStartsWith(
        EVERY_OTHER_ROW, list(self.map_df[idot.TARGET_WELL].values))

    self.assertSequenceStartsWith(
        EVERY_ROW_96, list(self.map_df[idot.SOURCE_WELL].values))

    self.assertCountEqual(
        self.map_df.columns,
        [idot.SOURCE_PLATE, idot.SOURCE_WELL, idot.SOURCE_WELL_INDEX,
         idot.DRUG, idot.TARGET_PLATE, idot.TARGET_WELL, idot.UL,
         idot.LIQUID_NAME])

  def test_condense_source_plates(self):

    # Now there should only be one unique drug per plate
    self.plates_df[idot.DRUG] = 'drug' + self.plates_df[idot.TARGET_PLATE]

    target_dfs = idot.split_target_df_by_plates(self.plates_df, 2)

    source_df, _ = idot.make_source_and_target_plates(target_dfs)

    self.assertEqual(list(source_df[idot.SOURCE_PLATE].values),
                     ['1', '1', '2', '2', '3', '3'])

    self.assertEqual(list(source_df[idot.SOURCE_WELL].values),
                     ['A1', 'B1', 'A1', 'B1', 'A1', 'B1'])

    idot.condense_source_plates(source_df)

    self.assertEqual(list(source_df[idot.SOURCE_PLATE].values),
                     [1, 1, 1, 1, 1, 1])

    self.assertEqual(list(source_df[idot.SOURCE_WELL].values),
                     ['A1', 'B1', 'C1', 'D1', 'E1', 'F1'])

  def test_mapped_df_to_idot_csv(self):

    # End-to-end test.

    # Preamble appears at the beginning.  Source_target_heading appears every
    # time the idot file switches plates.  This test verifies that the plates
    # are switched correctly in the output file.

    preamble = '<<BEGIN>>'
    source_target_heading = '<<{src}:{target}:{plate_type}>>'

    prime_plate_type = 'PRIME_PLATE'
    target_plate_type = 'TARGET_PLATE'

    csv_out = idot.mapped_df_to_idot_csv(
        self.map_df,
        idot_heading=preamble,
        source_target_heading=source_target_heading,
        prime_ul=0.123,
        prime_plate_type=prime_plate_type,
        target_plate_type=target_plate_type)

    headings = re.findall(r'(<<.*?>>)', csv_out)

    self.assertEqual(headings[0], preamble)

    heading_series = pd.Series(headings[1:])
    heading_df = heading_series.str.extract(r'\<\<(.*?)\:(.*?)\:(.*?)\>\>')

    heading_df.columns = [idot.SOURCE_PLATE, idot.TARGET_PLATE, 'TYPE']

    # First time source plate is used it should be primed.
    first_source = heading_df.groupby(idot.SOURCE_PLATE).first()

    self.assertTrue(
        first_source[idot.TARGET_PLATE].str.startswith('prime_plate').all())
    self.assertTrue((first_source.TYPE == prime_plate_type).all())

    # prime plates should never be reused.
    self.assertLen(first_source, first_source[idot.TARGET_PLATE].nunique())

    # prime plates should only be used during first time.
    self.assertLen(first_source, (heading_df.TYPE == prime_plate_type).sum())

    # Test target plates after priming.
    target_df = heading_df.loc[heading_df.TYPE == target_plate_type]

    # Repeat of target plates should be appended with 'A', 'B', 'C'
    self.assertSequenceStartsWith(['0A', '0B', '0C', '0D',
                                   '1A', '1B', '1C', '1D',
                                   '2A', '2B', '2C', '2D',
                                   '3A', '3B', '3C', '3D',
                                   '4A', '4B', '4C', '4D',
                                   '5A', '5B', '5C', '5D'],
                                  list(target_df[idot.TARGET_PLATE].values))


class SourceIndexPlateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    plates, wells = idot.zip_plates_and_wells(range(6), ROWS_384, COLS_384)
    self.plates_df = pd.DataFrame(data={idot.TARGET_PLATE: plates,
                                        idot.TARGET_WELL: wells})

    self.plates_df[idot.DRUG] = [f'drug{i // 8}'
                                 for i in range(len(self.plates_df))]
    self.plates_df[idot.UL] = 25
    self.plates_df[idot.LIQUID_NAME] = 'DMSO'

    self.source_plates, self.target_plates = idot.make_source_and_target_plates(
        [self.plates_df])

    self.map_df = idot.map_source_to_target(self.source_plates,
                                            self.target_plates)

  def test_source_well_index(self):
    self.assertTrue(
        (self.source_plates[idot.UL] <= idot.MAX_WORKING_VOLUME_UL).all())

    self.assertSequenceStartsWith(
        EVERY_OTHER_ROW, list(self.map_df[idot.TARGET_WELL].values))

    self.assertSequenceStartsWith(
        ['A1', 'A1',
         'B1', 'B1',
         'C1', 'C1',
         'D1', 'D1',
         'E1', 'E1',
         'F1', 'F1',
         'G1', 'G1',
         'H1', 'H1',
         'A2', 'A2',
         'B2', 'B2'],
        list(self.map_df[idot.SOURCE_WELL].values))


if __name__ == '__main__':
  absltest.main()
