"""Utilities for generating idot files."""

import io
import itertools
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd


SOURCE_PLATE = 'source_name'
TARGET_PLATE = 'target_name'

UL = 'Volume [uL]'
SOURCE_WELL_INDEX = 'source_index'
SOURCE_WELL = 'Source Well'
TARGET_WELL = 'Target Well'
LIQUID_NAME = 'Liquid Name'

DRUG = 'drug'
REP = 'rep'

ROWS_96 = tuple('ABCDEFGH')
COLS_96 = tuple(range(1, 13))

MAX_WORKING_VOLUME_UL = 60

HEADING_STRING = '''sep=,
MultiSourceMultiPlate,1.7.2021.1105,<User Name>,02/14/22,11:14 AM'''

SOURCE_TARGET_HEADING_STRING = '''S.100 Plate,{src},,8.00E-05,{plate_type},{target},,Waste Tube
DispenseToWaste=False,DispenseToWasteCycles=3,DispenseToWasteVolume=1e-7,UseDeionisation=True,OptimizationLevel=ReorderAndParallel,WasteErrorHandlingLevel=Ask,SaveLiquids=Always'''


def duplicate_drug_per_plate(rep_df: pd.DataFrame, drug: str,
                             plates_per_drug_instance: int) -> pd.DataFrame:
  """Converts single drug into multiple instances of drug.

  Args:
    rep_df: Standard DataFrame containing drug and target well info.
    drug: name of drug to make multiple instances of
    plates_per_drug_instance: Rate to create instances

  Returns:
    Dataframe with "drug" replaced by "drug0", "drug1" ... "drugn"
  """

  for i, target_plate in enumerate(rep_df[TARGET_PLATE].unique()):
    mask = rep_df[DRUG] == drug
    mask &= rep_df[TARGET_PLATE] == target_plate
    rep_df.loc[mask, DRUG] = f'{drug}{i // plates_per_drug_instance}'
  return rep_df


def split_target_df_by_plates(target_df: pd.DataFrame, plates_per_split: int
                              ) -> List[pd.DataFrame]:
  """Splits into multiple target_dfs based on plates_per_split."""

  df_plates = sorted(target_df[TARGET_PLATE].unique())
  target_df_chunks = list()
  for i in range(0, len(df_plates), plates_per_split):
    # pylint: disable=unused-variable
    query_plates = df_plates[i: i + plates_per_split]
    # pylint: enable=unused-variable
    target_df_chunks.append(
        target_df.query(f'{TARGET_PLATE} == @query_plates'))
  return target_df_chunks


def zip_plates_and_wells(plates: Sequence[Union[str, int]],
                         rows: Sequence[Union[str, int]] = ROWS_96,
                         cols: Sequence[Union[str, int]] = COLS_96
                         ) -> Tuple[Iterable[str], Iterable[str]]:
  """Returns plates and wells ordered by plate, column - row."""

  # Orders by plate, column-row. The idot robot grabs one column at a time so
  # we want the source wells packed into columns.

  product = itertools.product(plates, cols, rows)
  plates, wells = zip(*[(f'{p}', f'{r}{c}') for p, c, r in product])
  return plates, wells


def sort_96_with_384_wells(target_well_df: pd.DataFrame) -> pd.DataFrame:
  """Sorts 384 well plates by every other row to match 96 well pitch."""

  sort_df = target_well_df[TARGET_WELL].str.extract(r'([A-Z])(\d+)')
  sort_df.columns = ['row', 'col']
  sort_df['even'] = (sort_df.row.apply(ord) - ord('A')) % 2
  sort_df['col'] = sort_df['col'].astype(int)

  sort_df['plate'] = target_well_df[TARGET_PLATE]
  sort_df.sort_values(['plate', 'col', 'even', 'row'], inplace=True)
  return target_well_df.loc[sort_df.index].reset_index(drop=True)


def make_source_and_target_plates(target_df_list: List[pd.DataFrame],
                                  max_volume_ul: float = MAX_WORKING_VOLUME_UL
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Generates optimized source and target plates.

  Args:
    target_df_list: A list of standard dataframes.
      Each dataframe will have dedicated source plates generated for it.
    max_volume_ul: maximum volume of a source well.

  Returns:
    concatenated source plates, concatenated sorted target plates
  """

  source_plate_list = list()
  target_sorted_list = list()
  starting_source_plate = 1

  d_si = [DRUG, SOURCE_WELL_INDEX]

  for df in target_df_list:
    df_sorted = sort_96_with_384_wells(df)

    # If the volume of source drug is larger than maximum well volume, then we
    # need to put the drug into multiple wells. SOURCE_WELL_INDEX is used to
    # index which well the drug is sourced from.

    # Calculate in discrete steps to ensure that there is always enough liquid
    # in a source well for each discrete liquid transfer.

    source_well_not_declared = -1
    df_sorted[SOURCE_WELL_INDEX] = source_well_not_declared

    source_index = 0
    while True:
      undeclared = df_sorted[SOURCE_WELL_INDEX] == source_well_not_declared
      no_overflow = df_sorted.groupby(d_si)[UL].cumsum() <= max_volume_ul
      mask = no_overflow & undeclared

      df_sorted.loc[mask, SOURCE_WELL_INDEX] = source_index

      source_index += 1

      if not mask.any():
        break

    source_plate_df = df_sorted.groupby(d_si).sum()
    source_plate_df = source_plate_df.loc[
        df_sorted.set_index(d_si).index].reset_index()
    source_plate_df = source_plate_df.drop_duplicates()

    max_src_plates = np.ceil(len(source_plate_df) / 96).astype(int)
    src_plate_index = np.arange(
        starting_source_plate,
        starting_source_plate + max_src_plates + 1).astype(int)
    fill = np.floor(1 + np.log10(src_plate_index.max())).astype(int)

    src_plate_names = [i.zfill(fill) for i in src_plate_index.astype(str)]
    src_plates, src_wells = zip_plates_and_wells(src_plate_names)

    source_plate_df[SOURCE_PLATE] = list(src_plates)[:len(source_plate_df)]
    source_plate_df[SOURCE_WELL] = list(src_wells)[:len(source_plate_df)]

    source_plate_list.append(source_plate_df)
    target_sorted_list.append(df_sorted)
    starting_source_plate += max_src_plates

  return (pd.concat(source_plate_list, ignore_index=True),
          pd.concat(target_sorted_list, ignore_index=True))


def _condense_source_plates_recursive(sources):
  """Packs wells from multiple unfilled plates into 96 well plates."""

  source_well_count = sources.groupby(SOURCE_PLATE).size()
  condensable_plates = source_well_count.loc[source_well_count < 96]
  for i1, i2 in list(itertools.combinations(condensable_plates.index, 2)):
    len_i1 = source_well_count.loc[i1]
    len_i2 = source_well_count.loc[i2]
    if len_i1 + len_i2 <= 96:  # Enough empty wells on first plate
      i2_mask = sources[SOURCE_PLATE] == i2

      # Move plate2 wells to end of wells on plate1
      well_order = [f'{r}{c}' for c, r in itertools.product(COLS_96, ROWS_96)]
      sources.loc[i2_mask, SOURCE_WELL] = well_order[len_i1: len_i1 + len_i2]

      # Rename plate2 to plate1
      sources.loc[i2_mask, SOURCE_PLATE] = i1

      # Recurse to recalculate remaining wells on newly merged plate.
      _condense_source_plates_recursive(sources)
      break


def condense_source_plates(sources: pd.DataFrame):
  """Packs wells from multiple unfilled plates into full 96 well plates."""

  _condense_source_plates_recursive(sources)

  # Rename source plate names if plates have been subsumed into other plates.
  unique_source_names = sources[SOURCE_PLATE].unique()
  rename = dict(zip(sorted(unique_source_names),
                    (1 + np.arange(len(unique_source_names))).astype(int)))
  sources[SOURCE_PLATE] = sources[SOURCE_PLATE].apply(rename.get)


def map_source_to_target(source_df: pd.DataFrame, target_df: pd.DataFrame
                         ) -> pd.DataFrame:

  source = source_df[[SOURCE_PLATE, SOURCE_WELL, SOURCE_WELL_INDEX, DRUG]]
  target_plate_df = target_df.merge(source, on=[DRUG, SOURCE_WELL_INDEX],
                                    validate='m:1')
  return sort_96_with_384_wells(target_plate_df)


def prepend_source_name(source_df: pd.DataFrame, prepend_str: str):
  source_df[SOURCE_PLATE] = (prepend_str +
                             source_df[SOURCE_PLATE].astype(str))


def _df_to_csv_string(df: pd.DataFrame) -> str:
  buffer = io.StringIO()
  df.to_csv(buffer, index=False)
  return buffer.getvalue()


def mapped_df_to_idot_csv(
    target_plate_df: pd.DataFrame,
    idot_heading: str,
    source_target_heading: str,
    prime_ul: float,
    prime_plate_type: str,
    target_plate_type: str) -> str:
  """Generates csv file for idot machine.

  Args:
    target_plate_df: DataFrame containing SOURCE_PLATE, TARGET_PLATE,
      SOURCE_WELL, UL, LIQUID_NAME and TARGET_WELL columns.
    idot_heading: string which appears at the beginning of the csv file.
    source_target_heading: Formatable string which appears everytime
      source/target plate pairs switch.  Format string may contain the following
      named arguments
        '{src}': source_plate
        '{target}': target_plate
        '{plate_type}': replaced with prime_plate_type for prime plates and
        target_plate_type for target plates
    prime_ul: Volume in uL to prime prime-plates.
    prime_plate_type: Name of plate-type used for priming
    target_plate_type: name of plate-type used for dispensing fluid.

  Returns:
    idot machine readable string.
  """

  csvs = [idot_heading]
  target_ord_suffixes = {k: ord('A')
                         for k in target_plate_df[TARGET_PLATE].unique()}

  source_wells_dict = {k: df
                       for k, df in target_plate_df.groupby(SOURCE_PLATE)}
  already_primed_src_names = set()
  prime_plate_index = 1
  for (src_name, target_name), df in target_plate_df.groupby([SOURCE_PLATE,  # pytype: disable=attribute-error  # pandas-drop-duplicates-overloads
                                                              TARGET_PLATE]):
    if not df.empty:
      if src_name not in already_primed_src_names:
        # dispense to a prime plate the first time src_name is seen
        csvs.append(source_target_heading.format(
            src=src_name, target=f'prime_plate{prime_plate_index}',
            plate_type=prime_plate_type))
        prime_df = source_wells_dict[src_name]
        prime_df = prime_df[[SOURCE_WELL, SOURCE_WELL, UL, LIQUID_NAME]].copy()
        prime_df[UL] = prime_ul
        csvs.append(_df_to_csv_string(prime_df.drop_duplicates()))
        already_primed_src_names |= set([src_name])
        prime_plate_index += 1

      dn = target_name + chr(target_ord_suffixes[target_name])
      target_ord_suffixes[target_name] += 1

      csvs.append(source_target_heading.format(
          src=src_name, target=dn,
          plate_type=target_plate_type))
      csvs.append(
          _df_to_csv_string(df[[SOURCE_WELL, TARGET_WELL, UL, LIQUID_NAME]]))

  return '\n'.join([c.strip() for c in csvs])
