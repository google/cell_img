"""Functions for building files to drive echo robots.

The usage of this module covers two main tasks:
1. Creation of a randomized platemap. For these tasks, the desired set of
   wells is specified (i.e. the compounds, concentrations, and number of
   replicates) and then the wells are placed at random positions within the
   available set of wells.
1. Creation of the echo transfer file. Given the map of the source transfer
   plate and the map of the destination plate (i.e. the one created in the
   first step), these functions create a CSV file that instructs the Echo
   robot on the creation of the given platemap.

In the creation of the echo transfer file, there are two types of transfers:
1. **Dilution plate** : In this setup, the compounds in the source plate are
   all assumed to be at the same concentration, and the destination plate
   creates dilutions of these compounds using DMSO.
     * For this approach, use 'build_source_map' and 'create_echo_transfer_list'
     to set up the echo transfer information.
1. **No Dilutions** : In this setup, the source plate is assumed to already
   contain the exact compounds and concentrations needed for the destination
   plate. In this setup, it is assumed that the same volume will be used for
   each well in the destination plate.
     * In these cases, the echo is often used to add drug dosage to wells with
     existing volumes, so a dilution constant is used to convert from the
     concentrations on the source plate to the final concentration on the
     destination plate. (For example, if you are transferring 100 nl into a
     well that already has 49.9 ul, this is a 500-fold dilution, and a compound
     that should be 0.1 uM on the destination plate would be expected to be at
     50 uM on the source plate.)
     * For this approach, use 'build_source_map_no_dilution' and
     'create_echo_transfer_list_no_dilution' functions.
1. If neither of these assumptions fit your use case, you will need to write
   your own echo transfer functions using the functions above as examples.

In both dilution and no dilution setups, print the echo transfer CSV by passing
the echo transfer list to the 'build_echo_transfer_str' function.

See the example colab for usage examples and more documentation.

This module uses the following assumptions:
* A 'group' is a set of wells for a common purpose, for example control or
  sample wells. A group has a set of compounds, a set of concentrations, and
  a number of replicates for each compound/concentration in the group.
* A 'well_type' defines the group, compound, and concentration, for example
  'control|compound_1|0.1'. The generated platemap will have a number of wells
  of each well_type equal to the number of replicates for that compound/
  concentration in that well.
"""

import collections
import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# The typical solvent used is DMSO. If you want to use another solvent,
# change the name here. The name used here will be treated as a special
# compound, wells with this compound will have their concentrations ignored.
DMSO_COMPOUND = 'DMSO'

# If there are more wells in the available well list than there
# are wells required by the groups of compounds/concentrations/replicates
# they are 'empty wells'. If no specific specification is provided for these
# empty wells, then this specification is used.
DEFAULT_EMPTY_WELL_SPEC = ('empty_well', DMSO_COMPOUND, 1.)


def _is_valid_concentration(concentration: Any) -> bool:
  """Return the valid float concentration, or None if it is not valid."""
  if not (isinstance(concentration, int) or isinstance(concentration, float)):
    return False
  if concentration < 0:
    return False

  return True


def build_well_type(group: str, compound: str, concentration: float) -> str:
  return '%s|%s|%.3f' % (group, compound, concentration)


def _well_row_from_well(well: str) -> str:
  """Parses the row from the well name.

  This function currently assumes the row is the first letter of the well name,
  e.g. "A" is the row for well "A10". If you are using something other than a
  standard 96 or 384 well plate, you will need to change this.

  Args:
    well: The string name of the well, e.g. "A10"

  Returns:
    The string indicating the row of the well, e.g. "A".
  """
  return well[0]


def _well_col_from_well(well: str) -> str:
  """Parses the column from the well name.

  This function currently assumes the column the end of the well name,
  e.g. "10" is the column for well "A10". If you are using something other than
  a standard 96 or 384 well plate, you will need to change this.

  Args:
    well: The string name of the well, e.g. "A10"

  Returns:
    The string indicating the column of the well, e.g. "10".
  """
  return well[1:]


def validate_platemap_setup(concentrations_per_group_map: Dict[str,
                                                               List[float]],
                            compounds_per_group_map: Dict[str, List[str]],
                            num_replicates_map: Dict[Tuple[str, float], float],
                            available_wells: List[str],
                            non_random_wells: List[Tuple[str, str, str, float]],
                            extra_well_spec: Tuple[str, str, float]):
  """Validates that parameters do not have any errors."""

  if set(concentrations_per_group_map.keys()) != set(
      compounds_per_group_map.keys()):
    raise ValueError('The keys (types) for concentrations_per_type_map and '
                     'compounds_per_type_map need to be identical.')

  for group, conc_list in concentrations_per_group_map.items():
    for concentration in conc_list:
      if not _is_valid_concentration(concentration):
        raise ValueError('Group %s has invalid concentration %s' %
                         (group, concentration))
      if ('%.3f' % concentration) == '0.000':
        raise ValueError(
            'Concentrations must be above 0.001. Found %f for well %s.' %
            (concentration, group))
      if (group, concentration) not in num_replicates_map:
        raise ValueError(
            'Every concentration of every group must have num replicates set '
            'in the num_replicates_map')

  num_wells_used = len(non_random_wells)
  for group, conc_list in concentrations_per_group_map.items():
    num_compounds_in_group = len(compounds_per_group_map[group])
    well_count_for_group = 0
    for concentration in conc_list:
      well_count_for_group += num_replicates_map[
          (group, concentration)] * num_compounds_in_group
    print('The group "%s" has %d compounds, at %d different concentrations '
          'using %d wells total' %
          (group, num_compounds_in_group, len(conc_list), well_count_for_group))
    num_wells_used += well_count_for_group

  # Check the non_random_wells to make sure they are valid
  num_available_wells = len(available_wells)
  not_found_wells = []
  for (well, _, _, concentration) in non_random_wells:
    if well not in available_wells:
      not_found_wells.append(well)
    if not _is_valid_concentration(concentration):
      raise ValueError('Non-random well %s has invalid concentration %s' %
                       (well, concentration))
  if len(not_found_wells) >= 1:
    raise ValueError(
        'Non random well%s %s %s not in the set of available wells.' %
        ('' if len(not_found_wells) == 1 else 's', not_found_wells,
         'is' if len(not_found_wells) == 1 else 'are'))

  # check the extra well configuration
  if extra_well_spec:
    if not _is_valid_concentration(extra_well_spec[2]):
      raise ValueError(
          'The extra well specification has an invalid concentration %s' %
          (extra_well_spec[2]))

  print('Your compound map used up %d wells and you have %d sample '
        'wells.' % (num_wells_used, num_available_wells))
  if num_available_wells == num_wells_used:
    print('Your configuration uses all %d wells, leaving no extras' %
          num_available_wells)
  elif num_available_wells > num_wells_used:
    if not extra_well_spec:
      print('You will have %d unused wells that will be DMSO only.' %
            num_available_wells - num_wells_used)
    else:
      print('You will have %d unused wells that will be given '
            'compound "%s" at concentration "%.3f"' %
            (num_available_wells - num_wells_used, extra_well_spec[1],
             extra_well_spec[2]))
  else:
    raise ValueError('Your configuration uses %d wells but you only have %d '
                     'sample wells!' % (num_wells_used, num_available_wells))


def build_randomized_rep_dictionary(
    num_replicates_map: Dict[Tuple[str, float], int],
    total_num_wells: int,
    compounds_per_group_map: Dict[str, List[str]],
    concentrations_per_group_map: Dict[str, List[float]],
    extra_well_spec: Optional[Tuple[str, str, float]] = None
) -> Dict[Tuple[str, str, float], int]:
  """Returns dictionary of compound/conc name to number of replicates.

  The key in the dictionary is (compound_group, compound, concentration), for
  example "sample|drug_1|1.000". The value in the dictionary is a simple
  integer for the number of wells with those contents.

  Args:
    num_replicates_map: map describing number of wells for each
      compound/concentration.
    total_num_wells: integer total number of wells available. This does not
      include the control wells, this function is only concerned with the
      samples.
    compounds_per_group_map: A dictionary that matches each compound_group to a
      list of compounds. For example: "sample":['compound1', 'compound2'].
    concentrations_per_group_map: A dictionary that matches each compound_group
      to a list of concentrations. For example: "sample":[1.0, 10]
    extra_well_spec: Tuple with the [group, compound, concentration] to use for
      any remaining wells. Can be None.
  Returns: A dictionary that maps each particular group of well (as identified
    by its compound_group, compound, and concentration) to a number of wells for
    that group of well. This dictionary does NOT include the non-random wells
    that have set positions.
  """

  replicates_map = {}

  compound_groups = list(concentrations_per_group_map.keys())
  num_wells_used = 0
  for group in compound_groups:
    for compound in compounds_per_group_map[group]:
      for concentration in concentrations_per_group_map[group]:
        num_replicates = num_replicates_map[(group, concentration)]
        replicates_map[(group, compound, concentration)] = num_replicates
        num_wells_used += num_replicates

  if num_wells_used > total_num_wells:
    raise ValueError(
        'Not enough wells! Used %d in the compound map but there are only %d' %
        (num_wells_used, total_num_wells))

  if total_num_wells > num_wells_used:
    if not extra_well_spec:
      extra_well_spec = DEFAULT_EMPTY_WELL_SPEC
    print('Used %d wells building the plate map, leaving %d wells for '
          'control: %s' %
          (num_wells_used, total_num_wells - num_wells_used, extra_well_spec))
    replicates_map[extra_well_spec] = total_num_wells - num_wells_used

  if sum(replicates_map.values()) != total_num_wells:
    raise ValueError('Filled %d wells, but expected to fill the full %d.' %
                     (sum(replicates_map.values()), num_wells_used))

  print('Randomized platemap will have %d wells' % sum(replicates_map.values()))

  return replicates_map


def _assign_one(assignment_map: Dict[str, Any], well: str, group: str,
                compound: str, concentration: float):
  """Adds one well worth of data to the assignment_map.

  The assignment_map dictionary will later be turned into the platemap
  dataframe.
  Each key in the dictionary will be one column in the dataframe, and each
  value in the dictionary is a list with all the values for that column.

  This function does not validate the values sent in (i.e. there is no
  validation that "well" is a valid well name).

  Args:
    assignment_map: A dictionary with all the columns describing a row.
    well: The string name of a well, like "A02".
    group: The string name of the group, like "control".
    compound: The string name of the compound to be put into this well.
    concentration: The float value for the concentration of the compound in this
      well.
  """
  well_type = build_well_type(group, compound, concentration)
  well_row = _well_row_from_well(well)
  well_col = _well_col_from_well(well)
  assignment_map['well'].append(well)
  assignment_map['well_row'].append(well_row)
  assignment_map['well_col'].append(well_col)
  assignment_map['well_type'].append(well_type)
  assignment_map['group'].append(group)
  assignment_map['compound'].append(compound)
  assignment_map['concentration'].append(concentration)


def build_platemap(replicates_map: Dict[str, int],
                   non_random_wells: List[Tuple[str, str, str, float]],
                   available_wells: List[str]) -> pd.DataFrame:
  """Puts each compound/concentration into a well.

  Args:
    replicates_map: Dictionary, generally built in build_rep_dictionary, that
      maps each compound_group|compound|concentration to a number of replicates.
      This map only covers the sample wells, not the control wells. The
      replicates_map should include all the wells put into the non-control wells
      of the platemap. This function will not fill in empty sample wells.
    non_random_wells: Any hard-coded wells to use. This is a list where each
      value is a tuple of (well name, group, compound, concentration), like [
      ('A01', 'special', 'my_favorite_drug', 0.1) ]
    available_wells: List of well names to put the samples described in the
      replicates map into. Like ["C03", "C04", "C05"].

  Returns:
    platemap_df: A Pandas dataframe with one row per well on the plate. The
      columns in the dataframe are well, well_row, well_col, well_type,
      actives, compound, concentration.
      (well_row and well_col are used for visualizing the platemap. The other
      columns are described in _assign_one above.)
  """
  assignment_map = {
      'well': [],
      'well_row': [],
      'well_col': [],
      'well_type': [],
      'group': [],
      'compound': [],
      'concentration': []
  }
  shuffled_wells = copy.copy(available_wells)
  random.shuffle(shuffled_wells)

  # first, assign the non_random_wells into their position
  for (well_name, group, compound, concentration) in non_random_wells:
    if well_name not in shuffled_wells:
      raise ValueError('Non-random well %s is not in the list of available '
                       'wells.' % (well_name))

    _assign_one(assignment_map, well_name, group, compound, concentration)
    shuffled_wells.remove(well_name)

  print('Assigned %d non-random wells. Assigning %d randomized wells '
        'into the remaining %d available wells.' %
        (len(non_random_wells), sum(
            replicates_map.values()), len(shuffled_wells)))

  # then assign the randomized wells
  for (group, compound,
       concentration), num_replicates in replicates_map.items():
    for _ in range(num_replicates):
      _assign_one(assignment_map, shuffled_wells.pop(), group, compound,  # pytype: disable=wrong-arg-types  # mapping-is-not-sequence
                  concentration)

  return pd.DataFrame(assignment_map)


def build_source_map(
    source_df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
  """Creates the maps from compound to wells with compound and volumes.

  Args:
    source_df: Pandas dataframe with columns 'compound', 'well', and 'volume'.

  Returns:
    echo_source_map: Dictionary where the key is the compound and the value
      is the list of wells containing that compound in the source plate.
    volume_remaining_per_source_well_map: Dictionary where they key is the
      well in the source plate and the value is the float volume remaining
      in the well.
  """
  echo_source_map = {}
  volume_remaining_per_source_well_map = {}

  for _, row in source_df.iterrows():
    c = row['compound']
    # don't use defaultdict because we don't want to add compounds accidentally
    if c not in echo_source_map:
      echo_source_map[c] = []
    echo_source_map[c].append(row['well'])
    volume_remaining_per_source_well_map[row['well']] = float(row['volume'])

  return echo_source_map, volume_remaining_per_source_well_map


def _get_source_well_for_compound(source_map, compound, vol_to_transfer,
                                  volume_remaining_per_source_well_map,
                                  volume_used_per_source_well_map):

  for w in source_map[compound]:
    amount_remaining = volume_remaining_per_source_well_map[w]
    if amount_remaining >= vol_to_transfer:
      return w

  raise ValueError(
      'There appear to be no more wells with at least %.4f of %s. Was in %s,\n'
      'have used volumes %s\n'
      'have remaining volumes %s' %
      (vol_to_transfer, compound, source_map[compound],
       volume_used_per_source_well_map, volume_remaining_per_source_well_map))


def _transfer_one_compound(source_plate_barcode, dest_plate_barcode, source_map,
                           compound, dest_well, vol_to_transfer, transfer_list,
                           volume_per_compound_map,
                           volume_remaining_per_source_well_map,
                           volume_used_per_source_well_map,
                           volume_per_dest_well_map):
  """Adds one compound transfer information to the variables."""

  source_well = _get_source_well_for_compound(
      source_map, compound, vol_to_transfer,
      volume_remaining_per_source_well_map, volume_used_per_source_well_map)
  transfer_list.append((source_plate_barcode, source_well, dest_plate_barcode,
                        dest_well, vol_to_transfer))

  volume_per_compound_map[compound] += vol_to_transfer
  volume_used_per_source_well_map[source_well] += vol_to_transfer
  volume_remaining_per_source_well_map[source_well] -= vol_to_transfer
  volume_per_dest_well_map[dest_well] += vol_to_transfer


def create_echo_transfer_list(
    source_plate_barcode: str, dest_plate_barcode: str,
    source_map: Dict[str, List[str]], well_df: pd.DataFrame,
    volume_per_compound_map: Dict[str, float],
    volume_remaining_per_source_well_map: Dict[str, float],
    volume_used_per_source_well_map: Dict[str, float],
    total_vol_per_dest_well: Dict[str, float],
    source_compound_concentration: float,
    min_pipet_volume: float
    ) -> Tuple[List[Tuple[str, str, float]], Dict[str, float]]:
  """Based on the well_df and the source compounds, creates the echo info.

  Args:
    source_plate_barcode: String indicating the barcode on the source plate, for
      the echo robot.
    dest_plate_barcode: String indicating the barcode on the destination plate,
      for the echo robot.
    source_map: Dictionary where the key is the compound and the value is the
      list of wells containing that compound in the source plate.
    well_df: The pandas dataframe with the destination plate map, for example
      the return from the build_platemap function. Has columns compound,
      concentration, and well.
    volume_per_compound_map: Dictionary where the key is the compound and the
      value is the total volume of the given compound that has been moved to the
      destination plate.
    volume_remaining_per_source_well_map: Dictionary where they key is the well
      in the source plate and the value is the float volume remaining in the
      well.
    volume_used_per_source_well_map: Dictionary where they key is the well in
      the source plate and the value is the float volume used from the well.
    total_vol_per_dest_well: Dictionary where the value is the well in the
      destination plate and the value is the total volume in that well.
    source_compound_concentration: The concentration of the compounds in the
      source plate.
    min_pipet_volume: The minimum volume that the robot can accurately pipet.

  Returns:
    transfer_list: A list of tuples indicating source plate and well,
      destination plate and well, and volume to transfer.
    volume_per_dest_well_map: A dictionary where the key is the destination
      well name and the value is the float volume remaining in that well.
  """
  # create a list of source_plate_barcode, source_well, dest_plate_barcode,
  # dest_well, volume
  transfer_list = []
  volume_per_dest_well_map = collections.defaultdict(int)
  for _, row in well_df.iterrows():
    compound = row['compound']
    conc = row['concentration']
    compound_vol = 0
    # first, add any drug compound volume required
    if compound != DMSO_COMPOUND:
      compound_vol = total_vol_per_dest_well * (
          conc / source_compound_concentration)
      _transfer_one_compound(source_plate_barcode, dest_plate_barcode,
                             source_map, compound, row['well'], compound_vol,
                             transfer_list, volume_per_compound_map,
                             volume_remaining_per_source_well_map,
                             volume_used_per_source_well_map,
                             volume_per_dest_well_map)
      if compound_vol < min_pipet_volume:
        raise ValueError(
            'Moving %.4f of %s is to make a final concentration of %.4f is less than '
            'the echo minimum transfer volume' % (compound_vol, compound, conc))

    # then fill up to the total volume using DMSO
    dmso_vol = total_vol_per_dest_well - compound_vol

    if dmso_vol < 0:
      raise ValueError(
          'The compound volume of %.3f required to get concentration %.3f is '
          'more than the total volume of %.3f' %
          (compound_vol, conc, total_vol_per_dest_well))
    elif dmso_vol != 0:
      _transfer_one_compound(source_plate_barcode, dest_plate_barcode,
                             source_map, DMSO_COMPOUND, row['well'], dmso_vol,
                             transfer_list, volume_per_compound_map,
                             volume_remaining_per_source_well_map,
                             volume_used_per_source_well_map,
                             volume_per_dest_well_map)

  return (transfer_list, volume_per_dest_well_map)  # pytype: disable=bad-return-type  # mapping-is-not-sequence


def build_source_map_no_dilution(
    source_df: pd.DataFrame, dilution_constant: float
) -> Tuple[Dict[Tuple[str], List[str]], Dict[str, float]]:
  """Creates the maps from compound|conc to wells with compound and volumes.

  These source maps include concentration in the well identification with the
  assumption that the exact final destination concentration is already
  available in the source plate and no dilution is necessary.

  The key used is compound|conc where compound is the name of the compound
  and conc is the float (to three decimal places) concentration of the compound
  in the final destination well, given the dilution constant.

  For example, if the source plate compound is at 10 mM but the transfer
  volume is 100 nl and we are transferring into 49.9 ul of existing liquid
  the dilution constant will be 500x so the final concentration would be 20 uM.

  Args:
    source_df: Pandas dataframe with columns 'compound', 'concentration',
      'well', and 'volume'.
    dilution_constant: Float value that the initial concentration will be
      diluted by in the source plate. For example, transferring 100 nl into 49.9
      ul will give a dilution constant of 500x.

  Returns:
    echo_source_map: Dictionary where the key is the compound|conc and the value
      is the list of wells containing that compound|conc in the source plate.
    volume_remaining_per_source_well_map: Dictionary where they key is the
      well in the source plate and the value is the float volume remaining
      in the well.
  """
  echo_source_map = {}
  volume_remaining_per_source_well_map = {}

  for _, row in source_df.iterrows():
    if row['compound'] == DMSO_COMPOUND:
      c = DMSO_COMPOUND
    else:
      c = '%s|%.3f' % (row['compound'],
                       row['concentration'] / dilution_constant)

    # don't use defaultdict because we don't want to add compounds accidentally
    if c not in echo_source_map:
      echo_source_map[c] = []
    echo_source_map[c].append(row['well'])
    volume_remaining_per_source_well_map[row['well']] = float(row['volume'])

  return echo_source_map, volume_remaining_per_source_well_map  # pytype: disable=bad-return-type  # mapping-is-not-sequence


def create_echo_transfer_list_no_dilution(
    source_plate_barcode: str, dest_plate_barcode: str,
    source_map: Dict[str, List[str]], well_df: pd.DataFrame,
    volume_used_per_compound_concentration_map: Dict[str, float],
    volume_remaining_per_source_well_map: Dict[str, float],
    volume_used_per_source_well_map: Dict[str, float],
    transfer_volume: float
    ) -> Tuple[List[Tuple[str, str, float]], Dict[str, float]]:
  """Based on the well_df and the source compounds, creates the echo info.

  Args:
    source_plate_barcode: String indicating the barcode on the source plate, for
      the echo robot.
    dest_plate_barcode: String indicating the barcode on the destination plate,
      for the echo robot.
    source_map: Dictionary where the key is the compound and the value is the
      list of wells containing that compound in the source plate.
    well_df: The pandas dataframe with the destination plate map, for example
      the return from the build_platemap function. Has columns compound,
      concentration, and well.
    volume_used_per_compound_concentration_map: Dictionary where the key is the
      compound and the value is the total volume of the given compound that has
      been moved to the destination plate.
    volume_remaining_per_source_well_map: Dictionary where they key is the well
      in the source plate and the value is the float volume remaining in the
      well.
    volume_used_per_source_well_map: Dictionary where they key is the well in
      the source plate and the value is the float volume used from the well.
    transfer_volume: The float value indicating the amount to transfer from the
      source to the destination plate.

  Returns:
    transfer_list: A list of tuples indicating source plate and well,
      destination plate and well, and volume to transfer.
    volume_per_dest_well_map: A dictionary where the key is the destination
      well name and the value is the float volume remaining in that well.
  """

  # create a list of
  # source_plate_barcode, source_well, dest_plate_barcode, dest_well, volume
  transfer_list = []
  volume_per_dest_well_map = collections.defaultdict(int)
  for _, row in well_df.iterrows():
    if row['compound'] != 'DMSO':
      cc = '%s|%.3f' % (row['compound'], row['concentration'])
      _transfer_one_compound(source_plate_barcode, dest_plate_barcode,
                             source_map, cc, row['well'], transfer_volume,
                             transfer_list,
                             volume_used_per_compound_concentration_map,
                             volume_remaining_per_source_well_map,
                             volume_used_per_source_well_map,
                             volume_per_dest_well_map)

    else:
      _transfer_one_compound(source_plate_barcode, dest_plate_barcode,
                             source_map, DMSO_COMPOUND, row['well'],
                             transfer_volume, transfer_list,
                             volume_used_per_compound_concentration_map,
                             volume_remaining_per_source_well_map,
                             volume_used_per_source_well_map,
                             volume_per_dest_well_map)

  return (transfer_list, volume_per_dest_well_map)  # pytype: disable=bad-return-type  # mapping-is-not-sequence


def build_echo_transfer_str(transfer_list: List[Tuple[str, str, float]],
                            convert_volume_unit_mulitplier: float) -> str:
  """Creates the CSV-formatted transfer file for the echo.

  Args:
    transfer_list: The list of source, destination, and transfer volumes, built
      by create_echo_transfer_list.
    convert_volume_unit_mulitplier: The echo robot requires volumes in nl. This
      may or may not be the units used in the previous manipulations (frequntly
      it is easiest to think in terms of ul). This integer represents the value
      to muliply previous volumes to make them in the correct units for the echo
      robot. Typically, previous work is in ul so this value is set to 1000.

  Returns:
    String representing the CSV ready to be saved for the Echo robot.
  """
  # sort the list by source_well to minimize source plate movements
  transfer_list = sorted(transfer_list, key=lambda x: (x[1], x[3]))

  # print the results
  ret_list = []
  ret_list.append(','.join([
      'Source Plate Barcode', 'Source Well', 'Destination Plate Barcode',
      'Destination Well', 'Transfer Volume'
  ]))
  for row in transfer_list:
    # the transfer volume needs to be in nl, not ul
    ret_list.append(','.join([
        str(x) for x in [
            row[0], row[1], row[2], row[3], row[4] *
            convert_volume_unit_mulitplier
        ]
    ]))

  return '\n'.join(ret_list)
