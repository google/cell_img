"""Functions that generate morphological screening batch designs.

This library contains a set of functions that can be used to generate
experimental designs for well plate based experiments that are primarily used in
cellular morphology screening. They are particularly applicable when cells are
analyzed by High Throughput Screening (HTS) and ML based techniques.

ML based methods tend to be very sensitive and can easily pick up on non
relevant signals in cell images such as the row or column a cell was in, thus it
is imperitive to ensure that the signal of interest is not conflated with any
of these known nuisance signals. The functions in this library try to design
experimental layouts that minimize the chances of this happening.

The functions contained within the library create a very specific experimental
design used frequently in our disease model research but they could be
readily adapted for individual use cases. However, it is useful to understand
the use case to aid in the understanding of the library functions.
  These functions design experiments for use in disease model generation. This
involves selecting cell lines from patients with the disease of interest and
pairing them with demographically matched cell lines from patients without the
disease. An example of this type of experiment can be seen in Schiff et al,
Nature Communications 13,1590(2022). The experimental process is as follows:
1. Up to 60 cell lines are placed in individual wells of a 96 well plate (wp).
   Demographically matched healthy/disease cell line pairs are placed in
   adjacent wells to minimize variation in signal between the two cell lines
   caused by plate location. It should be noted that only 60 wells  are used
   with rows A to F and columns 1 to 10 being in use. This is due to the fact
   that only 240 wells within rows C to M and columns 3 to 22 are used in the
   384 well plates. The outer two rings of wells in the 384 plates are not used
   as they produce poor results and have large effects on cell growth &
   morphology. With each well of the 96 well plate 'seeding' four wells per 384
   well plate that means we can only use up to 60 wells in the 384 well plate.
2. Cells are grown and then transferred to a batch of 384 well plates. A batch
   for our purposes denotes 6 x 384 well plates. It should be noted that only
   wells within rows C to M and columns 3 to 22 are used. The cells are
   transferred to the 384 well plates in a unique pattern: cells in 96-A1 are
   transferred to wells 384-C3, 384-C4, 384-D3 & 384; cells in 96-A2 are
   transferred to 384-C5, 384-C6, 384-D5 & 384-D6. This pattern is repeated
   for the entire plate with finally 96-F10 being transferred to 384-M21,
   384-M22, 384-N21 & 384-N22. This unique pattern is a limitation of the
   Dynamic Devices Lynx Liquid Handling Robot used in the protocol. One 96 well
   plate has sufficient cells to populate six 384 well plates in this manner.
3. Unless a smaller number of lines is being used, such that they can replicated
   in the source 96 well plate we strongly recommend using the quadrant_swap
   option available in the functions that ensures no cell line is associated
   with a single well. 384 well plates 1, 2 & 3 are seeded as described above
   but plates 4, 5, & 6 have quadrants of the source plate diagonally swapped
   resulting in an altered layout. Practically this means that cells in rows
   A, B, C columns 1 to 6 of the 96 well plate are placed in rows I to N
   columns 13 to 22 in the 384 well plates. Similarly cells in rows D, E F
   columns 6 to 10 of the 96 well plate are placed in rows A to H columns
   3 to 12 of the 384 well plate. The other two quadrants of the 96 well plate
   are swapped in a similar fashion.
4. With the six 384 well plates filled they can be used for morphology
   screening.
"""


import copy
import random

import numpy as np
import pandas as pd
from pandas.core.base import DataError


def make_96_well_plate(quadrant_swap=False):
  """Generates an "empty" 96 well plate dataframe.

  Columns include well, well_col, well_row, cl_region, pair region &
  disease status of each well.

  Args:
    quadrant_swap: defaulted to false but if set to true will generate the
                   quadrant swap layout.
  Returns:
      A dataframe with the above described columns.
  """

  df = pd.DataFrame()

  df['well'] = _plate_map_96_wells()
  df['well_row'] = df.well.str[0]
  df['well_col'] = df.well.str[1:]
  df['cl_region'] = _plate_regions_96_wells(quadrant_swap)
  df['pair_region'] = _pair_regions_96_wells(quadrant_swap)
  df['disease_state'] = _disease_status_96_wells(quadrant_swap)

  return df


def _plate_map_96_wells():
  """Generates a list of wells in a ninety six well plate.

  Returns:
    An ordered list containing the wells in a ninety six well plate.
  """
  rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
  cols = list(map(str, list(range(1, 13))))
  cols = [x.zfill(2) for x in cols]
  plate_map = []
  for row in rows:
    rpf = [row] * len(cols)
    r_inds = [a + b for (a, b) in zip(rpf, cols)]
    plate_map.append(r_inds)
  map_96 = []
  for sublist in plate_map:
    for item in sublist:
      map_96.append(item)
  return map_96


def _plate_regions_96_wells(quadrant_swap=False):
  """Denotes wells in use by assigning a 'region' number.

  Generates an ordered list, which corresponds positionally to the ordered
  well list generated by _plate_map_96_wells(). It denotes which wells are in
  use by assigning a region number. Wells not in use are denoted by the label
  'unset'.
  Args:
    quadrant_swap: defaulted to false but if set to true will generate the
                   quadrant swap layout.
  Returns:
    An ordered list detailing the region assignments for each well.
  """

  unused = ['unset'] * 24
  region = list(map(str, list(range(1, 61))))
  region = [x.zfill(2) for x in region]

  if quadrant_swap:
    region = (
        region[35:40] + region[30:35] + region[45:50] + region[40:45] +
        region[55:60] + region[50:55] + region[5:10] + region[0:5] +
        region[15:20] + region[10:15] + region[25:30] + region[20:25])

  for x in [60, 60, 50, 50, 40, 40, 30, 30, 20, 20, 10, 10]:
    region.insert(x, 'unset')

  return region + unused


def _disease_status_96_wells(quadrant_swap=False):
  """Denotes the disease status of each well.

  Generates an ordered list, which corresponds positionally to the ordered
  well list generated by _plate_map_96_wells(). The list denotes the status of
  the cells that can go in a particular well. This constraint is enacted to
  ensure disease state is not conflated with well, column or row. Wells that
  are not used have their disease status set to 'unset'.
  Args:
    quadrant_swap: defaulted to false but if set to true will generate the
                   quadrant swap layout.
  Returns:
    An ordered list detailing disease status for each well.
  """
  states_1 = ['healthy', 'disease']
  states_2 = ['disease', 'healthy']
  unused = ['unset'] * 24
  assignments = ((states_1 * 5) + (states_2 * 5) + (states_1 * 5)) * 2
  if quadrant_swap:
    assignments = (assignments[35:40] + assignments[30:35] +
                   assignments[45:50] + assignments[40:45] +
                   assignments[55:60] + assignments[50:55] +
                   assignments[5:10] + assignments[0:5] +
                   assignments[15:20] + assignments[10:15] +
                   assignments[25:30] + assignments[20:25])

  for x in [60, 60, 50, 50, 40, 40, 30, 30, 20, 20, 10, 10]:
    assignments.insert(x, 'unset')

  return assignments + unused


def _pair_regions_96_wells(quadrant_swap=False):
  """Denotes what pair region a well belongs too.

  Generates an ordered list, which corresponds positionally to the ordered
  well list generated by _plate_map_96_wells(). This sets the pairs of wells
  that demographically matched healthy and disease can be assigned too.
  Args:
    quadrant_swap: defaulted to false but if set to true will generate the
                    quadrant swap layout.
  Returns:
    An ordered list setting the pair region id for each well.
  """
  pair_regions = [1, 1, 4, 4, 8, 8, 10, 10, 13, 13,
                  2, 2, 5, 5, 7, 9, 11, 11, 14, 14,
                  3, 3, 6, 6, 7, 9, 12, 12, 15, 15,
                  16, 16, 19, 19, 22, 24, 25, 25, 28, 28,
                  17, 17, 20, 20, 22, 24, 26, 26, 29, 29,
                  18, 18, 21, 21, 23, 23, 27, 27, 30, 30]
  pair_regions = [str(r).zfill(2) for r in pair_regions]
  unset = ['unset'] * 24
  if quadrant_swap:
    pair_regions = (pair_regions[35:40] + pair_regions[30:35] +
                    pair_regions[45:50] + pair_regions[40:45] +
                    pair_regions[55:60] + pair_regions[50:55] +
                    pair_regions[5:10] + pair_regions[0:5] +
                    pair_regions[15:20] + pair_regions[10:15] +
                    pair_regions[25:30] + pair_regions[20:25])

  for x in [60, 60, 50, 50, 40, 40, 30, 30, 20, 20, 10, 10]:
    pair_regions.insert(x, 'unset')

  return pair_regions + unset


def quality_control_cell_line_data(df, seeding_columns):
  """Tests specified columns are suitable for use in optimized seeding.

  Tests the data in columns specified by seeding_columns argument to ensure
  data in those columns is all numeric and there are no missing entries. This
  is a requirement for use in optimized seeding.

  Args:
    df: DataFrame containing data on the cell lines being used in the batch
        design.
    seeding_columns: List containing the name of columns to be used in
                     optimized seeding function.
  Raises:
    Value Error: At least one columns not suitable for optimized seeding.
  """
  unusable_cats = []
  for cat in seeding_columns:
    if _qc_column(df[cat]):
      print(f'{cat} is suitable for use in optimized seeding')
    else:
      print(f'{cat} is not suitable for use in optimized seeding')
      unusable_cats.append(cat)

  if not unusable_cats:
    print("""All user identified categories are suitable for use in
    optimized seeding""")

  elif unusable_cats:
    print(f'''{" & ".join(unusable_cats)} column(s) cannot be used in
          optimized seeding. Columns must be numeric with no missing entries''')
    raise ValueError('At least one columns not suitable for optimized seeding')


def seed_96_well_plate(cell_line_df,
                       optimized_seeding,
                       seed_categories,
                       num_iterations):
  """Function that seeds the cell lines into the ninety six well plates.

  This function seeds the cell lines, passed in the form of a dataframe with
  accompanying metadata into the ninety six well plate dataframe. The user has
  the option to select optimized_seeding. If set to True, optimized seeding
  takes the user defined categories, which must be columns in the the
  cell_line_df, and

  Args:
    cell_line_df: dataframe with cell line names and metadata. Must contain
                  at least two columns with the names 'cell_line_id' &
                  'pair_id'.
    optimized_seeding: Boolean value indicating whether optimized seeding
                       process should be used
    seed_categories: names of columns in the cell_line_df that will be used
                     for optimized seeding.
    num_iterations: number of random seedings performed during the
                    optimized_seeding process.
  Returns:
    Dataframe describing the ninety six well plate and location of cell lines
    within that plate.
  Raises:
    Exception: Raised when the cell_line_df contains more than 60 cell line
               entries.
    Exception: When soptimized seeding is set to True, seed_categories cannot
               be None or an empty list.
  """

  if cell_line_df.shape[0] > 60:
    raise Exception('Only up to 60 cell lines can be used with this method.')

  wp_96 = make_96_well_plate(quadrant_swap=False)
  wp_96_qs = make_96_well_plate(quadrant_swap=True)

  # Merge with cell line df to assign lines to regions
  cl_df = _add_region_assignment(cell_line_df)
  wp_96 = wp_96.merge(cl_df, on=['pair_region', 'disease_state'], how='left')
  wp_96_qs = wp_96_qs.merge(cl_df, on=['pair_region', 'disease_state'],
                            how='left')

  if not optimized_seeding:
    return wp_96
  wp_96_qs = wp_96_qs.merge(
      cl_df, on=['pair_region', 'disease_state'], how='left')
  # first check seed categories
  if seed_categories is None or not seed_categories:
    raise Exception('''When optimized seeding is set to True
                     seed_catgories cannot be empty or set to None''')

  # make sure specified columns are all numeric
  for cat in seed_categories:
    if not _qc_column(cell_line_df[cat]):
      raise DataError(f'''{cat} is not all numeric and cannot be used in
                         optimized seeding''')

  df_variance = []
  dataframes = []
  for _ in range(num_iterations):
    batch_design_df = pd.concat([wp_96, wp_96_qs], ignore_index=True)
    var = _measure_variation(seed_categories, batch_design_df)
    df_variance.append(var)
    dataframes.append(wp_96)
    cl_df = _add_region_assignment(cell_line_df)
    wp_96 = make_96_well_plate(quadrant_swap=False)
    wp_96 = wp_96.merge(cl_df, on=['pair_region', 'disease_state'],
                        how='left')
    wp_96_qs = make_96_well_plate(quadrant_swap=True)
    wp_96_qs = wp_96_qs.merge(
        cl_df, on=['pair_region', 'disease_state'], how='left')

  lowest_cv = min(df_variance)
  lowest_index = df_variance.index(lowest_cv)
  print('min variance is: ', lowest_cv)
  wp_96 = dataframes[lowest_index]

  return wp_96


def _add_region_assignment(cell_line_df):
  """Randomly assigns line pairs to pair regions in ninety six well plates.

  Adds a column to the cell_line_df called pair_region. This corresponds to
  the pair regions in the ninety six well plate dataframe and the pair_region
  numbers are randomly assigned to the cell line pairs. The column is used to
  merge with the ninety six well plates and effectively assign cell lines to
  wells.

  Args:
    cell_line_df: dataframe with cell line names and metadata. Must contain
                  at least two columns with the names 'cell_line_id' &
                  'pair_id'.
  Returns:
    Input dataframe with the extra pair_region column.
  """
  cell_line_df.sort_values(by=['pair_id', 'disease_state'])
  r_assignments = np.arange(1, 31)
  random.shuffle(r_assignments)
  cell_line_df['pair_region'] = np.repeat([str(x).zfill(2) for
                                           x in r_assignments], 2)

  return cell_line_df


def _measure_variation(seed_categories, df):
  """Measures the coefficient of variance in user specified catgories.

  Function that measures the cv in user specified catgories along well, column
  and row axes
  Args:
    seed_categories: These are the names of columns in 'df' that contain the
                     data the function will measure the cv of.
    df: Dataframe reporesenting the nonety six well plate layout under test.
  Returns:
    measured_var_in_cats: Sum of the variance for each category across
                            column, row & well.
  Raises:
    ValueError: Data in seed category may not be numeric
  """
  measured_var_in_cats = 0
  for cat in seed_categories:
    df4s = copy.deepcopy(df)
    df4s = df4s[df4s[cat].notna()]
    col_cv = ((df4s.groupby('well_col')[cat].mean().std()) /
              (df4s.groupby('well_col')[cat].mean().mean()))
    row_cv = ((df4s.groupby('well_row')[cat].mean().std()) /
              (df4s.groupby('well_row')[cat].mean().mean()))
    w_cv = ((df4s.groupby('well')[cat].mean().std()) /
            (df4s.groupby('well')[cat].mean().mean()))
    measured_var_in_cats += (col_cv + row_cv + w_cv)

  return measured_var_in_cats


def _qc_column(col_dat):
  """Checks the data in a DataFrame column is all numeric with no NaNs.

  Args:
    col_dat: Data from the Dataframe column under test, should be in the form
             df['col_name']
  Returns:
    Boolean value indicating if the column data meets requirements of being
    being numeric with no missing values.
  """
  if pd.api.types.is_numeric_dtype(
      col_dat.values) and not col_dat.isnull().values.any():
    return True

  return False


def make_384_wp(wp_96, layout):
  """Create 384 well plate layout from ninety six well plate design.

  Takes the ninety six well df and generates a dataframe representing a 384
  well plate in accordance with the operation of the robot used to transfer
  cells from the ninety six well plate to the three eighty four well plates.
  Args:
    wp_96: df representing the ninety six well plate
    layout: must be set to 'l1' for the standard layout or 'l2' for quadrant
            swap layout.
  Returns:
    Dataframe representing the layout of the three eighty four well plate.
  """

  # Create the 384 wp well layout
  layout_384 = _plate_layout_384()
  source_96 = _conversion_col_384_to_96(layout)
  df_384 = pd.DataFrame(
      list(zip(layout_384, source_96)), columns=['well', 'ns_source'])
  df_384['well_row'] = df_384.well.str[0]
  df_384['well_col'] = df_384.well.str[1:]

  # Clean up 96wp dataframe and remove columns not needed
  wp_96 = wp_96.drop(['well_row', 'well_col'], axis=1)
  wp_96 = wp_96.rename(columns={'well': 'ns_source'})

  # Merge the dfs to effectively 'seed' the 384 well plate
  df_384 = df_384.merge(wp_96, on=['ns_source'], how='left')
  df_384 = df_384.fillna('no_cells')

  return df_384


def _plate_layout_384():
  """Makes the list of wells in the 384 well plate.

  Returns:
    List of well names in a 384 well plate in order
  """
  rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P']
  cols = list(map(str, list(range(1, 25))))
  cols = [x.zfill(2) for x in cols]
  plate_map = []
  for row in rows:
    rpf = [row] * len(cols)
    r_inds = [a + b for (a, b) in zip(rpf, cols)]
    plate_map.append(r_inds)

  plate_map_384 = []
  for sublist in plate_map:
    for item in sublist:
      plate_map_384.append(item)

  return plate_map_384


def _conversion_col_384_to_96(layout):
  """Defines a list that identifies the 96wp source well for 384wp wells.

  Args:
    layout: must be set to 'l1' for the standard layout or 'l2' for quadrant
            swap layout.
  Returns:
    List of the 96 well plate source wells for each 384 well plate. The order
    of the list corresponds to 384_map.
  Raises:
    Exception: if layout argument is not set to l1 or l2.
  """
  nc_row = ['no_cells'] * 24
  row_end = ['no_cells', 'no_cells']
  source_map = []
  if layout == 'l1':
    in_use_rows_96 = ['A', 'B', 'C', 'D', 'E', 'F']
    in_use_cols_96 = list(map(str, list(range(1, 11))))
    in_use_cols_96 = [x.zfill(2) for x in in_use_cols_96]

  elif layout == 'l2':
    in_use_rows_96 = ['D', 'E', 'F', 'A', 'B', 'C']
    in_use_cols_96 = (
        list(map(str, list(range(6, 11)))) +
        list(map(str, list(range(1, 6)))))
    in_use_cols_96 = [x.zfill(2) for x in  in_use_cols_96]
  else:
    raise Exception(
        f'{layout} is not a valid layout entry. Layout must be l1 or l2')

  for row in in_use_rows_96:
    rpf = [row] * len(in_use_cols_96)
    r_inds = [a + b for (a, b) in zip(rpf, in_use_cols_96)]
    dup_r_inds = list(np.repeat(r_inds, 2))
    full_row = row_end + dup_r_inds + row_end
    row_doubles = 2 * full_row
    source_map.append(row_doubles)
  all_source = []
  for sublist in source_map:
    for item in sublist:
      all_source.append(item)
  source_wells = nc_row + nc_row + all_source + nc_row + nc_row

  return source_wells


def make_batch(cl_df, qc=True, qc_categories=None, quadrant_swap=False,
               optimized_seeding=False, seed_categories=None):
  """Master function generates the batch design.

  Generates the layout for all six 384 well plates and also returns the design
  of the 96 well seed plate.
  Args:
    cl_df: A pandas dataframe containing metadata for the cell lines to be used
           in the design. Defaulted to none, in which case a generated
           representative dataframe is used.
    qc: The is a boolean flag that when set to True performs runs the quality
        control function on cl_df. If set to True, must provide a list of
        columns to be analyzed through the qc_categories Arg.
    qc_categories: List of columns in cl_df to analyzed by the quality control
                   function.
    quadrant_swap: Boolean flag that when set to True generates the quadrant
                   swap layout for 384 plates 4, 5 & 6. Defaulted to False.
    optimized_seeding: Boolean flag that when set to True performs optimized
                       seeding to reduce variance between rows, columns and
                       wells for a set of user selected categories. If set to
                       False, cell lines are randomly seeded into the 96 well
                       source plate.
    seed_categories: List of columns in the cl_df that will be used for
                     optimized seeding.
  Returns:
    batch: Pandas Dataframe detailing the design for the entire batch
    source_wp_96: Pandas Dataframe detailing the layout of the source 96 well
                  plate.
  """

  if qc:
    quality_control_cell_line_data(cl_df, qc_categories)

  source_wp_96 = seed_96_well_plate(
      cl_df, optimized_seeding, seed_categories, num_iterations=1000)

  if not quadrant_swap:
    plates = []
    plate_layout_384 = make_384_wp(source_wp_96, 'l1')
    for i in range(1, 7):
      pl_num = 'plate' + str(i)
      temp_plate = copy.deepcopy(plate_layout_384)
      temp_plate['plate'] = [pl_num] * 384
      plates.append(temp_plate)
    batch = pd.concat(plates, ignore_index=True)

  elif quadrant_swap:
    plates = []
    plate_layout_384 = make_384_wp(source_wp_96, 'l1')
    for i in range(1, 4):
      pl_num = 'plate' + str(i)
      temp_plate = copy.deepcopy(plate_layout_384)
      temp_plate['plate'] = [pl_num] * 384
      plates.append(temp_plate)
    plate_layout_384 = make_384_wp(source_wp_96, 'l2')
    for i in range(4, 7):
      pl_num = 'plate' + str(i)
      temp_plate = copy.deepcopy(plate_layout_384)
      temp_plate['plate'] = [pl_num] * 384
      plates.append(temp_plate)
    batch = pd.concat(plates, ignore_index=True)

  return batch, source_wp_96
