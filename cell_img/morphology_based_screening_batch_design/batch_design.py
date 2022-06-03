"""Functions that generate morphological screening batch designs.

This Library contains a set of functions that can be used to generate
experimental designs for well plate based expeiments that are primarily used in
cellular morphology screening. They are particularly applicable when cells are
analyzed by High Throughput Screening (HTS) and ML based techniques.

ML based methods tend to be very sensitive and can easily pick up on non
relevant signals in cell images such as the row or column a cell was in, thus it
is imperitive to ensure that the signal of interest is not conflated with any
of these known nuscience signals. The functions in this library try to design
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
   caused by plate location. It should be noted that only wells with rows A to
   F and columns 1 to 10 are used.
2. Cells are grown and then transferred to a batch of 384 well plates. A batch
   for our purposes denotes 6 x 384 well plates. It should be noted that only
   wells within rows C to M and columns 3 to 22 are used. The cells are
   transferred to the 384 well plates in a unique pattern: cells in 96-A1 are
   transferred to wells 384-C3, 384-C4, 384-D3 & 384; cells in 96-A2 are
   transferred to 384-C5, 384-C6, 384-D5 & 384-D6. This pattern is repeated
   for the entire plate with finally 96-F10 being transferred to 384-M21,
   384-M22, 384-N21 & 384-N22. One 96 well plate has sufficient cells to
   populate six 384 well plates in this manner.
3. Unless a smaller number of lines is being used, such that they can replicated
   in the source 96 well plate we strongly recommend using the quadrant_swap
   option available in the functions that ensures no cell line is associated
   with a single well. 384 well plates 1, 2 & 3 are seeded as described above
   but plates 4, 5, & 6 have quadrants of the source plate diagnolly swapped
   resulting in an altered layout. Practically this means that cells in rows
   A, B, C columns 1 to 6 of the 96 well plate are placed in rows I to N
   columns 13 to 22 in the 384 well plates. Similarly cells in rows D, E F
   columns 6 to 10 of the 96 well plate are placed in rows A to H columns
   3 to 12 of the 384 well plate. The other two quadrants of the 96 well plate
   are swapped in a similar fashion.
4. With the six 384 well plates filled they can be used for morphology
   screening.

There are six functions in the library that can be called independently but
make_batch() is the master function that generates the batch design.
"""


import copy
import random

from IPython import display
import numpy as np
import pandas as pd


def big_display(df, n):
  """Displays, in colab, of the first n rows of a dataframe.

  Args:
    df: The dataframe you want to display
    n: number of rows you want to display
  """
  with pd.option_context('display.max.rows', n):
    display.display(df)


def make_ns_well_plate(quadrant_swap=False):
  """Generates an "empty" 96 well plate dataframe.

  Columns include well, well_col, well_row, plate region, pair region &
  disease status of each well.

  Args:
    quadrant_swap: defaulted to false but if set to true will generate the
                   quadrant swap layout.
  Returns:
      df: A dataframe with the above described columns.
  """

  df = pd.DataFrame(data=None, columns=[])

  def _ns_plate_map():
    """Generates a list of wells in a ninety six well plate.

    Returns:
      ns_map: an ordered list containing the wells in a ninety six well plate.
    """
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols = list(map(str, list(range(1, 13))))
    cols = [x.zfill(2) for x in cols]
    plate_map = []
    for row in rows:
      rpf = [row] * len(cols)
      r_inds = [a + b for (a, b) in zip(rpf, cols)]
      plate_map.append(r_inds)
    ns_map = []
    for sublist in plate_map:
      for item in sublist:
        ns_map.append(item)
    return ns_map

  def _ns_plate_regions(quadrant_swap=False):
    """Denotes wells in use by assigning a 'region' number.

    Generates an ordered list, which corresponds positionally to the ordered
    well list generated by _ns_plate_map(). It denotes which wells are in
    use by assigning a region number. Wells not in use are denoted by the label
    'unset'.
    Args:
      quadrant_swap: defaulted to false but if set to true will generate the
                     quadrant swap layout.
    Returns:
      region + unused: An ordered list detailing the region assignments for each
      well.
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

  def _ns_well_disease_status(quadrant_swap=False):
    """Denotes the disease status of each well.

    Generates an ordered list, which corresponds positionally to the ordered
    well list generated by _ns_plate_map(). The list denotes the status of the
    cells that can go in a particular well. This constraint is enacted to ensure
    disease state is not conflated with well, column or row. Wells that are not
    used have their disease status set to 'unset'.
    Args:
      quadrant_swap: defaulted to false but if set to true will generate the
                      quadrant swap layout.
    Returns:
      assignments + unused: An ordered list detailing disease status for each
      well.
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

  def _ns_pair_regions(quadrant_swap=False):
    """Denotes what pair region a well belongs too.

    Generates an ordered list, which corresponds positionally to the ordered
    well list generated by _ns_plate_map(). This sets the pairs of wells that
    demographically matched healthy and disease can be assigned too.
    Args:
       quadrant_swap: defaulted to false but if set to true will generate the
                       quadrant swap layout.
    Returns:
      pair_regions + unset: An ordered list setting the pair region id for
      each well.
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

  if not quadrant_swap:

    df['well'] = _ns_plate_map()
    df['well_row'] = df.well.str[0]
    df['well_col'] = df.well.str[1:]
    df['cl_region'] = _ns_plate_regions()
    df['pair_region'] = _ns_pair_regions()
    df['disease_state'] = _ns_well_disease_status()

  elif quadrant_swap:
    df['well'] = _ns_plate_map()
    df['well_row'] = df.well.str[0]
    df['well_col'] = df.well.str[1:]
    df['cl_region'] = _ns_plate_regions(quadrant_swap=True)
    df['pair_region'] = _ns_pair_regions(quadrant_swap=True)
    df['disease_state'] = _ns_well_disease_status(quadrant_swap=True)

  else:
    print('quadrant_swap needs to be set to a boolean value')

  return df


def sample_cell_line_frame():
  """Creates an example cell line dataframe.

  This example frame contains columns typical of cell line metadata and can be
  used to run the code, in the abscence of real data, for testing or for demo
  purposes.
  Returns:
    df: The example dataframe
  """
  cell_ids = [str(x) for x in list(np.arange(1, 61))]

  pair_ids = list(np.repeat([str(x) for x in list(np.arange(0, 30))], 2))

  disease_state = ['healthy', 'disease'] * 30

  ages = [random.randrange(50, 75, 1) for i in range(60)]
  ages = [str(y) for y in ages]
  ages[5] = ' '
  ages[32] = ' '

  doubling_time = [(random.randrange(200, 400, 1) / 100) for dt in range(60)]

  sex = ((['M'] * 20) + (['F'] * 10))
  random.shuffle(sex)
  sex = list(np.repeat(sex, 2))

  missing_num_col = [(random.randrange(200, 400, 1) / 100) for ptc in range(60)]
  missing_num_col = [str(z) for z in missing_num_col]
  for ind in [4, 10, 22, 45, 59]:
    missing_num_col[ind] = ' '
  perfect_str_col = [random.randrange(50, 75, 1) for i in range(60)]
  perfect_str_col = [str(z) for z in perfect_str_col]
  df = pd.DataFrame(
      list(
          zip(cell_ids, pair_ids, disease_state, ages, doubling_time, sex,
              missing_num_col, perfect_str_col)),
      columns=[
          'cell_id', 'pair_id', 'disease_state', 'age', 'doubling_time', 'sex',
          'missing_num_col', 'perfect_str_col'])

  return df


def quality_control_cell_line_data(df, important_categories):
  """Tests specified columns are correctly formated for use in seeding.

  In preparation for seeding the well plate, this function checks the quality of
  data in the user selected columns of the input dataframe that will be used as
  part of the seeding process. The function checks for only a couple of commonly
  encountered issues. It will initially check to ensure that the data is
  numeric, if not it will check to see if data is in string format. If in string
  format and complete it will convert to a float. If 3 or fewer entries are
  missing it will fill these with the sample mean. If there are more than 3
  entries missing, the user is warned to review the data. If the data is a non-
  numeric type, the user is warned and asked to convert to a numeric type.
  Args:
    df: input dataframe to be checked
    important_categories: list of dataframe column names to be tested.
  Returns:
    df: ajusted dataframe
  """

  for cat in important_categories:
    if all([isinstance(x, (int, float)) for x in df[cat].values]):
      print(f'Dataframe column {cat} is good')

    elif all([y.replace('.', '').isnumeric() for y in df[cat].values]):
      print('data currently in string form')
      print('trying to convert...')
      try:
        df[cat] = df[cat].astype(float)
        print('Conversion successful')
      except ValueError:
        print(f'Dataframe column: {cat} cannot be used in the current form.')
        print('Please review data and convert to a numeric form')
        break

    elif sum([y.replace('.', '').isnumeric() for y in df[cat].values
             ]) >= (len(df[cat].values) - 3):
      # The 3 is effectively the number of missing values we can relaistically
      # tolerate.
      print(f'empty values detected in {cat}, filling with mean of data')
      missing_inidicies = [
          i for i, j in enumerate(df[cat].values)
          if not j.replace('.', '').isnumeric()
      ]
      print(missing_inidicies)
      missing_inidicies.sort(reverse=True)
      col_vals_str = [
          num for num in df[cat].values if num.replace('.', '').isnumeric()
      ]
      col_vals = [float(val) for val in col_vals_str]
      for ind in missing_inidicies:
        col_vals.insert(ind, np.mean(col_vals))
      df[cat] = col_vals

    elif (sum([y.replace('.', '').isnumeric() for y in df[cat].values]) <
          (len(df[cat].values) - 3)) and (sum(
              [y.replace('.', '').isnumeric() for y in df[cat].values]) > 0):
      print(sum([y.replace('.', '').isnumeric() for y in df[cat].values]))
      print('''Data has too many missing entries to use for data based seeding
             please review metadata and correct missing values''')

    else:
      print('''Data may not be suitable for use in metadata based seeding.
               Please review data and covert to numeric form if you would still
               like to use it''')

  return df


def seed_ns_well_plate(cell_line_df,
                       optimized_seeding=False,
                       seed_categories=None,
                       num_iterations=None):
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
    ns_wp: dataframe describing the ninety six well plate and location of
           cell lines within that plate.
  """

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
      cell_line_df: the input dataframe with the extra pair_region column.



    """
    cell_line_df.sort_values(by=['pair_id', 'disease_state'])
    r_assignments = np.arange(1, 31)
    random.shuffle(r_assignments)
    cell_line_df['pair_region'] = np.repeat([str(x).zfill(2) for
                                             x in r_assignments], 2)

    return cell_line_df

  def _measure_variation(seed_categories, df):
    """Measures the cv in user specified catgories.

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
      try:
        df4s = copy.deepcopy(df)
        df4s = df4s[df4s[cat].notna()]
        col_cv = ((df4s.groupby('well_col')[cat].mean().std()) /
                  (df4s.groupby('well_col')[cat].mean().mean()))
        row_cv = ((df4s.groupby('well_row')[cat].mean().std()) /
                  (df4s.groupby('well_row')[cat].mean().mean()))
        w_cv = ((df4s.groupby('well')[cat].mean().std()) /
                (df4s.groupby('well')[cat].mean().mean()))
        measured_var_in_cats += (col_cv + row_cv + w_cv)
      except ValueError:
        print(f'{cat} may not be numeric.')
        print('Consider converting to a numneric representation')

    return measured_var_in_cats

  ns_wp = make_ns_well_plate(quadrant_swap=False)
  ns_wp_qs = make_ns_well_plate(quadrant_swap=True)

  # Merge with cell line df to assign lines to regions
  cl_df = _add_region_assignment(cell_line_df)
  ns_wp = ns_wp.merge(cl_df, on=['pair_region', 'disease_state'], how='left')
  ns_wp_qs = ns_wp_qs.merge(cl_df, on=['pair_region', 'disease_state'],
                            how='left')

  if optimized_seeding:
    df_variance = []
    dataframes = []
    for _ in range(num_iterations):
      batch_design_df = pd.concat([ns_wp, ns_wp_qs], ignore_index=True)
      var = _measure_variation(seed_categories, batch_design_df)
      df_variance.append(var)
      dataframes.append(ns_wp)
      cl_df = _add_region_assignment(cell_line_df)
      ns_wp = make_ns_well_plate(quadrant_swap=False)
      ns_wp = ns_wp.merge(cl_df, on=['pair_region', 'disease_state'],
                          how='left')
      ns_wp_qs = make_ns_well_plate(quadrant_swap=True)
      ns_wp_qs = ns_wp_qs.merge(
          cl_df, on=['pair_region', 'disease_state'], how='left')

    lowest_cv = min(df_variance)
    lowest_index = df_variance.index(lowest_cv)
    print('min variance is: ', lowest_cv)
    ns_wp = dataframes[lowest_index]

  return ns_wp


def make_384_wp(ns_wp, layout):
  """Create 384 well plate layout from ninety six well plate design.

  Takes the ninety six well df and generates a dataframe representing a 384
  well plate in accordance with the operation of the robot used to transfer
  cells from the ninety six well plate to the three eighty four well plates.
  Args:
    ns_wp: df representing the ninety six well plate
    layout: must be set to 'l1' for the standard layout or 'l2' for quadrant
            swap layout.
  Returns:
    tef_df: datafram representing the layout of the three eighty four well
            plate.
  """
  def _tef_layout():
    """Makes the list of wells in the 384 well plate.

    Returns:
      tef_map: list of well names in a 384 well plate in order
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

    tef_map = []
    for sublist in plate_map:
      for item in sublist:
        tef_map.append(item)

    return tef_map

  def _tef_to_ns_conv_col(layout):
    """Defines a list that identifies the 96wp source well for 384wp wells.

    Args:
      layout: must be set to 'l1' for the standard layout or 'l2' for quadrant
              swap layout.
    Returns:
      source_wells: list of the 96 well plate source wells for each 384 well
                    plate. The order of the list corresponds to tef_map.
    """
    nc_row = ['no_cells'] * 24
    row_end = ['no_cells', 'no_cells']
    source_map = []
    if layout == 'l1':
      ns_in_use_rows = ['A', 'B', 'C', 'D', 'E', 'F']
      ns_in_use_cols = list(map(str, list(range(1, 11))))
      ns_in_use_cols = [x.zfill(2) for x in ns_in_use_cols]

    elif layout == 'l2':
      ns_in_use_rows = ['D', 'E', 'F', 'A', 'B', 'C']
      ns_in_use_cols = (
          list(map(str, list(range(6, 11)))) +
          list(map(str, list(range(1, 6)))))
      ns_in_use_cols = [x.zfill(2) for x in ns_in_use_cols]
    else:
      print(f'{layout} is not a valid layout entry. Layout must be l1 or l2')

    for row in ns_in_use_rows:
      rpf = [row] * len(ns_in_use_cols)
      r_inds = [a + b for (a, b) in zip(rpf, ns_in_use_cols)]
      dup_r_inds = list(np.repeat(r_inds, 2))
      full_row = row_end + dup_r_inds + row_end
      row_doubles = 2 * full_row
      source_map.append(row_doubles)
    all_source = []
    for sublist in all_source:
      for item in sublist:
        all_source.append(item)
    source_wells = nc_row + nc_row + all_source + nc_row + nc_row

    return source_wells

  # Create the 384 wp well layout
  tef_layout = _tef_layout()
  ns_source = _tef_to_ns_conv_col(layout)
  tef_df = pd.DataFrame(
      list(zip(tef_layout, ns_source)), columns=['well', 'ns_source'])
  tef_df['well_row'] = tef_df.well.str[0]
  tef_df['well_col'] = tef_df.well.str[1:]

  # Clean up 96wp dataframe and remove columns not needed
  ns_source = ns_wp.drop(['well_row', 'well_col'], axis=1)
  ns_source = ns_source.rename(columns={'well': 'ns_source'})

  # Merge the dfs to effectively 'seed' the 384 well plate
  tef_df = tef_df.merge(ns_source, on=['ns_source'], how='left')
  tef_df = tef_df.fillna('no_cells')

  return tef_df


def make_batch(cl_df=None, qc=False, qc_categories=None, quadrant_swap=False,
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
        columns to be analyzed through the qc_categories Arg. Defaulted to
        False.
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
    ns_source_wp: Pandas Dataframe detailing the layout of the source 96 well
                  plate.
  """
  if cl_df is None:
    cl_df = sample_cell_line_frame()

  if qc:
    try:
      cl_df = quality_control_cell_line_data(cl_df, qc_categories)
    except ValueError:
      print('''please ensure qc categories have been entered if dataframe
            qc is desired''')

  ns_source_wp = seed_ns_well_plate(
      cl_df, optimized_seeding, seed_categories, num_iterations=1000)

  if not quadrant_swap:
    plates = []
    tef_plate = make_384_wp(ns_source_wp, 'l1')
    for i in range(1, 7):
      pl_num = 'plate' + str(i)
      temp_plate = copy.deepcopy(tef_plate)
      temp_plate['plate'] = [pl_num] * 384
      plates.append(temp_plate)
    batch = pd.concat(plates, ignore_index=True)

  elif quadrant_swap:
    plates = []
    tef_plate = make_384_wp(ns_source_wp, 'l1')
    for i in range(1, 4):
      pl_num = 'plate' + str(i)
      temp_plate = copy.deepcopy(tef_plate)
      temp_plate['plate'] = [pl_num] * 384
      plates.append(temp_plate)
    tef_plate = make_384_wp(ns_source_wp, 'l2')
    for i in range(4, 7):
      pl_num = 'plate' + str(i)
      temp_plate = copy.deepcopy(tef_plate)
      temp_plate['plate'] = [pl_num] * 384
      plates.append(temp_plate)
    batch = pd.concat(plates, ignore_index=True)

  return batch, ns_source_wp





