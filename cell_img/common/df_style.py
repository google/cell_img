"""Helper functions to style DataFrames.

Brief tutorial:
If you have a pd.DataFrame df and invoke "style = df.style" you obtain a generic
style object, which can be rendered to an html string with style.render()
or viewed in a cell with the usual cell value display.

This code provides a set_style function which takes a style object and
returns a new one with various default css properties reset. You'll usually
want to pass this through either the log_colorizer_core or the
discrete_colorizer_core functions.

To make this a bit more succint there are also the discrete_colorize and
log_colorize helper functions, which do both steps, so they accept style
arguments as well.

Examples:

1) Discrete-valued DataFrame with plain layout.
  df_cat = pd.DataFrame(np.resize(np.array(['A', 'B', 'C']), (3, 5)))
  df_cat.pipe(df_style.discrete_colorize)
  Remark: the former is short for:
    df_style.discrete_colorize_core(df_style.set_style(df.style))

2) Numerical DataFrame with logarithmic colors and plain layout.
  df_num = pd.DataFrame(np.resize(np.array([1., 10., 1000.]), (3, 5)))
  df_num.pipe(df_style.log_colorize)

3) Similar to 2, but with the "mini_zoom" feature that lays out a large
  DataFrame with small cells that "zoom" when you hover over them.
  df_big_num = pd.DataFrame(np.resize(np.array([1., 10., 1000.]), (30, 50)))
  df_big_num.pipe(df_style.log_colorize, mini_zoom=True,
    enforce_size=True,
    td_width=4, td_height=4,
    th_mini_font_size=3)

4) For internal borders in the dataframe which help align data to higher-levels
   in a multi-index:
   df.style.apply(df_style.border_data, axis=None)
"""

import matplotlib.colors
import matplotlib.markers
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns


_HTML_BORDERS = ['solid', 'dashed', 'dotted']
_BORDER_STYLE = 'border-style:'
_NO_BORDER = 'none'


def linear_colorize(df, value_range=None, cmap=None, **kwargs):
  """Apply linear color-scheme. kwargs are passed into set_style."""
  return colorize_core(set_style(df.style, **kwargs),
                       value_range=value_range,
                       cmap=cmap,
                       scale='linear')


def log_colorize(df, value_range=None, cmap=None, log_floor=1E-10, **kwargs):
  """Apply log-scaled color-scheme. kwargs are passed into set_style."""
  return colorize_core(set_style(df.style, **kwargs),
                       value_range=value_range,
                       cmap=cmap,
                       scale='log',
                       log_floor=log_floor)


def discrete_colorize(df, **kwargs):
  """Apply discrete color-scheme. kwargs are passed into set_style."""
  return discrete_colorize_core(
      set_style(df.style, **kwargs))


def set_style(style,
              text_color='white',
              highlight_hover=True,
              mini_zoom=False,
              th_mini_font_size=6,
              enforce_size=False,
              td_width=15,
              td_height=15):
  """Imposes some styling options on a DataFrame.style object."""
  properties = [
      {
          'selector': 'td',
          'props': [
              ('color', text_color),
          ]
      },
      {
          'selector':
              'table',
          'props': [
              ('table-layout', 'fixed'),
              ('width', '100%'),
              ('height', '100%'),
          ]
      },
      {
          'selector': '',
          'props': [('border-collapse', 'collapse')]
      },
      {
          'selector': 'table, tr, th, td',
          'props': [('border', '1px solid black')]
      },
  ]
  if enforce_size:
    properties.append({
        'selector':
            'tr, th, td',
        'props': [('overflow', 'hidden'), ('text-overflow',
                                           'clip'), ('width',
                                                     '{}px'.format(td_width)),
                  ('height',
                   '{}px'.format(td_height)), ('max-width',
                                               '{}px'.format(td_width)),
                  ('max-height', '{}px'.format(td_height)), ('white-space',
                                                             'nowrap')]
    })
  if highlight_hover:
    properties.append({
        'selector': 'tr:hover',
        'props': [('background-color', 'yellow')]
    })
  if mini_zoom:
    properties.extend([{
        'selector': 'th',
        'props': [('font-size', '{}pt'.format(th_mini_font_size))]
    }, {
        'selector': 'td',
        'props': [('padding', '0em 0em'),
                  ('font-size', '1px')]
    }, {
        'selector': 'td:hover',
        'props': [
            ('max-width', '200px'),
            ('font-size', '12pt'),
            ('padding', '10px')]
    }, {
        'selector': 'th:hover',
        'props': [
            ('max-width', '200px'),
            ('font-size', '12pt'),
            ('padding', '10px')]
    }, {
        'selector': 'tr:hover',
        'props': [
            ('max-width', '200px'),
            ('font-size', '12pt'),
            ('padding', '10px')]
    }])
  return style.set_table_styles(properties).set_table_attributes('border=1')


def colorize_core(style,
                  value_range=None,
                  cmap=None,
                  scale='log',
                  log_floor=1E-10):
  """Helper function for applying color-scheme."""
  if cmap is None:
    cmap = 'viridis'
  cmap = plt.get_cmap(cmap)
  if value_range is None:
    df = style.data
    if scale == 'linear':
      value_range = (df.min().min(), df.max().max())
    elif scale == 'log':
      value_range = (max(log_floor, df.min().min()), df.max().max())
  # pylint: disable=unpacking-non-sequence
  low, high = value_range
  # pylint: enable=unpacking-non-sequence

  def make_color(s):
    clipped = np.clip(s, low, high)
    if scale == 'linear':
      normed = (clipped - low) / (high - low)
    elif scale == 'log':
      normed = (np.log10(clipped) - np.log10(low)) / (
          np.log10(high) - np.log10(low))
    c = matplotlib.colors.rgb2hex(cmap(normed))
    if np.isnan(s):
      return 'background-color: #CCCCCC'
    else:
      return 'background-color: %s' % c

  return style.applymap(make_color)


def discrete_colorize_core(style):
  """Apply a categorical color-scheme."""
  colorizer = make_colorizer_from_series(pd.Series(style.data.values.flatten()))
  return style.applymap(colorizer)


def get_marker_and_color_dicts(series, palette='Dark2'):
  counts = series.value_counts(dropna=False)
  num_items = len(counts)
  markers = matplotlib.markers.MarkerStyle().filled_markers[:num_items]
  colors = sns.color_palette(palette, num_items)
  keys = list(counts.index)
  marker_dict = dict(zip(keys, markers))
  color_dict = dict(zip(keys, colors))
  return marker_dict, color_dict


def make_colorizer_from_series(series, palette='Dark2'):
  """Make colorizing mapfunction for pd.style.applymap."""
  _, color_dict = get_marker_and_color_dicts(series, palette=palette)

  def colorizer(s):

    try:
      color = color_dict[s]
    except KeyError:
      # Set to default color if not defined
      color = '#cccccc'

    c = matplotlib.colors.rgb2hex(color)
    return 'background-color: %s' % c

  return colorizer


def _border_columns(df):
  """Returns a style df with css border information."""

  style_df = pd.DataFrame(index=df.index, columns=df.columns, data=_NO_BORDER)

  cols = df.columns
  col_names = cols.to_frame().columns
  for level, border in reversed(list(zip(col_names[:-1], _HTML_BORDERS))):
    values = cols.get_level_values(level)
    value_changes = list(values[1:] != values[:-1]) + [False]
    style_df.loc[:, value_changes] = border

  return style_df


def border_data(df):
  """Aligns styled borders in df data with multi-index columns.

  Border styles are defined in _HTML_BORDERS and are applied to data aligned
  with the most-significant indices. e.g. levels (0, 1, 2).  The
  least-significant index is not bordered as the existing display does a good
  enough job at visually separating these values.

  Usage: my_dataframe.style.apply(df_style.border_data, axis=None)

  Args:
    df: Multi-indexed DataFrame to be styled.

  Returns:
    DataFrame with css border-style strings for html formatting.
  """

  bs = pd.DataFrame(index=df.index, columns=df.columns, data=_BORDER_STYLE)
  no_border = pd.DataFrame(index=df.index, columns=df.columns, data=_NO_BORDER)

  top = no_border
  right = _border_columns(df)
  down = _border_columns(df.T).T
  left = no_border

  return bs + ' ' + top + ' ' + right + ' ' + down + ' ' + left + ';'  # pytype: disable=unsupported-operands  # typed-pandas
