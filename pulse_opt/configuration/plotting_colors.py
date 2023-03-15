"""Defines a color palette to be used in plots.

When we display single qubit gates, then the 2x2 matrices have 4 real and 4 imaginary parts, so in total 8.
Similarly, in the two qubit case we have 4x4 matrices and so 4 x 4 x 2 = 32 parts in total.

Attributes:
    all_colors (dict): Lookup of all colors supported by Matplotlib with name (str) as key.
    color_choice_8 (list[str]): Selection of 8 color names to be used in the single qubit case.
    color_choice_32 (list[str]): Selection of 32 color names to be used in the two qubit case.
    selected_colors_8 (dict): Lookup all_colors snipped down to 8 key-value pairs.
    selected_colors_32 (dict): Lookup all_colors snipped down to 32 key-value pairs.
"""


import matplotlib.colors as colors


all_colors = colors.get_named_colors_mapping()
color_choice_32 = [
    "deepskyblue",
    "royalblue",
    "darkcyan",
    "cyan",
    "lemonchiffon",
    "gold",
    "orange",
    "goldenrod",
    "lightcoral",
    "indianred",
    "darkred",
    "maroon",
    "peachpuff",
    "sandybrown",
    "chocolate",
    "saddlebrown",
    "honeydew",
    "palegreen",
    "yellowgreen",
    "olivedrab",
    "thistle",
    "plum",
    "violet",
    "purple",
    "pink",
    "palevioletred",
    "mediumvioletred",
    "darkmagenta",
    "lightgrey",
    "darkgrey",
    "grey",
    "gray",
    "dimgrey",
    "dimgray",
    "black",
    "turquoise",
    "aquamarine",
    "mediumaquamarine",
    "lightseagreen",
    "mediumblue",
    "darkblue",
    "midnightblue",
    "navy",
]
selected_colors_32 = {key: all_colors[key] for key in color_choice_32}

color_choice_8 = [
    "yellow",
    "rebeccapurple",
    "tomato",
    "black",
    "dimgrey",
    "royalblue",
    "limegreen",
    "orchid",
]
selected_colors_8 = {key: all_colors[key] for key in color_choice_8}
