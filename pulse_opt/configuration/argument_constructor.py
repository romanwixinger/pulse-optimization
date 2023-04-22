""" Constructs a list of argument based on static and variable arguments provided as lookups.
"""

import itertools


def construct_args(static_args: dict, variable_args: dict) -> list:
    """ Expands two argument lookups into a list of function arguments.

    Args:
        static_args (dict): Arguments with a single option that should be included in each list item.
        variable_args (dict): Arguments with multiple options, each option getting their own list item.

    Example:
        from main.utilities.argument_constructor import construct_args

        static_args = {"a": 1.0}
        variable_args = {"scale": [0.1, 0.2], "n": [1, 2, 3]}

        # Will output [{"a": 1.0, "scale": 0.1, n: 1}, ... , {"a": 1.0, "scale": 0.2, n: 3}]
        args = construct_args(static_args, variable_args)
        print(args)

    Returns:
        List of all combination of variable arguments together with the static arguments in a lookup.
    """

    # Extract arguments and their possible options
    arguments = variable_args.keys()
    options = variable_args.values()

    # Get all combinations
    arg_combinations = list(itertools.product(*options))

    # Create dicts which contain the static args and the options
    args = []
    for combination in arg_combinations:
        arg = {**static_args}
        for i, key in enumerate(arguments):
            arg[key] = combination[i]
        args.append(arg)

    return args
