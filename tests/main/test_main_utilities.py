import pytest
from main.utilities.argument_constructor import construct_args


def test_construct_args_single_variable_arg():
    static_args = {"a": 1.0}
    variable_args = {"scale": [0.1]}
    expected_args = [{"a": 1.0, "scale": 0.1}]
    assert construct_args(static_args, variable_args) == expected_args, "Failed single variable args test."


def test_construct_args_no_variable_args():
    static_args = {"a": 1.0}
    variable_args = {}
    expected_args = [{"a": 1.0}]
    result_args = construct_args(static_args, variable_args)
    assert result_args == expected_args, \
        f"Failed no variable args test. Expected args {expected_args} but found {result_args} for static_args {static_args} and variable_args {variable_args} as input."

def test_construct_args_no_static_args():
    static_args = {}
    variable_args = {"a": [1.0, 2.0]}
    expected_args = [{"a": 1.0}, {"a": 2.0}]
    result_args = construct_args(static_args, variable_args)
    assert result_args == expected_args, \
        f"Failed no static args test. Expected args {expected_args} but found {result_args} for static_args {static_args} and variable_args {variable_args} as input."


def test_construct_args_mixed():
    static_args = {"a": 1.0}
    variable_args = {"scale": [0.1], "n": [1, 2]}
    expected_args = [{"a": 1.0, "scale": 0.1, "n": 1}, {"a": 1.0, "scale": 0.1, "n": 2}]
    result_args = construct_args(static_args, variable_args)
    assert result_args == expected_args, \
        f"Failed mixed args test. Expected args {expected_args} but found {result_args} for static_args {static_args} and variable_args {variable_args} as input."


def test_construct_args_empty_args():
    static_args = {}
    variable_args = {}
    expected_args = [{}]
    assert construct_args(static_args, variable_args) == expected_args, "Failed empty args test."


def test_construct_args_variable_arg_not_list():
    static_args = {"a": 1.0}
    variable_args = {"scale": 0.1}

    with pytest.raises(TypeError):
        construct_args(static_args, variable_args), \
        "Failed to detect wrong format in variable args (float instead of list)."
