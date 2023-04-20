import pytest
import os
import pandas as pd
import json


from pulse_opt.integrals.utilities import (
    flatten_dict,
    create_table,
    add_prefix,
    setup_logging,
    save_result_as_json,
    save_table_as_csv,
    save_table_as_pickle,
    construct_filename,
    run_with_multiprocessing,
    run_without_multiprocessing,
)


def test_flatten_dict_none():
    assert flatten_dict(None) == {}, f"Expected None input to result in trivial dict, but got {flatten_dict(None)} instead."


def test_flatten_dict_trivial():
    assert flatten_dict({"key": "value"}) == {"key": "value"}, \
        f"Expected flat dict to be not modified at all, but found otherwise."


def test_flatten_dict_with_hierarchy():
    dict_with_hierarchy = {"top1": {"child1": "value11", "child2": "value12"}, "top2": "value2"}
    expected = {"top1.child1": "value11", "top1.child2": "value12", "top2": "value2"}
    assert flatten_dict(dict_with_hierarchy) == expected, \
        f"Expected {expected} but found flatten_dict(input) from the input {dict_with_hierarchy} to flatten_dict()."


def test_create_table():
    pass


def test_add_prefix():
    assert add_prefix({"key": "value"}, "prefix.") == {"prefix.key": "value"}, "Expected different result."


# Todo: Fix the setup such that it can be tested.
@pytest.mark.skip("Unable to unit test because the function just has side effects.")
def test_setup_logging():
    pass


def test_save_result_as_json(tmp_path):
    pass


def test_save_result_as_json_with_float(tmp_path):
    pass


def test_save_result_as_json_and_load_again(tmp_path):
    pass


def test_save_table_as_csv(tmp_path):
    df = pd.DataFrame([{"column": "value"}])
    run = "test_run"
    save_table_as_csv(df=df, run=run, folder_path=tmp_path)
    filepath = tmp_path / "results.csv"
    assert os.path.isfile(filepath), f"Expected to find file at {filepath} but it was not the case."
    filepath.unlink()


def test_save_table_as_pickle(tmp_path):
    df = pd.DataFrame([{"column": "value"}])
    run = "test_run"
    save_table_as_pickle(df=df, run=run, folder_path=tmp_path)
    filepath = tmp_path / "results.pkl"
    assert os.path.isfile(filepath), f"Expected to find file at {filepath} but it was not the case."
    filepath.unlink()


def test_construct_filename():
    expected = 'default-loss_param1_1_param2_2.json'
    loss = 'default-loss'
    variable_args = ['param1', 'param2']
    loss_args = {'param1': 1, 'param2': 2}
    result = construct_filename(loss, variable_args, loss_args)
    assert result == expected, \
        f"Expected {expected} for loss={loss}, variable_args={variable_args}, loss_arg={loss_args} but found {result}."


def test_construct_filename_with_special_filetype():
    expected = 'default-loss_param1_1_param2_2.csv'
    loss = 'default-loss'
    variable_args = ['param1', 'param2']
    loss_args = {'param1': 1, 'param2': 2}
    result = construct_filename(loss, variable_args, loss_args, "csv")
    assert result == expected, \
        f"Expected {expected} for loss={loss}, variable_args={variable_args}, loss_arg={loss_args} but found {result}."


def test_construct_filename_no_args():
    assert construct_filename('default-loss', [], {}) == 'default-loss.json'


def test_run_with_multiprocessing():
    pass


def test_run_without_multiprocessing():
    pass
