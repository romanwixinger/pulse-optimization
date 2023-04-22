import pytest
import os
import numpy as np
import pandas as pd


from pulse_opt.integrals.utilities import (
    serialize_table,
    deserialize_table,
    save_result_as_json,
    save_table_as_csv,
    save_table_as_pickle,
    load_table_from_csv,
    load_table_from_pickle,
    construct_filename,
    run_with_multiprocessing,
    run_without_multiprocessing,
)

test_dataframes = [
    pd.DataFrame([{"column": "value"}]),
    pd.DataFrame([{"column": np.array([1])}]),
    pd.DataFrame([{"column": [1]}]),
    pd.DataFrame([{"column": np.array([1.0])}]),
    pd.DataFrame([{"column": [1.0]}]),
]
serialized_test_dataframes = [
    pd.DataFrame([{"column": "value"}]),
    pd.DataFrame([{"column": np.array2string(np.array([1]))}]),
    pd.DataFrame([{"column": [1]}]),
    pd.DataFrame([{"column": np.array2string(np.array([1.0]))}]),
    pd.DataFrame([{"column": [1.0]}]),
]


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


@pytest.mark.parametrize(
    "df,exp_df",
    [(df, ser_df) for df, ser_df in zip(test_dataframes, serialized_test_dataframes)]
)
def test_serialize_table(df: pd.DataFrame, exp_df: pd.DataFrame):
    serialized_df = serialize_table(df)
    assert compare_dataframes(serialized_df, exp_df), \
        f"Expected serialized df to be {exp_df} but found {serialized_df} for input {df}."


@pytest.mark.parametrize(
    "df,exp_df",
    [(ser_df, deserialized_df) for ser_df, deserialized_df in zip(serialized_test_dataframes, test_dataframes)]
)
def test_deserialize_table(df: pd.DataFrame, exp_df: pd.DataFrame):
    deserialized_df = deserialize_table(df)
    assert compare_dataframes(deserialized_df, exp_df), \
        f"Expected deserialized df to be {exp_df} but found {deserialized_df} for input {df}."


@pytest.mark.parametrize(
    "df,exp_df",
    [(pd.DataFrame([{"column": [1.0]}]), pd.DataFrame([{"column": [1.0]}])),
     (pd.DataFrame([{"column":  np.array2string(np.array([1.0]))}]), pd.DataFrame([{"column": np.array([1.0])}]))]
)
def test_deserialize_table_special(df: pd.DataFrame, exp_df: pd.DataFrame):
    deserialized_df = deserialize_table(df)
    assert compare_dataframes(deserialized_df, exp_df), \
        f"Expected deserialized df to be {exp_df} but found {deserialized_df} for input {df}."


@pytest.mark.parametrize("df", test_dataframes)
def test_save_table_as_csv(tmp_path, df):
    run = "test_run"
    save_table_as_csv(df=df, run=run, folder_path=tmp_path)
    filepath = tmp_path / "results.csv"
    assert os.path.isfile(filepath), f"Expected to find file at {filepath} but it was not the case."
    filepath.unlink()


@pytest.mark.parametrize("df", test_dataframes)
def test_save_table_as_pickle(tmp_path, df):
    run = "test_run"
    save_table_as_pickle(df=df, run=run, folder_path=tmp_path)
    filepath = tmp_path / "results.pkl"
    assert os.path.isfile(filepath), f"Expected to find file at {filepath} but it was not the case."
    filepath.unlink()

@pytest.mark.skip("At the moment, list are always deserialized to np.arrays. We need to fix this in the future.")
@pytest.mark.parametrize("df", test_dataframes)
def test_load_table_from_csv(tmp_path, df):
    run = "test_run"
    save_table_as_csv(df=df, run=run, folder_path=tmp_path)
    df_reloaded = load_table_from_csv(run=run, folder_path=tmp_path)
    assert compare_dataframes(df, df_reloaded), \
        f"Assumed original df to be the same as the reloaded one, but found {df} and {df_reloaded}."


@pytest.mark.parametrize("df", test_dataframes)
def test_load_table_from_pickle(tmp_path, df):
    run = "test_run"
    save_table_as_pickle(df=df, run=run, folder_path=tmp_path)
    df_reloaded = load_table_from_pickle(run=run, folder_path=tmp_path)
    assert compare_dataframes(df, df_reloaded), \
        f"Assumed original df to be the same as the reloaded one, but found {df} and {df_reloaded}."


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


def compare_dataframes(df1, df2, tol=1e-8):
    """Compares two pandas dataframes for near-equality of floats and numpy arrays and perfect equality of strings.
    """
    # Check if the two dataframes have the same shape
    if df1.shape != df2.shape:
        return False

    # Check if the column names are the same
    if not all(df1.columns == df2.columns):
        print("The dfs are not equal, they do not have the same columns.")
        return False

    # Iterate over each cell in the dataframes
    for i in range(df1.shape[0]):
        for j in range(df1.shape[1]):
            cell1 = df1.iloc[i,j]
            cell2 = df2.iloc[i,j]

            # Check for equality of strings
            if isinstance(cell1, str) and isinstance(cell2, str):
                if cell1 != cell2:
                    print("Dfs are not the same, some string values are not equal.")
                    return False

            # Check for near-equality of floats and numpy arrays
            elif isinstance(cell1, (float, np.floating)) and isinstance(cell2, (float, np.floating)):
                if abs(cell1 - cell2) > tol:
                    print("Dfs are not the same, some float values are not equal.")
                    return False

            # Check for near equality of np.arrays
            elif isinstance(cell1, np.ndarray) and isinstance(cell2, np.ndarray):
                if not np.allclose(cell1, cell2, atol=tol):
                    print("Dfs are not the same, some numpy array values are not equal.")
                    return False

            elif isinstance(cell1, list) and isinstance(cell2, list):
                if isinstance(cell1[0], str) and cell1 != cell2:
                    print("Dfs are not the same, some str values in a list are not equal.")
                    return False
                if not all(abs(a - b) < tol for a, b in zip(cell1, cell2)):
                    print("Dfs are not the same, some float list values are not equal.")
                    return False

            # Otherwise, the cell types do not match, and the dataframes are not equal
            else:
                print(f"Dfs contain unexpected types: {type(cell1)} and {type(cell2)}.")
                return False

    # If all cells pass the tests, the dataframes are equal
    return True
