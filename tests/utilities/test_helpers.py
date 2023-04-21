import pytest

from pulse_opt.utilities.helpers import (
    load_function_or_class,
    flatten_dict,
    add_prefix,
    CustomEncoder,
    create_folder,
)


def test_load_function_or_class():
    pass


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


def test_add_prefix():
    assert add_prefix({"key": "value"}, "prefix.") == {"prefix.key": "value"}, "Expected different result."


def test_custom_encoder(tmp_path):
    pass


def test_create_folder(tmp_path):
    pass
