import numpy as np
import pandas as pd

from recsys.rankers import group as group_utils


def test_build_groups_basic():
    pairs = pd.read_csv("recsys/tests/fixtures/tiny_pairs_train.csv")
    groups = group_utils.build_groups(pairs, "query_id")
    assert groups == [2, 1]
    group_utils.assert_group_sums(groups, len(pairs))
