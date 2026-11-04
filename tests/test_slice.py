import polars as pl

from lpw.data.download import month_range
from lpw.data.slice import split_groups


def test_month_range_crosses_year():
    assert month_range("2021-11:2022-02") == ["2021-11", "2021-12", "2022-01", "2022-02"]


def test_split_keeps_groups_whole_and_balances_roles():
    rows = [(i, i // 3, int(i % 2 == 0)) for i in range(3000)]
    a = pl.DataFrame(rows, schema=["node_id", "group_id", "is_scam"], orient="row")
    s = split_groups(a)
    assert s.group_by("group_id").agg(pl.col("split").n_unique().alias("k"))["k"].max() == 1
    frac = s.group_by("split").len().with_columns(pl.col("len") / s.height)
    got = dict(frac.iter_rows())
    assert abs(got["train"] - 0.70) < 0.02 and abs(got["val"] - 0.15) < 0.02 and abs(got["test"] - 0.15) < 0.02
