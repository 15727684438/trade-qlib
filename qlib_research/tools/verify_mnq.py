# -*- coding: utf-8 -*-
"""Verify the ingested MNQ multi-frequency qlib bins are readable and sane."""
import qlib
from qlib.constant import REG_CN

qlib.init(provider_uri=r"D:\qlib_research\data\mnq", region=REG_CN)
from qlib.data import D  # noqa: E402

FIELDS = ["$open", "$high", "$low", "$close", "$volume"]

for f, s, e in [
    ("day", "2019-05-06", "2026-10-03"),
    ("60min", "2026-09-01", "2026-10-03"),
    ("15min", "2026-09-30", "2026-10-03"),
    ("5min", "2026-10-01", "2026-10-03"),
    ("1min", "2026-10-02", "2026-10-03"),
]:
    df = D.features(["MNQ"], FIELDS, start_time=s, end_time=e, freq=f)
    rng = df.index.get_level_values("datetime")
    print(f"[{f}] shape={df.shape} nan={int(df.isna().sum().sum())} "
          f"range={rng.min()} .. {rng.max()}")
    print(df.tail(2).to_string(), "\n")

print("5min calendar sample:", D.calendar(freq="5min", start_time="2019-05-06", end_time="2019-05-06 02:00")[:6])
print("day tail:\n", D.features(["MNQ"], ["$close"], start_time="2026-09-28", end_time="2026-10-03", freq="day").to_string())
