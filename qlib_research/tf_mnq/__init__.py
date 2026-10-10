"""tf_mnq: MNQ feature/gate layer + no-lookahead event backtester (Milestone M1).

All consumers must restrict every metric/validation to IN-SAMPLE
(2019-05-06 .. 2023-12-31). The OOS period (2024+) is frozen.
"""

from __future__ import annotations

IS_START = "2019-05-06"
IS_END = "2023-12-31"

__all__ = ["IS_START", "IS_END"]
