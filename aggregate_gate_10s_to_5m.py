import csv
import gzip
import json
import os
from pathlib import Path


SOURCE_DIR = Path(r"D:\freqtrade\user_data\data\gate_10s\monthly")
TARGET_DIR = Path(r"D:\freqtrade\user_data\data\gate")


def write_timeframe(minutes: int, label: str) -> int:
    files = sorted(SOURCE_DIR.glob("BTC_USDT-*.csv.gz"))
    TARGET = TARGET_DIR / f"BTC_USDT-{label}.json.gz"
    temporary = TARGET.with_suffix(".tmp.gz")
    count = 0
    current_bucket = None
    candle = None

    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as output:
        output.write("[")
        for source in files:
            with gzip.open(source, "rt", encoding="utf-8", newline="") as stream:
                for row in csv.reader(stream):
                    if len(row) < 6:
                        continue
                    timestamp = int(float(row[0]))
                    bucket = (timestamp // (minutes * 60)) * (minutes * 60)
                    volume = float(row[1])
                    close = float(row[2])
                    high = float(row[3])
                    low = float(row[4])
                    open_price = float(row[5])

                    if bucket != current_bucket:
                        if candle is not None:
                            if count:
                                output.write(",")
                            output.write(json.dumps(candle, separators=(",", ":")))
                            count += 1
                        current_bucket = bucket
                        candle = [bucket * 1000, open_price, high, low, close, volume]
                    else:
                        candle[2] = max(candle[2], high)
                        candle[3] = min(candle[3], low)
                        candle[4] = close
                        candle[5] += volume

        if candle is not None:
            if count:
                output.write(",")
            output.write(json.dumps(candle, separators=(",", ":")))
            count += 1
        output.write("]")

    os.replace(temporary, TARGET)
    return count


if __name__ == "__main__":
    for minutes, label in ((5, "5m"), (15, "15m"), (30, "30m"), (60, "1h"), (240, "4h"), (1440, "1d")):
        print(f"{label} candles: {write_timeframe(minutes, label)}")
