"""Independent Melbourne control: stdlib CSV, explicit sums and sorted quantiles.

No pandas or production statistics imports. Output is a reproducible reference artifact.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def reference(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        names = reader.fieldnames or []
        rows = list(reader)
    missing = {name: sum(row[name] == "" for row in rows) for name in names}
    summary = {}
    for name in ("Price", "Rooms", "BuildingArea"):
        values = sorted(float(row[name]) for row in rows if row[name] != "")
        count = len(values)
        mean = math.fsum(values) / count

        def quantile(p: float) -> float:
            position = (count - 1) * p
            low = math.floor(position)
            high = math.ceil(position)
            return values[low] + (values[high] - values[low]) * (position - low)

        summary[name] = {
            "count": count,
            "mean": mean,
            "std": math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (count - 1)),
            "min": values[0],
            "q1": quantile(0.25),
            "median": quantile(0.5),
            "q3": quantile(0.75),
            "max": values[-1],
        }
    pairs = [
        (float(row["Price"]), float(row["Rooms"])) for row in rows if row["Price"] and row["Rooms"]
    ]
    count = len(pairs)
    mx = math.fsum(x for x, _ in pairs) / count
    my = math.fsum(y for _, y in pairs) / count
    r = math.fsum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(
        math.fsum((x - mx) ** 2 for x, _ in pairs) * math.fsum((y - my) ** 2 for _, y in pairs)
    )
    return {
        "source": "samples/melb_data.csv",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "method": "stdlib csv; blank cells; math.fsum; n-1 std; sorted linear quantiles; centered Pearson sums",
        "rows": len(rows),
        "columns": len(names),
        "missing": missing,
        "rows_with_missing": sum(any(row[name] == "" for name in names) for row in rows),
        "descriptive": summary,
        "Price_Rooms": {"r": r, "n": count},
    }


if __name__ == "__main__":
    destination = ROOT / "tests" / "fixtures" / "statistics-melbourne-reference.json"
    destination.write_text(
        json.dumps(reference(ROOT / "samples" / "melb_data.csv"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(destination)
