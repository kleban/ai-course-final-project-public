"""Generate the synthetic demo dataset in samples/ (regression: apartment price).

The numbers are invented for demonstrations and tests of stage 02; they are not market data.
Run: .venv\\Scripts\\python.exe tools\\make_demo_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DISTRICTS = {"Центр": 450, "Поділ": 300, "Оболонь": 120, "Позняки": 60, "Троєщина": 0}


def build(rows: int = 150, seed: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area = rng.normal(62, 18, rows).clip(24, 140).round(1)
    rooms = np.clip(np.round(area / 25 + rng.normal(0, 0.5, rows)), 1, 5).astype(int)
    floor = rng.integers(1, 26, rows)
    distance = rng.gamma(2.2, 3.0, rows).clip(0.5, 25).round(1)
    year = rng.integers(1955, 2025, rows)
    district = rng.choice(list(DISTRICTS), rows, p=[0.15, 0.15, 0.25, 0.25, 0.2])
    price = (
        900
        + 38 * area
        - 45 * distance
        + 6 * (year - 1990)
        + np.array([DISTRICTS[d] for d in district])
        + rng.normal(0, 220, rows)
    ).round(0)
    frame = pd.DataFrame(
        {
            "площа_м2": area,
            "кімнат": rooms,
            "поверх": floor,
            "до_центру_км": distance,
            "рік_побудови": year,
            "район": district,
            "ціна_тис_грн": price,
        }
    )
    frame.loc[rng.choice(rows, 6, replace=False), "рік_побудови"] = np.nan
    frame.loc[rng.choice(rows, 3, replace=False), "до_центру_км"] = np.nan
    frame["рік_побудови"] = frame["рік_побудови"].astype("Int64")
    return frame


if __name__ == "__main__":
    data = build()
    target = ROOT / "samples"
    target.mkdir(exist_ok=True)
    data.to_csv(target / "demo_apartments.csv", sep=";", index=False, encoding="utf-8")
    print(f"{len(data)} rows → {target}")
