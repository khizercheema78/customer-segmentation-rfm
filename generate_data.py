"""
Generates a realistic synthetic e-commerce transactions dataset
(customers in Pakistani cities, 2024-2025) for the segmentation project.
The data is synthetic so the project runs anywhere without downloads.

Run:  python generate_data.py   -> data/transactions.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N_CUSTOMERS = 1200
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
CITIES = ["Lahore", "Karachi", "Islamabad", "Sialkot", "Faisalabad", "Multan", "Peshawar"]
CATEGORIES = {"Electronics": 9000, "Fashion": 2500, "Home": 3500, "Beauty": 1500, "Groceries": 1200}

# Hidden behaviour profiles -> the model should rediscover something like these.
PROFILES = {
    #             share, orders/yr, basket multiplier, days since last order (mean)
    "loyal_big":  (0.12, 18, 2.2, 10),
    "regular":    (0.33, 8, 1.0, 35),
    "occasional": (0.30, 3, 0.8, 120),
    "lapsed":     (0.25, 2, 0.9, 380),
}


def main():
    rows, cid = [], 1000
    names = list(PROFILES)
    shares = [PROFILES[n][0] for n in names]
    for _ in range(N_CUSTOMERS):
        cid += 1
        prof = rng.choice(names, p=shares)
        _, per_year, mult, recency_mean = PROFILES[prof]
        n_orders = max(1, rng.poisson(per_year * 2))
        last = END - pd.Timedelta(days=int(min(rng.exponential(recency_mean), 700)))
        span = (last - START).days
        days = np.sort(rng.integers(0, max(span, 1), size=n_orders))
        city = rng.choice(CITIES, p=[0.25, 0.25, 0.12, 0.1, 0.12, 0.08, 0.08])
        for i, d in enumerate(days):
            date = last if i == n_orders - 1 else START + pd.Timedelta(days=int(d))
            cat = rng.choice(list(CATEGORIES))
            amount = rng.gamma(2.0, CATEGORIES[cat] / 2) * mult
            rows.append({
                "invoice_id": f"INV{len(rows) + 1:06d}",
                "customer_id": f"C{cid}",
                "invoice_date": date.date(),
                "city": city,
                "category": cat,
                "quantity": int(rng.integers(1, 4)),
                "amount_pkr": round(float(amount), 2),
            })
    df = pd.DataFrame(rows).sort_values("invoice_date")
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)
    print(f"Saved {len(df):,} transactions for {df.customer_id.nunique():,} customers -> data/transactions.csv")


if __name__ == "__main__":
    main()
