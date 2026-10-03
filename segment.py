"""
Customer Segmentation with RFM + K-Means
----------------------------------------
1. Builds Recency, Frequency, Monetary (RFM) features per customer
2. Log-transforms and scales them
3. Picks the number of clusters with the elbow method + silhouette score
4. Clusters customers with K-Means and labels each segment in business terms
5. Exports segment profiles and an actionable customer list

Run:  python generate_data.py && python segment.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    snapshot = df["invoice_date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        recency=("invoice_date", lambda d: (snapshot - d.max()).days),
        frequency=("invoice_id", "nunique"),
        monetary=("amount_pkr", "sum"),
        city=("city", "first"),
    )
    return rfm


def choose_k(X: np.ndarray) -> int:
    ks, inertias, sils = range(2, 9), [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ks, inertias, "o-", color="#7aa2f7", label="Inertia (elbow)")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia")
    ax2 = ax1.twinx()
    ax2.plot(ks, sils, "s--", color="#bb9af7", label="Silhouette")
    ax2.set_ylabel("Silhouette score")
    fig.suptitle("Choosing k: elbow method and silhouette score")
    fig.tight_layout()
    fig.savefig(OUT / "choose_k.png", dpi=150)
    plt.close(fig)
    print("k  silhouette")
    for k, s in zip(ks, sils):
        print(f"{k}  {s:.3f}")
    # Business-friendly choice: best silhouette among k = 3..6
    best = max(range(3, 7), key=lambda k: sils[k - 2])
    print(f"Chosen k = {best}")
    return best


def name_segments(profile: pd.DataFrame) -> dict:
    """Map cluster ids to readable names using their relative RFM ranks."""
    score = (profile["frequency"].rank() + profile["monetary"].rank()
             - profile["recency"].rank())
    order = score.sort_values(ascending=False).index.tolist()
    labels = ["Champions", "Loyal Customers", "Potential Loyalists", "Need Attention",
              "At Risk", "Hibernating"]
    names = {cid: labels[i] for i, cid in enumerate(order[:-1])}
    names[order[-1]] = "Lost / Hibernating"
    return names


def main():
    df = pd.read_csv("data/transactions.csv")
    rfm = build_rfm(df)
    print(f"Customers: {len(rfm):,}")

    X = StandardScaler().fit_transform(np.log1p(rfm[["recency", "frequency", "monetary"]]))
    k = choose_k(X)
    rfm["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

    profile = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    names = name_segments(profile)
    rfm["segment"] = rfm["cluster"].map(names)

    summary = (rfm.groupby("segment")
               .agg(customers=("cluster", "size"),
                    avg_recency_days=("recency", "mean"),
                    avg_orders=("frequency", "mean"),
                    avg_spend_pkr=("monetary", "mean"),
                    total_revenue_pkr=("monetary", "sum"))
               .sort_values("avg_spend_pkr", ascending=False))
    summary["revenue_share"] = summary["total_revenue_pkr"] / summary["total_revenue_pkr"].sum()
    print("\nSegment profiles:")
    print(summary.round(1).to_string())

    summary.to_csv(OUT / "segment_summary.csv")
    rfm.to_csv(OUT / "customer_segments.csv")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for seg, g in rfm.groupby("segment"):
        ax.scatter(g["recency"], g["monetary"], s=12 + g["frequency"] * 2, alpha=0.6, label=seg)
    ax.set_xlabel("Recency (days since last order)")
    ax.set_ylabel("Total spend (PKR)")
    ax.set_yscale("log")
    ax.set_title("Customer segments (bubble size = number of orders)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "segments.png", dpi=150)
    plt.close(fig)

    actions = {
        "Champions": "Reward with VIP perks, early access, referral codes",
        "Loyal Customers": "Upsell premium categories, loyalty points",
        "Potential Loyalists": "Personalised recommendations, membership offer",
        "Need Attention": "Limited-time discounts on favourite categories",
        "At Risk": "Win-back email/SMS with strong incentive",
        "Lost / Hibernating": "Low-cost reactivation campaign or exclude from spend",
    }
    print("\nRecommended actions:")
    for seg in summary.index:
        print(f"- {seg}: {actions.get(seg, '')}")


if __name__ == "__main__":
    main()
