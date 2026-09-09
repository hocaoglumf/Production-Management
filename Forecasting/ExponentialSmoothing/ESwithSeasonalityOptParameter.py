import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -------------------------------------------------
# Veri
# -------------------------------------------------
data = [
    120, 135, 150, 160,
    140, 155, 170, 180,
    160, 175, 190, 205,
    185, 200, 215, 225
]

series = pd.Series(data)

# Mevsim uzunluğu
season_length = 4

# -------------------------------------------------
# Grid Search
# -------------------------------------------------
results = []

best_alpha = None
best_beta = None
best_sse = float("inf")

alphas = np.arange(0.0, 1.01, 0.1)
betas = np.arange(0.0, 1.01, 0.1)

for alpha in alphas:
    for beta in betas:

        try:
            model = ExponentialSmoothing(
                series,
                trend=None,
                seasonal="add",
                seasonal_periods=season_length
            )

            fit = model.fit(
                smoothing_level=alpha,
                smoothing_seasonal=beta,
                optimized=False
            )

            fitted = fit.fittedvalues

            # SSE
            sse = np.sum((series - fitted) ** 2)

            # Sonuçları kaydet
            results.append([
                alpha,
                beta,
                sse,
                f"α={alpha:.1f}, β={beta:.1f}"
            ])

            # En iyi sonucu bul
            if sse < best_sse:
                best_sse = sse
                best_alpha = alpha
                best_beta = beta

        except Exception:
            pass


# -------------------------------------------------
# Sonuçları DataFrame'e aktar
# -------------------------------------------------
results_df = pd.DataFrame(
    results,
    columns=["Alpha", "Beta", "SSE", "Combination"]
)

# SSE'ye göre sırala
#results_df = results_df.sort_values("SSE").reset_index(drop=True)


# -------------------------------------------------
# En iyi sonucu yazdır
# -------------------------------------------------
print("En iyi parametreler")
print("-------------------")
print(f"Alpha : {best_alpha:.1f}")
print(f"Beta  : {best_beta:.1f}")
print(f"SSE   : {best_sse:.2f}")

print("\nEn iyi 10 kombinasyon:")
print(results_df.head(10))


# -------------------------------------------------
# SSE Grafiği
# -------------------------------------------------
plt.figure(figsize=(16, 6))

plt.plot(
    results_df["Combination"],
    results_df["SSE"],
    marker="o"
)

# En iyi noktayı kırmızı olarak göster
best_index = results_df["SSE"].idxmin()

plt.scatter(
    results_df.loc[best_index, "Combination"],
    results_df.loc[best_index, "SSE"],
    s=100,
    zorder=5,
    label="En iyi kombinasyon"
)

plt.xlabel("Alpha - Beta kombinasyonu")
plt.ylabel("SSE")
plt.title("Alpha ve Beta Kombinasyonlarına Göre SSE")
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()