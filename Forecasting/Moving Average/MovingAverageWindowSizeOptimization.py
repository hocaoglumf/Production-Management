import numpy as np


def optimize_moving_average(demand, windows=range(1, 7)):
    """
    Select a simple moving average window using rolling forecasts.

    All windows are evaluated on the same periods, starting after
    the largest candidate window.

    Returns:
        best_window: Window with the lowest MAE.
        results: Error measures for each candidate window.
        next_forecast: Forecast using the best window.
    """
    demand = np.asarray(demand, dtype=float)

    if demand.ndim != 1 or not np.isfinite(demand).all():
        raise ValueError("Demand must be a finite, one-dimensional sequence.")

    windows = sorted(set(windows))
    if not windows or any(
        not isinstance(w, (int, np.integer)) or w < 1
        for w in windows
    ):
        raise ValueError("Windows must contain positive integers.")

    start = max(windows)
    if len(demand) <= start:
        raise ValueError(
            "Demand must contain more observations than the largest window."
        )

    actual = demand[start:]
    results = []

    for window in windows:
        forecasts = np.array([
            demand[t - window:t].mean()
            for t in range(start, len(demand))
        ])

        errors = actual - forecasts

        results.append({
            "window": window,
            "MAE": float(np.mean(np.abs(errors))),
            "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        })

    # Prefer the smaller window if MAE values tie.
    best = min(results, key=lambda row: (row["MAE"], row["window"]))
    best_window = best["window"]
    next_forecast = float(demand[-best_window:].mean())

    return best_window, results, next_forecast


# Demand values from your Excel example
#demand = [
#    6000, 5000, 8000, 7000, 4000, 7000,
#    6000, 8000, 9000, 10000, 12000, 13000
#]
demand = [
    98, 85, 108, 99, 93, 89, 103, 107,
    118, 97, 86, 114, 105, 108, 83, 92,
    111, 114, 97, 112, 114, 95, 116, 91
]

best_window, results, next_forecast = optimize_moving_average(
    demand,
    windows=range(1, 7)
)

print(f"{'Window':>6} {'MAE':>12} {'RMSE':>12}")
for row in results:
    print(
        f"{row['window']:>6} "
        f"{row['MAE']:>12,.2f} "
        f"{row['RMSE']:>12,.2f}"
    )

print(f"\nBest window: {best_window}")
print(f"Next-period forecast: {next_forecast:,.2f}")