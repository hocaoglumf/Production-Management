import numpy as np
import pandas as pd


def optimize_exponential_smoothing(data):
    """
    Üstel düzleştirme modelinde 0.1 - 1.0 aralığındaki alpha değerlerini test eder,
    Talep-Tahmin tablosunu ve metrik sonuçlarını üretir.
    """
    alphas = np.round(np.arange(0.1, 1.1, 0.1), 1)

    # Talep ve Tahmin tablosunu tutacak veri yapısı
    demand_forecast_dict = {
        'Dönem (t)': list(range(1, len(data) + 1)),
        'Gerçekleşen Talep (Y_t)': data
    }

    metrics_results = []

    for alpha in alphas:
        forecasts = [data[0]]  # İlk dönemin tahmini ilk gerçekleşen değere eşit atanır

        for t in range(1, len(data)):
            f_next = alpha * data[t - 1] + (1 - alpha) * forecasts[t - 1]
            forecasts.append(f_next)

        # Her alpha için tahmin sütununu ekle
        demand_forecast_dict[f'Tahmin (α={alpha})'] = np.round(forecasts, 2)

        # Hata hesaplamaları
        errors = np.array(data) - np.array(forecasts)

        # Performans Metrikleri
        mad = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        tracking_signal = np.sum(errors) / mad if mad != 0 else 0

        metrics_results.append({
            'Alpha': alpha,
            'MSE': round(mse, 4),
            'MAD': round(mad, 4),
            'Tracking Signal': round(tracking_signal, 4),
            '|TS| (0\'a Yakınlık)': round(abs(tracking_signal), 4)
        })

    # Dataframe Dönüşümleri
    df_demand_forecasts = pd.DataFrame(demand_forecast_dict)
    df_metrics = pd.DataFrame(metrics_results)

    # Optimum Değerlerin Seçimi
    best_mse = df_metrics.loc[df_metrics['MSE'].idxmin()]
    best_mad = df_metrics.loc[df_metrics['MAD'].idxmin()]
    best_ts = df_metrics.loc[df_metrics['|TS| (0\'a Yakınlık)'].idxmin()]

    return df_demand_forecasts, df_metrics, best_mse, best_mad, best_ts


def get_alpha_detail_table(data, alpha):
    """Belirli bir alpha değeri için detaylı hata döküm tablosu oluşturur."""
    forecasts = [data[0]]
    for t in range(1, len(data)):
        f_next = alpha * data[t - 1] + (1 - alpha) * forecasts[t - 1]
        forecasts.append(f_next)

    errors = np.array(data) - np.array(forecasts)

    return pd.DataFrame({
        'Dönem (t)': range(1, len(data) + 1),
        'Talep (Y_t)': data,
        'Tahmin (F_t)': np.round(forecasts, 2),
        'Hata (e_t)': np.round(errors, 2),
        '|Hata|': np.round(np.abs(errors), 2),
        'Hata^2': np.round(errors ** 2, 2)
    })


# --- ÖRNEK KULLANIM ---
if __name__ == "__main__":
    # Örnek Talep Verisi
    data = [120, 135, 128, 145, 150, 142, 160, 155, 165, 170]

    df_demand_forecasts, df_metrics, best_mse, best_mad, best_ts = optimize_exponential_smoothing(data)

    print("=== 1. TÜM ALPHA DEĞERLERİ İÇİN TALEP VE TAHMİN TABLOSU ===")
    print(df_demand_forecasts.to_string(index=False))

    print("\n=== 2. PERFORMANS VE METRİK TABLOSU ===")
    print(df_metrics[['Alpha', 'MSE', 'MAD', 'Tracking Signal']].to_string(index=False))

    print("\n=== 3. OPTİMUM SONUÇLAR ===")
    print(f"• En Düşük MSE Veren Alpha: {best_mse['Alpha']} (MSE: {best_mse['MSE']})")
    print(f"• En Düşük MAD Veren Alpha: {best_mad['Alpha']} (MAD: {best_mad['MAD']})")
    print(f"• 0'a En Yakın Tracking Signal Veren Alpha: {best_ts['Alpha']} (TS: {best_ts['Tracking Signal']})")

    print(f"\n=== 4. OPTİMUM (EN DÜŞÜK MSE) ALPHA={best_mse['Alpha']} İÇİN DETAYLI DÖNEM TABLOSU ===")
    df_detail = get_alpha_detail_table(data, best_mse['Alpha'])
    print(df_detail.to_string(index=False))