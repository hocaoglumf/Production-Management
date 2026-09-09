import numpy as np


class DoubleExponentialSmoothingTracker:
    """
    Gemi konumu ve hız kestirimi için Çift Üstel Düzeltme (Holt's Method) Filtresi
    """

    def __init__(self, alpha=0.35, beta=0.25):
        self.alpha = alpha  # Konum düzeltme katsayısı (küçük veri = çok filtre)
        self.beta = beta  # Trend/Hız düzeltme katsayısı
        self.S_x, self.S_y = None, None
        self.T_x, self.T_y = 0.0, 0.0
        self.initialized = False

    def update(self, x_meas, y_meas, dt):
        # İlk veri geldiğinde başlatma adımı
        if not self.initialized:
            self.S_x, self.S_y = x_meas, y_meas
            self.T_x, self.T_y = 0.0, 0.0
            self.initialized = True
            return self.S_x, self.S_y, self.T_x, self.T_y

        # X Ekseni Güncellemesi
        S_x_old = self.S_x
        self.S_x = self.alpha * x_meas + (1 - self.alpha) * (self.S_x + self.T_x * dt)
        self.T_x = self.beta * ((self.S_x - S_x_old) / dt) + (1 - self.beta) * self.T_x

        # Y Ekseni Güncellemesi
        S_y_old = self.S_y
        self.S_y = self.alpha * y_meas + (1 - self.alpha) * (self.S_y + self.T_y * dt)
        self.T_y = self.beta * ((self.S_y - S_y_old) / dt) + (1 - self.beta) * self.T_y

        return self.S_x, self.S_y, self.T_x, self.T_y


def compute_intercept_point(S_x, S_y, T_x, T_y, x_gun, y_gun, v_bullet, max_iter=50, tol=0.01):
    """
    Merminin hedefe ulaşma süresini ve önleme noktasını iteratif olarak hesaplar.
    """
    tau = 1.0  # Başlangıç uçuş süresi tahmini (1 saniye)

    for _ in range(max_iter):
        # tau süre sonra geminin olacağı tahmini konum
        x_pred = S_x + T_x * tau
        y_pred = S_y + T_y * tau

        # Top ile tahmini konum arasındaki geometrik mesafe
        distance = np.sqrt((x_pred - x_gun) ** 2 + (y_pred - y_gun) ** 2)

        # Bu mesafeye göre yeni uçuş süresi
        tau_new = distance / v_bullet

        # İterasyon yakınsadıysa döngüden çık
        if abs(tau_new - tau) < tol:
            return x_pred, y_pred, tau_new

        tau = tau_new

    return S_x + T_x * tau, S_y + T_y * tau, tau


# --- ÖRNEK SENARYO ÇALIŞTIRMASI ---
if __name__ == "__main__":
    dt = 1.0  # Radardan 1 saniyede bir veri geliyor varsayalım
    tracker = DoubleExponentialSmoothingTracker(alpha=0.35, beta=0.25)

    # Top Koordinatları ve Mermi Hızı (Örn: 76mm Deniz Topu namlu çıkış hızı ~850 m/s)
    x_gun, y_gun, v_bullet = 0.0, 0.0, 850.0

    # Geminin zamana bağlı gürültülü radar ölçüm simülasyonu
    # (Gerçekte gemi X=1500, Y=2000 noktasından X yönünde 15 m/s, Y yönünde 5 m/s ile gidiyor olsun)
    simulated_measurements = [
        (1502.1, 2000.5),
        (1516.8, 2004.8),
        (1533.0, 2009.2),
        (1544.5, 2016.1)
    ]

    print("--- Hedef Takip ve Atış Kontrol Döngüsü ---")
    for idx, (x_m, y_m) in enumerate(simulated_measurements):
        # Filtreyi güncelle ve anlık hız vektörünü al
        S_x, S_y, T_x, T_y = tracker.update(x_m, y_m, dt)

        # Önleme noktasını hesapla
        x_int, y_int, tof = compute_intercept_point(S_x, S_y, T_x, T_y, x_gun, y_gun, v_bullet)

        print(f"Adım {idx + 1} | Ölçüm: ({x_m:.1f}, {y_m:.1f})")
        print(f"        -> Filtrelenmiş Hız Hız: Vx={T_x:.2f} m/s, Vy={T_y:.2f} m/s")
        print(f"        -> ÖNLEME NOKTASI: X={x_int:.2f}m, Y={y_int:.2f}m | Mermi Uçuş Süresi (ToF): {tof:.3f} sn\n")
