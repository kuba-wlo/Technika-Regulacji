import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.ticker import MultipleLocator


def mikhailov_M_jw(w: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Analitycznie:
      M(s) = s^4 + a s^3 + b s^2 + c s + d
    Po podstawieniu s = j w:
      M(jw) = (w^4 - b w^2 + d) + j (c w - a w^3)
    """
    w = np.asarray(w, dtype=float)
    re = w**4 - b * w**2 + d
    im = c * w - a * w**3
    return re + 1j * im


def hurwitz_deltas_4th(a: float, b: float, c: float, d: float, k: float) -> tuple[float, float, float, float]:
    """
    Dla wielomianu:
      s^4 + a s^3 + b s^2 + c s + (d + k)
    delty Hurwitza (potrzebne do zakresu stabilności po k):
      Δ1 = a
      Δ2 = a b - c
      Δ3 = a b c - a^2 (d+k) - c^2
      Δ4 = (d+k) Δ3   (dla monicznego 4 rzędu)
    """
    d1 = a
    d2 = a * b - c
    d3 = a * b * c - (a**2) * (d + k) - (c**2)
    d4 = (d + k) * d3
    return d1, d2, d3, d4


def is_stable_closed_loop(a: float, b: float, c: float, d: float, k: float, eps: float = 1e-10) -> bool:
    d1, d2, d3, _ = hurwitz_deltas_4th(a, b, c, d, k)
    return bool((d + k) > eps and d1 > eps and d2 > eps and d3 > eps)


def k_stability_bounds(a: float, b: float, c: float, d: float) -> tuple[float, float]:
    """
    Z warunków Hurwitza dla 4 rzędu jedyne ograniczenie zależne od k to:
      Δ3 > 0  =>  k < (a b c - a^2 d - c^2) / a^2
    oraz
      d + k > 0  =>  k > -d
    """
    k_min = -d
    k_max = (a * b * c - (a**2) * d - (c**2)) / (a**2)
    return float(k_min), float(k_max)


def pick_timebase(den: np.ndarray, points: int = 4000) -> np.ndarray:
    den = np.asarray(den, dtype=float)
    poles = np.roots(den)
    stable_reals = np.real(poles[np.real(poles) < -1e-9])
    if stable_reals.size == 0:
        t_end = 10.0
    else:
        tau_max = 1.0 / np.min(np.abs(stable_reals))
        t_end = float(np.clip(8.0 * tau_max, 5.0, 60.0))
    return np.linspace(0.0, t_end, points)


def step_response_open(a: float, b: float, c: float, d: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    # Uklad otwarty: L(s) = k / (s^4 + a s^3 + b s^2 + c s + d)
    num = np.array([k], dtype=float)
    den = np.array([1.0, a, b, c, d], dtype=float)
    sys = signal.TransferFunction(num, den)
    t = pick_timebase(den)
    t, y = signal.step(sys, T=t)
    return t, y


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # Twoje parametry
    a = 12.0
    b = 53.0
    c = 102.0
    d = 72.0
    k0 = 12.0

    # ===== Michajlow dla ukladu otwartego (M(s)=P(s)) =====
    w = np.linspace(0.0, 200.0, 20000)
    M = mikhailov_M_jw(w, a, b, c, d)
    phi = np.unwrap(np.angle(M))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    mask = w <= 12.0
    ax1.plot(M[mask].real, M[mask].imag, color="#2b83ba", lw=2, label=r"$M(j\omega)$")
    ax1.scatter(M[0].real, M[0].imag, color="red", zorder=3, label=r"$\omega=0$")
    ax1.axhline(0.0, color="black")
    ax1.axvline(0.0, color="black")
    ax1.set_title("Wykres Michajłowa")
    ax1.set_xlabel("Re")
    ax1.set_ylabel("Im")
    ax1.grid(True, ls="--", alpha=0.6)
    ax1.legend()

    ax2.plot(w, phi, color="#1a9850", lw=2, label=r"$\arg M(j\omega)$")
    ax2.axhline(2.0 * np.pi, color="red", ls="--", label=r"Cel (stabilny rząd 4): $2\pi$")
    ax2.set_title("Zmiana argumentu funkcji")
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel("Kąt [rad]")
    ax2.grid(True, ls="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print("=== Michajlow (uklad otwarty) ===")
    print("M(s) = s^4 + a s^3 + b s^2 + c s + d")
    print("M(jw) = (w^4 - b w^2 + d) + j (c w - a w^3)")
    print(f"Delta arg M(jw) (0->{w[-1]:.1f}) = {phi[-1]:.6f} rad, oczekiwane ~ {2.0*np.pi:.6f} rad")
    print("Uwaga: stabilnosc ukladu otwartego nie zalezy od k (k jest tylko w liczniku).")

    # ===== Wplyw k na odpowiedz skokowa (uklad otwarty) =====
    # Jesli P(s) jest stabilny, to dla kazdego k odpowiedz jest stabilna i skaluje sie liniowo.
    den_open = np.array([1.0, a, b, c, d], dtype=float)
    poles_open = np.roots(den_open)
    open_stable = np.all(np.real(poles_open) < 0.0)
    print("\n=== Stabilnosc ukladu otwartego ===")
    print(f"Bieguny P(s): {np.array2string(poles_open, precision=3)}")
    print(f"Stabilny (Re<0): {bool(open_stable)}")

    # Zestaw k do pokazania (w tym blisko granicy i za granicą)
    ks = np.array(
        [
            -60.0,
            0.0,
            k0,
            50.0,
            150.0,
            300.0,
        ],
        dtype=float,
    )

    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
    for k in ks:
        t, y = step_response_open(a, b, c, d, k)
        ax.plot(t, y, lw=2, label=f"k={k:.2f}  (k/d={k/d:.3f})")

    ax.set_title("Wplyw parametru k na odpowiedz skokowa (uklad otwarty)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("y(t)")

    # Ladna, czytelna siatka: start od 0, rowny krok w osi czasu
    ax.set_xlim(0.0, 5.0)
    ax.xaxis.set_major_locator(MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(True, which="major", ls="-", lw=0.9, alpha=0.35, color="#888888")
    ax.grid(True, which="minor", ls=":", lw=0.6, alpha=0.25, color="#aaaaaa")

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
