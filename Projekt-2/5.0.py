import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# Parametry z zadania
a = 12.0
b = 53.0
c = 102.0
d = 72.0
k0 = 12.0
KS = np.array([-60.0, 0.0, 12.0, 50.0, 150.0, 300.0, 327.0], dtype=float)


def p_jw(w: np.ndarray) -> np.ndarray:
    # P(s) = s^4 + a s^3 + b s^2 + c s + d
    s = 1j * w
    return s**4 + a * s**3 + b * s**2 + c * s + d


def k_otw_jw(w: np.ndarray, k: float) -> np.ndarray:
    # K_otw(s) = k / P(s)
    return k / p_jw(w)


def is_stable_closed(k: float, eps: float = 1e-10) -> bool:
    # Uklad zamkniety: s^4 + a s^3 + b s^2 + c s + (d+k)
    d1 = a
    d2 = a * b - c
    d3 = a * b * c - a**2 * (d + k) - c**2
    return bool((d + k) > eps and d1 > eps and d2 > eps and d3 > eps)


def k_bounds_closed() -> tuple[float, float]:
    k_min = -d
    k_max = (a * b * c - a**2 * d - c**2) / a**2
    return float(k_min), float(k_max)


def step_closed(k: float) -> tuple[np.ndarray, np.ndarray]:
    # T(s) = k / (P(s) + k)
    num = np.array([k], dtype=float)
    den = np.array([1.0, a, b, c, d + k], dtype=float)
    sys = signal.TransferFunction(num, den)
    t = np.linspace(0.0, 12.0, 6000)
    t, y = signal.step(sys, T=t)
    return t, y


def nyquist_plot_single(k: float) -> None:
    w = np.linspace(0.0, 60.0, 12000)
    k_jw = k_otw_jw(w, k)
    phi = np.unwrap(np.angle(1.0 + k_jw))
    status = "stabilny" if is_stable_closed(k) else "NIEstabilny"
    style = "-" if is_stable_closed(k) else "--"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(k_jw.real, k_jw.imag, lw=2, ls=style, color="#2b83ba", label=f"k={k:.2f} ({status})")
    ax1.axhline(0.0, color="black")
    ax1.axvline(0.0, color="black")
    ax1.set_title(f"Wykres Nyquista (k={k:.2f})")
    ax1.set_xlabel("Re")
    ax1.set_ylabel("Im")
    ax1.grid(True, ls="--", alpha=0.6)
    ax1.legend()

    ax2.plot(w, phi, lw=2, ls=style, color="#1a9850", label=r"$\Delta arg[1 + K_{otw}(j\omega)]$")
    ax2.axhline(0.0, color="red", ls="--", label="Granica stabilnosci (0 rad)")
    ax2.set_title(f"Zmiana argumentu 1 + K_otw(jw) (k={k:.2f})")
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel("Kat [rad]")
    ax2.grid(True, ls="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"Nyquist: k={k:.2f}, Delta arg = {phi[-1]:.6f} rad, stabilny={is_stable_closed(k)}")


def nyquist_plot_all_ks(ks: np.ndarray) -> None:
    w = np.linspace(0.0, 60.0, 12000)
    plt.figure(figsize=(12, 6))
    for k in ks:
        k_jw = k_otw_jw(w, k)
        style = "-" if is_stable_closed(k) else "--"
        status = "stabilny" if is_stable_closed(k) else "NIEstabilny"
        plt.plot(k_jw.real, k_jw.imag, lw=2, ls=style, label=f"k={k:.2f} ({status})")

    plt.axhline(0.0, color="black")
    plt.axvline(0.0, color="black")
    plt.title("Nyquisty dla k uzytych na wykresie odpowiedzi skokowej")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nNyquist dla k z listy:")
    for k in ks:
        k_jw = k_otw_jw(w, k)
        phi = np.unwrap(np.angle(1.0 + k_jw))
        print(f"k={k:.2f} -> Delta arg = {phi[-1]:.6f} rad, stabilny={is_stable_closed(k)}")


def k_impact_step_plot() -> None:
    k_min, k_max = k_bounds_closed()
    print("\nZakres stabilnosci ukladu zamknietego:")
    print(f"{k_min:.6f} < k < {k_max:.6f}")

    ks = KS

    # Zakres Y z przebiegow stabilnych (zostawiamy czytelny wykres porownawczy)
    y_stable = []
    for k in ks:
        if is_stable_closed(k):
            _, y = step_closed(k)
            y_stable.append(y)

    if y_stable:
        yy = np.concatenate(y_stable)
        y_lo = float(np.min(yy))
        y_hi = float(np.max(yy))
        pad = 0.1 * max(1e-9, y_hi - y_lo)
        y_lo -= pad
        y_hi += pad
    else:
        y_lo, y_hi = -2.0, 2.0

    plt.figure(figsize=(12, 6))
    for k in ks:
        t, y = step_closed(k)
        if is_stable_closed(k):
            plt.plot(t, y, lw=2, label=f"k={k:.2f} (stabilny)")
        else:
            den = np.array([1.0, a, b, c, d + k], dtype=float)
            poles = np.roots(den)
            y_clip = np.clip(y, y_lo, y_hi)
            plt.plot(t, y_clip, lw=2, ls="--", label=f"k={k:.2f} (NIEstabilny, przyciety)")
            print(f"k={k:.2f} -> NIEstabilny, bieguny: {np.array2string(poles, precision=3)}")

    plt.title("Wplyw parametru k na odpowiedz skokowa (uklad zamkniety)")
    plt.xlabel("t [s]")
    plt.ylabel("y(t)")
    plt.xlim(0.0, 12.0)
    plt.ylim(y_lo, y_hi)
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    nyquist_plot_single(k0)
    k_impact_step_plot()
    nyquist_plot_all_ks(KS)


if __name__ == "__main__":
    main()
