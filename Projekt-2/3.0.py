import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def hurwitz_matrix(coeffs: np.ndarray) -> np.ndarray:
    """
    Hurwitz matrix for polynomial:
      a0 s^n + a1 s^(n-1) + ... + an
    coeffs = [a0, a1, ..., an]
    """
    coeffs = np.asarray(coeffs, dtype=float)
    n = len(coeffs) - 1
    a = coeffs

    h = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            # n=4 example (a0..a4):
            # [a1 a3 0  0]
            # [a0 a2 a4 0]
            # [0  a1 a3 0]
            # [0  a0 a2 a4]
            idx = 2 * j + (1 - i)
            if 0 <= idx <= n:
                h[i, j] = a[idx]
    return h


def hurwitz_principal_minors(coeffs: np.ndarray) -> np.ndarray:
    h = hurwitz_matrix(coeffs)
    n = h.shape[0]
    deltas = np.zeros(n, dtype=float)
    for k in range(1, n + 1):
        deltas[k - 1] = float(np.linalg.det(h[:k, :k]))
    return deltas


def is_hurwitz_stable(coeffs: np.ndarray, eps: float = 1e-10) -> tuple[bool, np.ndarray]:
    coeffs = np.asarray(coeffs, dtype=float)
    if np.any(~np.isfinite(coeffs)):
        return False, np.array([])
    if coeffs[0] <= 0:
        return False, np.array([])
    deltas = hurwitz_principal_minors(coeffs)
    return bool(np.all(deltas > eps)), deltas


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


def step_response(num: np.ndarray, den: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sys = signal.TransferFunction(np.asarray(num, dtype=float), np.asarray(den, dtype=float))
    t = pick_timebase(den)
    t, y = signal.step(sys, T=t)
    return t, y


def main() -> None:
    # Twoje parametry
    a = 12.0
    b = 53.0
    c = 102.0
    d = 72.0
    k = 12.0

    # Uklad otwarty: L(s) = k / (s^4 + a s^3 + b s^2 + c s + d)
    den_open = np.array([1.0, a, b, c, d], dtype=float)
    num_open = np.array([k], dtype=float)

    # Uklad zamkniety (sprzezenie jednostkowe): T(s) = L(s) / (1 + L(s)) = k / (den + k)
    den_closed = np.array([1.0, a, b, c, d + k], dtype=float)
    num_closed = np.array([k], dtype=float)

    stable_open, deltas_open = is_hurwitz_stable(den_open)
    stable_closed, deltas_closed = is_hurwitz_stable(den_closed)

    print("=== Hurwitz ===")
    print(f"Otwarty:  mianownik = {den_open}")
    print(f"  delty  = {deltas_open}")
    print(f"  stabilny = {stable_open}")
    print(f"Zamkniety: mianownik = {den_closed}")
    print(f"  delty  = {deltas_closed}")
    print(f"  stabilny = {stable_closed}")

    kss_open = k / d
    kss_closed = k / (d + k)
    print("\n=== Wzmocnienie w stanie ustalonym ===")
    print(f"K_ust (otwarty)  = L(0) = k/d = {kss_open:.6f}")
    print(f"K_ust (zamkniety)= T(0) = k/(d+k) = {kss_closed:.6f}")

    t1, y1 = step_response(num_open, den_open)
    t2, y2 = step_response(num_closed, den_closed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(t1, y1, lw=2, color="#2b83ba", label="odpowiedź skokowa")
    ax1.axhline(
        kss_open,
        color="red",
        ls="--",
        lw=1.8,
        label=f"wzmocnienie w stanie ustalonym = {kss_open:.4f}",
    )
    ax1.set_title("Odpowiedź skokowa — układ otwarty")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("y(t)")
    ax1.grid(True, ls="--", alpha=0.6)
    ax1.legend()

    ax2.plot(t2, y2, lw=2, color="#1a9850", label="odpowiedź skokowa")
    ax2.axhline(
        kss_closed,
        color="red",
        ls="--",
        lw=1.8,
        label=f"wzmocnienie w stanie ustalonym = {kss_closed:.4f}",
    )
    ax2.set_title("Odpowiedź skokowa — układ zamknięty")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("y(t)")
    ax2.grid(True, ls="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
