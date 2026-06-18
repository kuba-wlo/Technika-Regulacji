import numpy as np
import matplotlib.pyplot as plt

# ZADANIE 4.0: ODPOWIEDZ SKOKOWA I IMPULSOWA
# Rownanie roznicowe:  a*y_n + b*y_(n-1) = c*u_n + d*u_(n-1)


# PARAMETRY
a = 1.0
b = 0.5
c = 2.0
d = 4.0


# DZIEDZINA (numer probki)
N = 15
n = np.arange(0, N)


# POBUDZENIA
u_imp = np.zeros(N)
u_imp[0] = 1.0          # impuls Kroneckera: u_n = delta_n
u_skok = np.ones(N)     # skok jednostkowy: u_n = 1 dla n >= 0


# FUNKCJA SYMULUJACA REKURENCJE (zerowe warunki poczatkowe)
# y_n = ( c*u_n + d*u_(n-1) - b*y_(n-1) ) / a
def symuluj(u):
    y = np.zeros(N)
    for k in range(N):
        u_prev = u[k - 1] if k >= 1 else 0.0
        y_prev = y[k - 1] if k >= 1 else 0.0
        y[k] = (c * u[k] + d * u_prev - b * y_prev) / a
    return y


# ROZWIAZANIE SYMULACYJNE
h_sym = symuluj(u_imp)      # odpowiedz impulsowa
y_sym = symuluj(u_skok)     # odpowiedz skokowa


# ROZWIAZANIE ANALITYCZNE (transformata Z)
# Impulsowa:  H(z) = 2(z+2)/(z+0.5) = 8 - 6 z/(z+0.5)  ->  h_n = 8*delta_n - 6*(-1/2)^n
h_an = -6.0 * (-0.5) ** n
h_an[0] += 8.0

# Skokowa:  Y(z) = H(z)*z/(z-1) = 4 z/(z-1) - 2 z/(z+0.5)  ->  y_n = 4 - 2*(-1/2)^n
y_an = 4.0 - 2.0 * (-0.5) ** n


# WYKRESY

# Wykres 1: Odpowiedz impulsowa
plt.figure(figsize=(10, 5))
plt.plot(n, h_sym, color="#1f77b4", lw=2, marker="o", markersize=7,
         label="Rozwiazanie symulacyjne (rekurencja)")
plt.plot(n, h_an, color="#ff7f0e", ls="--", lw=2, marker="x",
         markersize=8, label="Rozwiazanie analityczne")
plt.title("Odpowiedz impulsowa systemu", fontsize=13, fontweight="bold")
plt.xlabel("n")
plt.ylabel("$h_n$")
plt.axhline(0, color="gray", lw=0.8)
plt.grid(True, ls="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()

# Wykres 2: Odpowiedz skokowa
plt.figure(figsize=(10, 5))
plt.plot(n, y_sym, color="#1f77b4", lw=2, marker="o", markersize=7,
         label="Rozwiazanie symulacyjne (rekurencja)")
plt.plot(n, y_an, color="#ff7f0e", ls="--", lw=2, marker="x",
         markersize=8, label="Rozwiazanie analityczne")
plt.axhline(4.0, color="red", ls=":", lw=1.5, label="stan ustalony = 4")
plt.title("Odpowiedz skokowa systemu", fontsize=13, fontweight="bold")
plt.xlabel("n")
plt.ylabel("$y_n$")
plt.ylim(0, 6)
plt.grid(True, ls="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()

plt.show()
