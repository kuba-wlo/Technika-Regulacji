import numpy as np
import matplotlib.pyplot as plt

# ZADANIE 3.0: ODPOWIEDZ NA ZADANE POBUDZENIE u_n = (-2)^n
# Rownanie roznicowe:  a*y_n + b*y_(n-1) = c*u_n + d*u_(n-1)


# PARAMETRY
a = 1.0
b = 0.5
c = 2.0
d = 4.0


# DZIEDZINA (numer probki)
N = 15
n = np.arange(0, N)


# POBUDZENIE
u = (-2.0) ** n


# ROZWIAZANIE SYMULACYJNE (rekurencja, zerowe warunki poczatkowe)
# y_n = ( c*u_n + d*u_(n-1) - b*y_(n-1) ) / a
y_sym = np.zeros(N)
for k in range(N):
    u_prev = u[k - 1] if k >= 1 else 0.0   # zerowy warunek poczatkowy
    y_prev = y_sym[k - 1] if k >= 1 else 0.0
    y_sym[k] = (c * u[k] + d * u_prev - b * y_prev) / a


# ROZWIAZANIE ANALITYCZNE (transformata Z)
# H(z) = (2z+4)/(z+0.5) = 2(z+2)/(z+0.5),   U(z) = z/(z+2)
# Y(z) = 2(z+2)/(z+0.5) * z/(z+2) = 2z/(z+0.5)   ->   y_n = 2*(-1/2)^n
y_an = 2.0 * (-0.5) ** n


# WYKRESY

# Wykres 1: Pobudzenie
plt.figure(figsize=(10, 5))
plt.plot(n, u, color="#1f77b4", lw=2, marker="o", markersize=7,
         label="$u_n = (-2)^n$")
plt.title("Pobudzenie $u_n = (-2)^n$", fontsize=13, fontweight="bold")
plt.xlabel("n")
plt.ylabel("$u_n$")
plt.axhline(0, color="gray", lw=0.8)
plt.grid(True, ls="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()

# Wykres 2: Odpowiedz - symulacja vs analitycznie
plt.figure(figsize=(10, 5))
plt.plot(n, y_sym, color="#1f77b4", lw=2, marker="o", markersize=7,
         label="Rozwiazanie symulacyjne (rekurencja)")
plt.plot(n, y_an, color="#ff7f0e", ls="--", lw=2, marker="x",
         markersize=8, label="Rozwiazanie analityczne $y_n = 2(-1/2)^n$")
plt.title("Odpowiedz systemu $y_n$", fontsize=13, fontweight="bold")
plt.xlabel("n")
plt.ylabel("$y_n$")
plt.axhline(0, color="gray", lw=0.8)
plt.grid(True, ls="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()

plt.show()
