import numpy as np
import matplotlib.pyplot as plt

# ZADANIE 5.0: WPLYW WARUNKU POCZATKOWEGO y_(-1) NA ODPOWIEDZ SKOKOWA
# Rownanie roznicowe:  a*y_n + b*y_(n-1) = c*u_n + d*u_(n-1)


# PARAMETRY
a = 1.0
b = 0.5
c = 2.0
d = 4.0


# DZIEDZINA (numer probki)
N = 15
n = np.arange(0, N)


# POBUDZENIE: skok jednostkowy u_n = 1 dla n >= 0
u = np.ones(N)


# ROZNE WARUNKI POCZATKOWE y_(-1) DO PORONANIA
warunki = [-4.0, 0.0, 4.0, 8.0]


# FUNKCJA SYMULUJACA REKURENCJE Z WARUNKIEM POCZATKOWYM y_(-1)
# y_n = ( c*u_n + d*u_(n-1) - b*y_(n-1) ) / a
def symuluj(y_init):
    y = np.zeros(N)
    for k in range(N):
        u_prev = u[k - 1] if k >= 1 else 0.0   # skok wlaczony w n=0, u_(-1)=0
        y_prev = y[k - 1] if k >= 1 else y_init
        y[k] = (c * u[k] + d * u_prev - b * y_prev) / a
    return y


# WZOR ANALITYCZNY:  y_n = 4 - (2 + 0.5*y_(-1)) * (-1/2)^n
def analitycznie(y_init):
    return 4.0 - (2.0 + 0.5 * y_init) * (-0.5) ** n


# WYKRESY
kolory = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Wykres 1: Wplyw roznych warunkow poczatkowych (symulacja)
plt.figure(figsize=(10, 5))
for y_init, kol in zip(warunki, kolory):
    y_sym = symuluj(y_init)
    plt.plot(n, y_sym, color=kol, lw=2, marker="o", markersize=6,
             label=f"$y_{{-1}} = {y_init:g}$")
plt.axhline(4.0, color="gray", ls=":", lw=1.5, label="stan ustalony = 4")
plt.title("Wplyw warunku poczatkowego $y_{-1}$ na odpowiedz skokowa",
          fontsize=13, fontweight="bold")
plt.xlabel("n")
plt.ylabel("$y_n$")
plt.grid(True, ls="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()

# Wykres 2: Weryfikacja symulacji wzorem analitycznym
plt.figure(figsize=(10, 5))
for y_init, kol in zip(warunki, kolory):
    y_sym = symuluj(y_init)
    y_an = analitycznie(y_init)
    plt.plot(n, y_sym, color=kol, lw=2, marker="o", markersize=6,
             label=f"symulacja, $y_{{-1}} = {y_init:g}$")
    plt.plot(n, y_an, color="black", ls="--", lw=1.5, marker="x", markersize=7)
plt.plot([], [], color="black", ls="--", marker="x",
         label="wzor analityczny")
plt.title("Porownanie symulacji ze wzorem analitycznym",
          fontsize=13, fontweight="bold")
plt.xlabel("n")
plt.ylabel("$y_n$")
plt.grid(True, ls="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()

plt.show()
