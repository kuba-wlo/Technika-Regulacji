import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ZADANIE 5.0: BADANIE WPŁYWU WARUNKÓW POCZĄTKOWYCH

# UKŁAD RÓWNAŃ (3 rząd)
# x1 = y, x2 = y', x3 = y''
def dydt_system(u_vec, t):
    x1, x2, x3 = u_vec
    
    u = 1.0  # Odpowiedź skokowa dla 5.0
    
    dx1dt = x2
    dx2dt = x3
    dx3dt = -10*x3 - 31*x2 - 30*x1 + u
    
    return [dx1dt, dx2dt, dx3dt]


# CZAS
t = np.linspace(0, 10, 500)


# WARUNKI POCZĄTKOWE (Różne przypadki do porównania)
u0_1 = [0, 0, 0]      # Zerowe
u0_2 = [0.1, 0, 0]    # Tylko y(0)
u0_3 = [0, 0.5, 0]    # Tylko y'(0)
u0_4 = [0, 0, 1.0]    # Tylko y''(0)


# ROZWIĄZANIE NUMERYCZNE (ODEINT)
sol_1 = odeint(dydt_system, u0_1, t)[:, 0]
sol_2 = odeint(dydt_system, u0_2, t)[:, 0]
sol_3 = odeint(dydt_system, u0_3, t)[:, 0]
sol_4 = odeint(dydt_system, u0_4, t)[:, 0]


# ROZWIĄZANIE ANALITYCZNE (Przykładowe wzory wyliczone wcześniej)
# Dla u0_1 (zerowe):
f_an_1 = (1/30) - (1/6)*np.exp(-2*t) + (1/6)*np.exp(-3*t) - (1/30)*np.exp(-5*t)

# Dla u0_2 (y0=0.1):
f_an_2 = 0.0333 + 0.1167*np.exp(-2*t) - 0.1*np.exp(-3*t) + 0.05*np.exp(-5*t)


# WYKRESY
plt.figure(figsize=(12, 10))

# Podwykres 1: Wszystkie przypadki symulacji
plt.subplot(2, 1, 1)
plt.plot(t, sol_1, label="WP: [0, 0, 0]")
plt.plot(t, sol_2, label="WP: [0.1, 0, 0]")
plt.plot(t, sol_3, label="WP: [0, 0.5, 0]")
plt.plot(t, sol_4, label="WP: [0, 0, 1.0]")
plt.title("Wpływ poszczególnych warunków początkowych na odpowiedź skokową")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid()
plt.legend()
plt.xlim(0, 5)

# Podwykres 2: Weryfikacja analityczna dla wybranego przypadku (np. y0=0.1)
plt.subplot(2, 1, 2)
plt.plot(t, sol_2, 'b', label="Symulacja ODE (y0=0.1)")
plt.plot(t, f_an_2, 'r--', label="Wzór analityczny (y0=0.1)")
plt.title("Porównanie symulacji z wyliczonym wzorem")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid()
plt.legend()
plt.xlim(0, 5)

plt.tight_layout()
plt.show()