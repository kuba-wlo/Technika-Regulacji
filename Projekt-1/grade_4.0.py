import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# UKŁAD RÓWNAŃ (3 rząd)
# x1 = y, x2 = y', x3 = y''
def dydt_system(u_vec, t, mode):
    x1, x2, x3 = u_vec
    
    # Wybór wymuszenia u(t) w zależności od zadania
    if mode == 'skok':
        u = 1.0              # Pobudzenie skokowe do 4.0 
    else:
        u = 0.0              # Dla impulsu (u=0, wymuszenie warunkiem pocz.)
    
    dx1dt = x2
    dx2dt = x3
    dx3dt = -10*x3 - 31*x2 - 30*x1 + u
    
    return [dx1dt, dx2dt, dx3dt]


# CZAS
t = np.linspace(0, 10, 500)


# WARUNKI POCZĄTKOWE
y0 = 0
y0p = 0
y0pp = 0
u0 = [y0, y0p, y0pp]


# ROZWIĄZANIE NUMERYCZNE (ODEINT)

# 1. Odpowiedź skokowa
sol_skok_num = odeint(dydt_system, u0, t, args=('skok',))[:, 0]

# 2. Odpowiedź impulsowa
# Symulacja impulsu przez warunek początkowy y''(0) = 1 
u0_imp = [0, 0, 1] 
sol_imp_num = odeint(dydt_system, u0_imp, t, args=('impuls',))[:, 0]


# ROZWIĄZANIE ANALITYCZNE

# Analityczna odpowiedź skokowa
f_skok_an = (1/30) \
          - (1/6)*np.exp(-2*t) \
          + (1/6)*np.exp(-3*t) \
          - (1/30)*np.exp(-5*t)

# Analityczna odpowiedź impulsowa
f_imp_an = (1/3)*np.exp(-2*t) \
         - (1/2)*np.exp(-3*t) \
         + (1/6)*np.exp(-5*t)


# WYKRESY
plt.figure(figsize=(10, 8))

# Podwykres 1: Odpowiedź impulsowa
plt.subplot(2, 1, 1)
plt.plot(t, sol_imp_num, 'm', label="Rozwiązanie ODE")
plt.plot(t, f_imp_an, 'r--', label="Rozwiązanie analityczne")
plt.title("Odpowiedź impulsowa systemu")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid()
plt.legend()
plt.xlim(0, 5)


# Podwykres 2: Odpowiedź skokowa
plt.subplot(2, 1, 2)
plt.plot(t, sol_skok_num, 'g', label="Rozwiązanie ODE")
plt.plot(t, f_skok_an, 'k--', label="Rozwiązanie analityczne")
plt.title("Odpowiedź skokowa systemu")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid()
plt.legend()
plt.xlim(0, 5)


plt.tight_layout()
plt.show()