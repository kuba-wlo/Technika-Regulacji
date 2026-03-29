import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# UKŁAD RÓWNAŃ (3 rząd)
def dydt(X, t):
    y, yp, ypp = X   # y, y', y''
    
    dy_dt = yp
    dyp_dt = ypp
    dypp_dt = -10*ypp - 31*yp - 30*y + np.exp(-5*t)
    
    return [dy_dt, dyp_dt, dypp_dt]



# CZAS
t = np.linspace(0, 10, 500)


# WARUNKI POCZĄTKOWE
y0 = 0
y0p = 0
y0pp = 0

x0 = [y0, y0p, y0pp]



# ROZWIĄZANIE ANALITYCZNE
f = (1/9)*np.exp(-2*t) \
  - (1/4)*np.exp(-3*t) \
  + (5/36)*np.exp(-5*t) \
  + (1/6)*t*np.exp(-5*t)


# ROZWIĄZANIE NUMERYCZNE
solution = odeint(dydt, x0, t)

print(solution[:10])


# WYKRES
plt.plot(t, solution[:,0], label="Rozwiązanie ODEINT")
plt.plot(t, f, label="Rozwiązanie analityczne", linestyle='--')

plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid()
plt.legend()
plt.xlim(0, 6)

plt.show()