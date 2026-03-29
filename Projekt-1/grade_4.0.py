from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


# UKŁAD RÓWNAŃ (3 rząd)
# x1 = y
# x2 = y'
# x3 = y''

def dydt(u, t):
    x1, x2, x3 = u
    
    dx1dt = x2
    dx2dt = x3
    dx3dt = -10*x3 - 31*x2 - 30*x1 + np.exp(-5*t)
    
    return [dx1dt, dx2dt, dx3dt]


# CZAS
t = np.linspace(0, 10, 500)


# WARUNKI POCZĄTKOWE
y0 = 0      # y(0)
y0p = 0     # y'(0)
y0pp = 0    # y''(0)

u0 = [y0, y0p, y0pp]


# ROZWIĄZANIE NUMERYCZNE
solution = odeint(dydt, u0, t)
y_num = solution[:, 0]



# ROZWIĄZANIE ANALITYCZNE
y_an = (1/9)*np.exp(-2*t) \
     - (1/4)*np.exp(-3*t) \
     + (5/36)*np.exp(-5*t) \
     + (1/6)*t*np.exp(-5*t)



# WYKRES
plt.figure(figsize=(8,5))
plt.plot(t, y_num, label="Rozwiązanie numeryczne (odeint)")
plt.plot(t, y_an, '--', label="Rozwiązanie analityczne")

plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Porównanie rozwiązania numerycznego i analitycznego')
plt.grid()
plt.legend()

plt.xlim(0, 5) 

plt.show()