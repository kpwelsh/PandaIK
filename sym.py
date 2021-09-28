# %%
import sympy
from sympy.polys.specialpolys import f_polys
from sympy.simplify.fu import L
from sympy.solvers import solve_rational_inequalities, reduce_inequalities, solve
q,qp,dq,ddq,dddq,qr,dqr,ddqr,dddqr,dt,t = sympy.symbols('q,qp,dq,ddq,dddq,qr,dqr,ddqr,dddqr,dt,t')

# %%

dqp = (qp - q) / dt
ddqp = (dqp - dq) / dt
dddqp = (ddqp - ddq) / dt

ineqs = [
    qp <= qr,
    dqp <= dqr,
    ddqp <= ddqr,
    ddqp <= dddqr
]

qt = q + dqp * t + ddqp * t**2 / 2 + dddqp * t**3 / 6

qtprime = qt.diff(t)
for s in solve(qt.diff(t), t):
    extremum = qt.subs(t, s)
    ineqs.append(sympy.simplify(qt.subs(t, extremum) <= qr))

print(ineqs)
# %%
reduce_inequalities(ineqs, qp)

# %%
import numpy as np
import matplotlib.pyplot as plt

v_max = 2.61
a_max = 20
j_max = 12


# Velocity Distance
def velocity_stopping_distance(a0):
    return a0**2 / (2 * j_max)


a0 = -2
v0 = velocity_stopping_distance(a0)

t = np.linspace(0, 0.5, 50)
v = lambda t: v0 + a0 * t + j_max * t **2 / 2
t_min = (-a0 + np.sqrt(a0**2 - 2 * j_max * v0)) / (j_max)
a = lambda t: a0 + j_max * t
print(t_min)
print(v(t_min))
print(a(t_min))
plt.plot(t, v(t))
plt.plot(t, a(t))


# %%
from scipy.optimize import minimize_scalar

# qt = q + dq t + ddq t^2 / 2 + dddq t^3 / 6
# Want qr - qt to have 1 root

def descriminant(a,b,c,d):
    return (
        (4 * (b**2 - 3 * a * c)**3 - (2 * b**3 - 9 * a * b * c + 27 * a**2 * d)**2) / (27 * a **2)
    )

a = j_max / 6
b = -a_max / 2 / 2
c = -v_max + velocity_stopping_distance(-a_max / 2)
print(a, b, c)
#d = 50

A = (-27 * a**2)
B = (18 * a * b * c - 4 * b**3)
C = (b**2 * c**2 - 4 * a * c**3)

desc = B**2 - 4 * A * C

d = -(B + np.sqrt(desc)) / (2 * A)
print(f'Desc: {desc}, {A * d**2 + B * d + C}')

t = np.linspace(0, 5, 50)

qt = lambda t: a * t**3 + b * t**2 + c * t + d
vt = lambda t: 3 * a * t**2 + 2 * b * t + c
at = lambda t: 6 * a * t + 2 * b

t_min = minimize_scalar(qt, bounds = [0, np.inf])['x']
print(t_min)

print(qt(t_min))
print(vt(t_min))
print(at(t_min))

plt.plot(t, qt(t))

plt.plot(t, vt(t))
plt.plot(t, at(t))

# accel = np.clip(-20 + 12 * t, -20, 20)
# plt.plot(t, accel)
# vel = np.clip(-2.61 + np.cumsum(accel) * dt, -2.61, 2.61)
# plt.plot(t, vel)
# p = d + np.cumsum(vel) * dt
# plt.plot(t, p)

plt.show()



# %%


v_max = 2.61
a_max = 20
j_max = 12

t_mid = 1
t = np.linspace(0, 2 * t_mid, 50)

v0 = 0
a0 = -10
v = v0 + a0 * t + j_max * t**2 / 2
v5 = v0 + a0 * t_mid + j_max * (t_mid)**2 / 2
a5 = a0 + j_max * t_mid
v[t > t_mid] =  v5 + a5 * (t[t>t_mid]-t_mid) - j_max * (t[t>t_mid] - t_mid)**2 / 2

plt.plot(t, v)

# %%

v_max = 2.61
a_max = 20
j_max = 12

t_mid = 2
t = np.linspace(0, 5, 50)

q0 = 0
v0 = -5
a0 = -10
q = q0 + v0 * t + a0 * t**2 / 2 + j_max * t**3 / 6
qmid = q0 + v0 * (t_mid) + a0 * (t_mid)**2 / 2 + j_max * (t_mid)**3 / 6
vmid = v0 + a0 * t_mid + j_max * t_mid**2 / 2
amid = a0 + j_max * t_mid
q[t > t_mid] = qmid + vmid * (t[t>t_mid] - t_mid) + amid * (t[t>t_mid] - t_mid)**2 / 2 - j_max * (t[t>t_mid] - t_mid)**3 / 6

v = v0 + a0 * t + j_max * t**2 / 2
v5 = v0 + a0 * t_mid + j_max * (t_mid)**2 / 2
a5 = a0 + j_max * t_mid
v[t > t_mid] =  v5 + a5 * (t[t>t_mid]-t_mid) - j_max * (t[t>t_mid] - t_mid)**2 / 2

plt.plot(q, v)


# %%

v_max = 2.61
a_max = 20
j_max = 12

class SecondOrderController:
    def __init__(self, max_a, max_v):
        self.A = max_a
        self.V = max_v
        self.X0 = None
        self.V0 = None
        self.T1 = None
        self.T2 = None
        self.T3 = None
    
    def set_init(self, x, v):
        self.X0 = x
        self.V0 = v

    def set_goal(self, x, v):
        self.XG = x
        self.VG = v
    
    def update(self):
        dt1 = (self.V - self.V0) / self.A
        dt3 = self.V / self.A
        dt2 = (self.XG - (self.X0 + dt1 + dt3)) / self.V

        d = 1

        if dt2 < 0:
            v = np.sqrt(d * self.A * (self.XG - self.X0) + self.V0**2 / 2)
            dt2 = 0
            dt1 = (v - d * self.V0) / self.A
            dt3 = v / self.A
        
        self.T1 = dt1
        self.T2 = self.T1 + dt2
        self.T3 = self.T2 + dt3



def poly_bounds(a,b,c,d,t_min,t_max):
    f0 = a * t_min ** 3 + b * t_min**2 + c * t_min + d
    f1 = a * t_max ** 3 + b * t_max**2 + c * t_max + d

    extremes = []
    if a == 0:
        if b == 0:
            extremes.append(0)
        else:
            extremes.append(-c / (2*b))
    else:
        extremes.append((-2*b + np.sqrt((2*b)**2 - 4 * (3*a) * c)) / (2 * (3*a)))
        extremes.append((-2*b - np.sqrt((2*b)**2 - 4 * (3*a) * c)) / (2 * (3*a)))

    extreme_values = [f0, f1]
    for t in extremes:
        if t > t_min and t < t_max:
            extreme_values.append(a * t ** 3 + b * t**2 + c * t + d)
    return min(*extreme_values), max(*extreme_values)


def second_order_bounds(v0, a0):

    stopping_time = np.abs(a0 / j_max)
    stopping_v = v0 + stopping_time * a0 - np.sign(a0) * stopping_time ** 2 * j_max / 2

    d = np.sign(-stopping_v)

    dt1 = (a_max - a0) / j_max
    dv1 = d * j_max * dt1**2 / 2 + a0 * dt1
    dt3 = (a_max) / j_max
    dv3 = -d * j_max * dt3**2 / 2 + d * a_max * dt3
    dt2 = (-v0 - (dv1 + dv3)) / a_max


    if dt2 < 0:
        a = np.sqrt(d * j_max * (-v0) + 0.5 * a0**2)
        dt2 = 0
        dt1 = (a - d * a0) / j_max
        dt3 = a / j_max
    
    x1t = lambda t: v0 * t + a0 * t**2 / 2 + d * j_max * t**3 / 6
    v1t = lambda t: v0 + a0 * t + d * j_max * t**2 / 2
    a1t = lambda t: a0 + d * j_max * t

    x2t = lambda t: x1t(dt1) + v1t(dt1) * (t - dt1) + a1t(dt1) * (t - dt1) ** 2 / 2
    v2t = lambda t: v1t(dt1) + a1t(dt1) * (t - dt1)
    a2t = lambda t: a1t(dt1)

    x3t = lambda t: x2t(dt1 + dt2) + v2t(dt1 + dt2) * (t - (dt1 + dt2)) + a2t(dt1 + dt2) * (t - (dt1 + dt2)) ** 2 / 2 - d * j_max * (t - (dt1 + dt2))**3 / 6
    v3t = lambda t: v2t(dt1 + dt2) + a2t(dt1 + dt2) * (t - (dt1 + dt2)) - d * j_max * (t - (dt1 + dt2))**2 / 2
    a3t = lambda t: a2t(dt1 + dt2) - d * j_max * (t - (dt1 + dt2))

    def val(t):
        if t < dt1:
            return x1t(t), v1t(t), a1t(t)
        elif t < dt1 + dt2:
            return x2t(t), v2t(t), a2t(t)
        else:
            return x3t(t), v3t(t), a3t(t)

    T = np.linspace(0, dt1, 50)
    #plt.plot(T, [a1t(t) for t in T])

    bounds_x = list(zip(*[
        poly_bounds(d * j_max / 6, a0 / 2, v0, 0, 0, dt1),
        poly_bounds(-d * j_max / 6, a2t(dt1+dt2) / 2, v2t(dt1+dt2), x2t(dt1+dt2), 0, dt3),
    ]))
    bounds_v = list(zip(*[
        poly_bounds(0, d * j_max / 2, a0, v0, 0, dt1),
        poly_bounds(0, -d * j_max / 2, a2t(dt1+dt2), v2t(dt1+dt2), 0, dt3)
    ]))
    print(bounds_x)
    print(bounds_v)
    return val



v0 = -1
a0 = -5
d = 1


dt1 = (a_max - a0) / j_max
dv1 = d * j_max * dt1**2 / 2 + a0 * dt1
dt3 = (a_max) / j_max
dv3 = -d * j_max * dt3**2 / 2 + d * a_max * dt3
dt2 = (-v0 - (dv1 + dv3)) / a_max

if dt2 < 0:
    a = np.sqrt(d * j_max * (-v0) + 0.5 * a0**2)
    dt2 = 0
    dt1 = (a - d * a0) / j_max
    dt3 = a / j_max
print(dt1, dt2, dt3)


t = np.linspace(0, dt1 + dt2 + dt3, 500)

vt = np.zeros_like(t)
v1 = v0 + a0 * dt1 + j_max * dt1**2 / 2
a1 = a0 + j_max * dt1

v2 = v1 + a1 * dt2
a2 = a1

v3 = v2 + a2 * dt3 - j_max * dt3**2 / 2
a3 = a2 - j_max * dt3

jt = np.zeros_like(t)
jt[t < dt1] = d * j_max
jt[(t > dt1) & (t < (dt1+dt2))] = 0
jt[(t > (dt1+dt2))] = - d * j_max

val = second_order_bounds(v0, a0)

dt = t[1] - t[0]
vt = np.cumsum(np.cumsum(jt) * dt + a0) * dt + v0
# plt.plot(t, vt)
# plt.plot(t, np.cumsum(vt * dt))

# plt.plot(t, [val(_t)[0] for _t in t], linestyle = '--')
# plt.plot(t, [val(_t)[1] for _t in t], linestyle = '--')
plt.plot(np.cumsum(vt * dt), vt, linestyle = '--')

plt.ylim([-2.61,2.61])
plt.xlim([-10,10])





# %%
