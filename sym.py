# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# %%

v_max = 2.61
a_max = 7.5
j_max = 3750
dt = 1e-3
def max_distance(v0, a0):
    q = 0
    v = v0
    a = a0
    max_a = abs(a)
    max_v = abs(v)
    while v > 0 or a > 0:
        dq = max(-v_max * dt, v * dt - a_max * dt**2, v * dt + a * dt**2 - j_max * dt**3)
        q = q + dq
        a = (dq / dt - v) / dt
        v = dq / dt

        max_a = max(max_a, abs(a))
        max_v = max(max_v, abs(v))
    return q
V = np.linspace(0, v_max, 50)
for a0 in np.linspace(-20, 20, 5):
    y = [max_distance(v, a0) for v in V]
    plt.plot(y, V, label = str(a0))
    print(curve_fit(lambda x, a, b: a * np.sqrt(x) + b, y, V)[0])
#plt.legend()
#print(1.1/ (2 * a_max) - np.polyfit(V, y, 2)[0])
# %%
plt.plot(V, 0.025 * V**2)

# %%

v_max = 2.61
a_max = 20
j_max = 10000

print(a_max * dt - j_max * dt**2)

dt = 1e-3
def max_velocity(a0):
    v = 0
    a = a0
    while a > 0:
        dv = min(a * dt - j_max * dt**2, a**2 / (2 * j_max))
        v = v + dv
        a = dv / dt

    return v

A = np.linspace(0, a_max, 50)
y = [max_velocity(a) for a in A]
#plt.plot(A, y)
print(np.polyfit(A, y, 2))
c = (1 / (2 * j_max))
vl = -v_max
vr = v_max
v = np.linspace(vl, vr, 500)
def upper_bound(v):
    return (2 * c * v - dt**2 + np.sqrt((2 * c * v - dt**2)**2 + 4 * c * (dt**2 * vr - c * v**2))) / (2 * c)

def lower_bound(v):
    return (2 * c * v + dt**2 - np.sqrt((2 * c * v + dt**2)**2 + 4 * c * (-dt**2 * vl - c * v**2))) / (2 * c)


plt.plot(v, upper_bound(v) - v)
plt.plot(v, lower_bound(v) - v)

# %%

vs = [0]
ass = [0]
v = vs[-1]
a = ass[-1]

for i in range(2000):
    dv = min(a_max * dt, a * dt + j_max * dt**2, upper_bound(v) - v)
    v = v + dv
    a = dv/dt

    vs.append(v)
    ass.append(a)

plt.plot(vs)
plt.plot(ass)

print(np.max(np.abs(vs)))
print(np.max(np.abs(ass)))


# %%
c = 0.025
c = 1 / (2 * (a_max))
d = 1e-3
qr = np.pi
ql = -qr
q = np.linspace(ql, qr, 500)
v = np.linspace(-v_max, v_max, 500)
jm = j_max
def upper_bound(q):
    return q - a_max * dt**2 - d * dt + np.sqrt((q - a_max * dt**2 - d * dt) ** 2 - (q**2 - 2 * a_max * dt**2 * qr - d * dt * q))

def lower_bound(q):
    return q + a_max * dt**2 - d * dt - np.sqrt((q + a_max * dt**2 - d * dt) ** 2 - (q**2 - 2 * a_max * dt**2 * (-ql) - d * dt * q))


def upper_bound_v(v):
    return v - jm * dt**2 - d * dt + np.sqrt((v - jm * dt**2 - d * dt) ** 2 - (v**2 - 2 * jm * (dt**2) * v_max - d * dt * v))

def lower_bound_v(v):
    return v + jm * dt**2 - d * dt - np.sqrt((v + jm * dt**2 - d * dt) ** 2 - (v**2 - 2 * jm * (dt**2) * v_max - d * dt * v))


plt.plot(q, upper_bound(q) - q)
plt.plot(q, lower_bound(q) - q)
plt.figure()
plt.plot(v, upper_bound_v(v) - v)
plt.plot(v, lower_bound_v(v) - v)

# %%
qs = [0]
vs = [0]
ass = [0]
js = [0]
q = qs[-1]
v = vs[-1]
a = ass[-1]
for i in range(20000):

    uvb = upper_bound_v(v)
    lvb = lower_bound_v(v)

    max_dq = min(uvb * dt, v * dt + a_max * dt**2, v * dt + a * dt**2 + j_max * dt**3, upper_bound(q) - q)
    min_dq = max(lvb * dt, v * dt - a_max * dt**2, v * dt + a * dt**2 - j_max * dt**3, lower_bound(q) - q)

    if max_dq < min_dq:
        print(q, v, a)
        print(uvb, lvb)
        print(max_dq, min_dq)
    dq = 1 * (max_dq - min_dq) + min_dq
    q = q + dq
    a = (dq / dt - v) / dt
    v = dq / dt

    qs.append(q)
    vs.append(v)
    ass.append(a)
    js.append((ass[-1] - ass[-2]) / dt)

plt.plot(qs)
plt.plot(vs)
plt.plot(ass)
plt.plot(js)

print(np.max(np.abs(qs)))
print(np.max(np.abs(vs)))
print(np.max(np.abs(ass)))
print(np.max(np.abs(js)))

# %%
q_max = np.pi
v_max = 2.61
a_max = 20
j_max = 10000
dt = 1e-4

def max_v(dq, a):
    if abs(a) > 0:
        a = np.clip(a + j_max * (a / abs(a)) * dt, -a_max, a_max)
    jm = -j_max
    am = -a_max
    tA = (am - a) / jm
    A = 1
    B = -2 * am * tA + 2 * a * tA + jm * tA**2
    C = 2 * am * dq - 2 * am * (a * tA**2 / 2 + jm * tA**3 / 6) + a**2 * tA**2 + a * tA * jm * tA * tA + jm**2 * tA**4 / 4

    #print(A, B, C)
    vm = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    # qa = vm * tA + a * tA**2 / 2 + jm * tA**3 / 6

    # va = vm + a * tA + jm * tA**2 / 2
    # tB = -va / am
    # qb = qa + va * tB + am * tB**2 / 2 
    # print(qa, qb)
    return np.clip(vm, -v_max, v_max)

def max_a(dv):
    jm = -j_max
    return np.clip(np.sqrt(-2 * jm * dv) / 4, -a_max, a_max)

dq = np.linspace(0, np.pi, 500)

for a in np.linspace(-a_max, a_max, 4):
    plt.plot(dq, other_max_v(dq, a))
#plt.plot(dq, max_v(dq))

# %%

qs = [0]
vs = [0]
ass = [0]
js = [0]
q = qs[-1]
v = vs[-1]
a = ass[-1]
for i in range(50000):

    uvb = max_v(q_max - q, a)
    lvb = -max_v(q + q_max, a)

    uab = max_a(uvb - v)
    lab = -max_a(v - lvb)

    ujb = np.clip((uab - a) / dt, -j_max, j_max)

    ljb = np.clip((lab - a) / dt, -j_max, j_max)

    max_dq = min(uvb * dt, v * dt + uab * dt**2, v * dt + a * dt**2 + ujb * dt**3)
    min_dq = max(lvb * dt, v * dt + lab * dt**2, v * dt + a * dt**2 + ljb * dt**3)

    if max_dq < min_dq:
        print(q, v, a)
        print(uvb, lvb)
        print(uab, lab)
        print(ujb, ljb)
        print('max da: ', ljb * dt)
        print(max_dq, min_dq)
        print((uvb * dt, v * dt + uab * dt**2, v * dt + a * dt**2 + ujb * dt**3))
        print((lvb * dt, v * dt + lab * dt**2, v * dt + a * dt**2 + ljb * dt**3))
        break
    dq = (1) * (max_dq - min_dq) + min_dq
    q = q + dq
    a = (dq / dt - v) / dt
    v = dq / dt

    qs.append(q)
    vs.append(v)
    ass.append(a)
    js.append((ass[-1] - ass[-2]) / dt)

plt.plot(qs)
plt.plot(vs)
plt.plot(ass)
#plt.plot(js)

print(np.max(np.abs(qs)))
print(np.max(np.abs(vs)))
print(np.max(np.abs(ass)))
print(np.max(np.abs(js)))

# %%
from scipy.optimize import minimize_scalar, root_scalar

q_max = np.pi
v_max = 2.61
a_max = 7.5
j_max = 3750
dt = 1e-4


v0 = np.linspace(0, 2.61, 500)
a0 = a_max

def max_j_v(v0,a0):
    jm = -j_max
    A = -dt**2 / (2 * jm)
    B = dt**2 - a0 * dt / jm
    C = -(v_max - v0) + a0 * dt - a0**2 / (2 * jm)
    return np.clip((-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A), -j_max, j_max)


def max_tA_j(dq, v0, a0):
    am = -a_max
    jm = -j_max

    tA = am / jm - (np.clip(a0 + j_max * dt, -a_max, a_max)) / jm
    #tA = (am - a0) / jm + dt
    Vc = v0 + a0 * (dt + tA) + jm * tA**2 / 2
    A = -(dt**2 + dt * tA)**2 / (2 * am)
    B = (dt**3 + dt**2 * tA + dt * tA**2) - Vc * (dt**2 + dt * tA) / (am)
    C = -dq + jm * tA**3 / 6 + v0 * (dt + tA) + a0 * (dt**2 + dt * tA + tA**2 / 2) - Vc**2 / (2*am)
    return (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)

def max_j_q(dq,v0,a0):
    return (dq - v0 * dt - a0 * dt**2) / dt**3

def max_j_a(da):
    return (da-1e-5) / dt


def total_max_j(dq, v0, a0, da):
    dq = np.clip(dq, 0, np.inf)
    return min(max_j_q(dq, v0, a0), max_j_v(v0, a0), max_tA_j(dq, v0, a0), max_j_a(da))

dq = np.linspace(0, np.pi, 500)
#plt.plot(dq, np.clip(max_tA_j(dq, v0, a0), -j_max, j_max))
a0 = np.linspace(-0, 1e-4, 50)
plt.plot(a0, max_tA_j(0, 0, a0))


# %%
v_max = 2.61
a_max = 7.5
j_max = 3750


def stopping_distance(v0, a0):
    q = v0 * dt + a0 * dt**2 + j_max * dt**3
    v = v0 + a0 * dt + j_max * dt**2
    a = a0 + j_max * dt
    jm = -j_max
    am = -a_max

    tA = (am - a) / jm
    vA = v + a * tA + jm * tA**2 /2
    if vA > 0:
        qA = q + v * tA + a * tA**2 / 2 + jm * tA**3 / 6
        tV = - vA / am
        return qA + vA * tV + am * tV **2 / 2
    
 
    A = - 27 * (-jm / 6)**2
    B = 18 * (-jm / 6 * a / 2 * v) - 4 * (a / 2)**3
    C = (a / 2)**2 * v**2 - 4 * (jm / 6) * v**3

    desc = B**2 - 4 * A * C
    if desc < 0:
        return 0

    return (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A) - q

def stopping_v(a0):
    v = a0 * dt + j_max * dt**2
    a = a0 + j_max * dt
    jm = - j_max
    return v - a**2 / (2 * jm)


def q_bounds(q, v, a):

    
    suqb = q_max - 1e-4 - stopping_distance(v, a)
    slqb = -q_max + 1e-4 + stopping_distance(-v, -a)
    suvb = v_max - 1e-4 - stopping_v(a)
    slvb = -v_max + 1e-4 + stopping_v(-a)
    suab = a_max
    slab = -a_max
    sujb = max(a_max - a, 0) / dt
    sljb = min(-a_max - a, 0) / dt

    sub = min(suqb, q + suvb * dt, q + v * dt + suab * dt**2, q + v * dt + a * dt**2 + sujb * dt**3)
    slb = max(slqb, q + slvb * dt, q + v * dt + slab * dt**2, q + v * dt + a * dt**2 + sljb * dt**3)


    ujb = min((q_max - q - v * dt - a*dt**2) / dt**3, (max(v_max - v,0) - a * dt) / dt**2, max(a_max - a,0) / dt, j_max)
    ljb = max((-q_max + q + v * dt + a*dt**2) / dt**3, (min(-v_max - v,0) + a * dt) / dt**2, min(-a_max - a,0) / dt, -j_max)


    hub = q + v * dt + a * dt**2 + ujb * dt**3
    hlb = q + v * dt + a * dt**2 + ljb * dt**3
    sub = max(sub, hlb)
    slb = min(slb, hub)

    if sub < slb:
        mb = (sub + slb) / 2
        sub, slb = mb, mb

    if sub < hlb or slb > hub:
        print(hub, sub, hlb, slb)
    return min(hub, sub) - q, max(hlb, slb) - q
    



qs = [0]
vs = [0]
ass = [0]
js = [0]
q = qs[-1]
v = vs[-1]
a = ass[-1]
mj = []
for i in range(40000):

    # ujb = total_max_j(q_max - q, v, a, a_max - a)
    # ljb = -total_max_j(q + q_max, -v, -a, a + a_max)
    # mj.append(ujb)

    #max_dq = v * dt + a * dt**2 + ujb * dt**3
    #min_dq = v * dt + a * dt**2 + ljb * dt**3
    #max_dq = max(max_dq, min_dq)
    max_dq, min_dq = q_bounds(q, v, a)
    if max_dq < min_dq:
        print(q, v, a)
        print(max_dq, min_dq)
        break
    dq = (1) * (max_dq - min_dq) + min_dq
    q = q + dq
    a = (dq / dt - v) / dt
    v = dq / dt

    qs.append(q)
    vs.append(v)
    ass.append(a)
    js.append((ass[-1] - ass[-2]) / dt)

#plt.plot(mj)
plt.plot(qs)
plt.plot(vs)
plt.plot(ass)
#plt.plot(js)

print(q_max - np.max(np.abs(qs)))
print(np.max(np.abs(vs)))
print(np.max(np.abs(ass)))
print(np.max(np.abs(js)))
# %%




q_max = np.pi
v_max = 2.61
a_max = 7.5
j_max = 375
dt = 1e-4

def stopping_distance(v0,a0):
    q = v0 * dt + a0 * dt**2 + j_max * dt**3
    v = v0 + a0 * dt + j_max * dt**2
    a = a0 + j_max * dt
    jm = -j_max
    am = -a_max

    tA = (am - a) / jm
    vA = v + a * tA + jm * tA**2 /2
    if vA > 0:
        qA = q + v * tA + a * tA**2 / 2 + jm * tA**3 / 6
        tV = - vA / am
        return qA + vA * tV + am * tV **2 / 2
    
 
    A = - 27 * (-jm / 6)**2
    B = 18 * (-jm / 6 * a / 2 * v) - 4 * (a / 2)**3
    C = (a / 2)**2 * v**2 - 4 * (jm / 6) * v**3

    desc = B**2 - 4 * A * C
    if desc < 0:
        return 0

    return (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A) - q

def stopping_v(a0):
    v = a0 * dt + j_max * dt**2
    a = a0 + j_max * dt
    jm = - j_max
    return v - a**2 / (2 * jm)


for v0 in np.linspace(-v_max, v_max, 5):
    for a0 in np.linspace(-a_max, a_max, 5):
        print(v0, a0, stopping_distance(v0, a0))



# %%
