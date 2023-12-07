import math as mth
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
from scipy.optimize import bisect
from scipy.integrate import quad
from Graphic import graphics as graph

materialName = 'InSb'
Tmax = 400
Tmin = 70

Na1 = 1e10
Na2 = 1e12
Ea1 = 0.05
Ea2 = 0.1

Eg0 = 1.156
at  = 7.021e-4
bt = 1108
ε = 11.8
mn = 1.08
mp = 0.56

q = -1.6e-19
k = 8.62e-5
m0 = 9.11e-31

Xsame = 4.57
ε0 = 8.85e-14
workMe = 5.25 
tox = 10e-7
εOx = 3.725

aa  = 9.6
bb = 2.13
cc = 12/5

VBO = 4.4
CBO = 3.5  

T_range = np.arange(Tmin, Tmax + 1, 1)
η_range = np.arange(-10, 10, (20 / (T_range.size)))

T300 = 230

plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (9.6,5.4) 
plt.rcParams['axes.grid'] = True

#расчет Eg 
Eg = Eg0 - (at * (T_range**2)) / (bt + T_range)
# graph.print(T_range, Eg, title = 'Ширина запрещённой зоны')

#расчет Nc, расчет Nv 
Nc = 2.5e19 * mn**(3/2) * (T_range / 300)**(3/2)
Nv = 2.5e19 * mp**(3/2) * (T_range / 300)**(3/2)
# graph.print(T_range, Nc, T_range, Nv, title = 'Плотность квантовых состояний', log = True)

#расчет интеграла Ферми порядка 1/2 
F_12 = ((np.sqrt(np.pi/2))**(-1))*((((1.5*(2**1.5))/(bb+η_range+(((np.fabs(η_range-bb))**(1/cc))+aa)**1.5))+((np.exp(-η_range))/((np.sqrt(np.pi))/(2))))**(-1))
# graph.print(η_range, F_12, title = 'График проверки интеграла Ферми порядка 1/2', log = True)

# Зонная диаграмма
# Дно зоны проводимости, Потолок валентной зоны
Ec = Eg.copy()
Ev = np.zeros(T_range.size)

Ferm =  lambda η: (((np.sqrt(np.pi/2))**(-1))*((((1.5*(2**1.5))/(bb+η+(((np.fabs(η-bb))**(1/cc))+aa)**1.5))+((np.exp(-η))/((np.sqrt(np.pi))/(2))))**(-1)))
Ei = Eg / (k * T_range)

def f(η, i):
    p = Nv[i] * Ferm(-η - Ei[i])
    n = Nc[i] * Ferm(η)
    na1 = Na1 / (1 + 2 * np.exp(-η + (Ea1 - Eg[i]) / (k * T_range[i])))
    na2 = Na2 / (1 + 2 * np.exp(-η + (Ea2 - Eg[i]) / (k * T_range[i])))
    return p - n - na1 - na2

Fimax = Ec[330]/2 + 0.75 * k * T_range[330] * mth.log((mp/mn),(mth.exp(1)))
η_start = (Fimax - Eg[T_range.size - 1]) / (k * T_range[T_range.size - 1])
ETA = [1]
ETA[0] = sci.optimize.fsolve(f, η_start , args = (330), xtol = 1e-11)[0]
for i in range(1,T_range.size):
    ETA.append(sci.optimize.fsolve(f, ETA[i-1] , args = (330 - i), xtol=1e-11)[0])

# print(T_range[330],η_start,Fimax)
ETAn = np.array(ETA [::-1])
# print(ETA[0], ETAn[330])

#Собственный уровенль Ферми
Es = Eg/2 + 0.75 * k * T_range * mth.log((mp/mn), (mth.exp(1)))
#Уровень ферми
F = k * ETAn * T_range + Eg

# graph.print(T_range, Ev, T_range, Ec, T_range, Es, T_range, [Ea1] * len(T_range), T_range, [Ea2] * len(T_range), T_range, F, title = 'Зонная диаграмма')

#Расчет концентраций электронов, дырок и собственной концентрации
p = Nv * Ferm(-ETAn - (Eg/(k * T_range)))
n = Nc * Ferm(ETAn)
ni = np.sqrt(n * p)

# graph.print(1/T_range, p, 1/T_range, n, 1/T_range, ni, title = 'Концентрация', log = True)
#Расчет концентраций ионов акцепторов
na1 = Na1 / (1 + 2 * np.exp(-ETAn + (Ea1 - Eg)/(k * T_range)))
na2 = Na2 / (1 + 2 * np.exp(-ETAn + (Ea2 - Eg)/(k * T_range)))

# graph.print(1/T_range, na1, 1/T_range, na2, title = 'Концентрации ионов акцепторов', log = True)

ff = np.fabs(p - n - na1 - na2)
ff[ff == 0] = 1e-5

# graph.print(T_range, ff, title = 'электронейтральность', log = True)

print('{---------------------------------Вторая часть------------------------------------------}')

#Расчет положения уровня Ферми от зарядового уравнения
FermiLevelQeq = k * T_range[T300] * np.log(Nv[T300]/(na1[T300]+na2[T300])) + Ev[T300]
#Расчет положения уровня Ферми от потенциала Ферми
φF = k * T_range[T300] * np.log(((na1[T300] + na2[T300]) / ni[T300]))

# print(F[T300],' рассчитанынй уровень Ферми при 300K')
# print(Es[T300] - q * φF,' уровень Ферми от Потенциала Ферми при 300K')
# print(FermiLevelQeq,' уровень Ферми от основных носителей зарядов при 300K')
#Работа выхода полупроводника
WorkOut = Xsame + Eg[T300] - F[T300]
print(WorkOut,' работа выхода полупроводника при 300K')
#Расчет напряжения плоских зон
Vfb = workMe - WorkOut
# print(Vfb,' напряжение плоских зон полупроводника при 300K')

#Расчет зависимости удельного поверхностного заряда от поверхностного потенциала
f_Ψs = lambda Ψs: 0.026 * p[T300] * (np.exp(-Ψs/0.026) - 1) + 0.026 * n[T300] * (np.exp(Ψs/0.026) - 1) + Ψs * (Na1 + Na2)
Qs_Ψs = lambda Ψs: (-np.sign(Ψs) * ((2 * -q * ε * ε0 * f_Ψs(Ψs)) ** (1 / 2)))

Ψs_range = np.arange(-0.3, 0.3, 2 * (0.3 / T_range.size))
Qs = Qs_Ψs(Ψs_range)

# graph.print(Ψs_range, Qs, title = 'Поверхностный заряд')

#Расчет зависимости поверхностного потенциала ψs от напряжения затвор—подложка Vgb, используя численные методы.
Coxp = εOx * ε0 / tox
SPE = lambda Ψs, Vgb: Vfb + Ψs - Qs_Ψs(Ψs) / Coxp - Vgb
SPE_Ev = SPE(Ev[T300] - F[T300], 0)
SPE_Ec = SPE(Ec[T300] - F[T300], 0)

Vgb = np.linspace(SPE_Ev, SPE_Ec, 331)

Ψs = np.array([])
for val in Vgb:
    Ψs = np.hstack((Ψs, bisect(f = SPE, a = Ev[T300] - F[T300], b = Ec[T300] - F[T300], args = val, xtol = 1e-12)))

# plt.xlabel('Vgb, В')
# plt.ylabel('Ψs, эВ')
# plt.title('Поверхностный потенциал')
# plt.plot(Vgb, Ψs, label = 'Vgb');
# plt.axvline(x = Vfb, color = 'red', label = 'Vfb')
# plt.axvline(x = SPE_Ev, color = 'green', label = 'SPE(Ev-F)')
# plt.axvline(x = SPE_Ec, color = 'yellow', label = 'SPE(Ec-F)')
# plt.show()

Qnew = (Ψs + Vfb - Vgb) * Coxp

# plt.xlabel('Vgb, В')
# plt.ylabel('Qs, Кл/см^2')
# plt.title('Поверхностный заряд')
# plt.plot(Vgb, Qnew, label = 'Qs')
# plt.axvline(x = Vfb, color='red', label = 'Vfb')
# plt.axvline(x = SPE_Ev, color = 'green', label = 'SPE(Ev-F)')
# plt.axvline(x = SPE_Ec, color = 'yellow', label = 'SPE(Ec-F)')
# plt.show()

print('{---------------------------------Третья часть------------------------------------------}')

qpos = -q

#Пункт 1
n_Ψs = lambda Ψs: n[T300] * np.exp(Ψs / 0.026)
p_Ψs = lambda Ψs: p[T300] * np.exp(-Ψs / 0.026)
E_Ψs = lambda Ψs: np.sign(Ψs) * np.sqrt(2 * (qpos) * f_Ψs(Ψs)/ (ε*ε0))

I_Ψs = lambda Ψs: 1 - np.abs(np.sign(Ψs))
Sp = lambda f, ep: (f + np.sqrt(f**2 + 4 * ep**2)) / 2
Sn = lambda f, ep: (f - np.sqrt(f**2 + 4 * ep**2)) / 2

integr_inv =  lambda Ψs: (qpos * n[T300] * (np.exp(Ψs / 0.026) - 1))/((np.sign(Ψs) * np.sqrt(2 * qpos * f_Ψs(Ψs) / (ε * ε0))) + I_Ψs(Ψs))
integr_bulk =  lambda Ψs: (-qpos * p[T300] * (np.exp(-Ψs / 0.026) - 1))/((np.sign(Ψs) * np.sqrt(2 * qpos * f_Ψs(Ψs) / (ε * ε0))) + I_Ψs(Ψs))

Qi = np.zeros(331) 
Qb = np.zeros(331)

for i in range(331):
    result_i, error = quad(integr_inv, Ψs[i], 0)
    Qi[i] = Sn(result_i, 1e-11)

for i in range(331):
    result_i, error = quad(integr_bulk, Ψs[i], 0)
    Qb[i] = result_i

# plt.xlabel('Vgb, В')
# plt.ylabel('Qi, Кл')
# plt.title('Заряд в инверсном слое')
# plt.plot(Vgb, np.abs(Qi))
# plt.axvline(x = Vfb, color='red', label = 'Vfb')
# plt.semilogy()
# plt.legend()
# plt.show()

# plt.xlabel('Vgb, В')
# plt.ylabel('Qb, Кл')
# plt.title('Заряд в Толщине полупроводника')
# plt.plot(Vgb, np.abs(Qb))
# plt.axvline(x = Vfb, color = 'red', label = 'Vfb')
# plt.semilogy()
# plt.legend()
# plt.show()

# plt.xlabel('Vgb, В')
# plt.ylabel('Qs, Кл/см^2')
# plt.title('Поверхностный заряд')
# plt.plot(Vgb, Qb + Qi, 'x',  label = 'Qb+Qi - расчет численными методами')
# plt.plot(Vgb, Qnew, label = 'Qs - расчет из второй части РЗ')
# plt.axvline(x = Vfb, color = 'red', label = 'Vfb')
# plt.legend()
# plt.show()

#Пункт 2
fA_Ψs = lambda Ψs: 0.026 * p[T300] * (np.exp(-Ψs / 0.026) - 1) + Ψs * (Na1 + Na2)
QbA_Ψs = lambda Ψs: -np.sign(Ψs) * np.sqrt(2 * qpos * (ε * ε0) * fA_Ψs(Ψs))
QiA_Ψs = lambda Ψs: 2 * qpos * (ε * ε0) * 0.026 * n[T300] * (np.exp(Ψs / 0.026) - 1) /(Qnew + QbA_Ψs(Ψs) + I_Ψs(Ψs))

QiA = QiA_Ψs(Ψs)
QbA = QbA_Ψs(Ψs)

# plt.xlabel('Vgb, В')
# plt.ylabel('QiA, Кл/см^2')
# plt.title('Заряд в инверсном аналитически')
# plt.plot(Vgb, np.abs(QiA), '*', label = 'QiA - аналитически')
# plt.plot(Vgb, np.abs(Qi),'-', color = 'r', label = 'Qi - численно')
# plt.legend()
# plt.semilogy()
# plt.axvline(x = Vfb, color = 'green', label = 'Vfb')
# plt.show()

# plt.xlabel('Vgb, В')
# plt.ylabel('QbA, Кл/см^2')
# plt.title('Заряд в толще полупроводника аналитически')
# plt.plot(Vgb, np.abs(QbA), '*', label = 'QbA - аналитически')
# plt.plot(Vgb, np.abs(Qb), '-', color = 'r', label = 'Qb - численно')
# plt.axvline(x = Vfb, color = 'green', label = 'Vfb')
# plt.semilogy()
# plt.legend()
# plt.show()

# plt.xlabel('Vgb, В')
# plt.ylabel('QbA, Кл/см^2')
# plt.title('Поверхностый заряд аналитически')
# plt.plot(Vgb, QiA + QbA, '*', label = 'QiA')
# plt.plot(Vgb, Qnew, label = 'Qs')
# plt.axvline(x = Vfb,color = 'green', label = 'Vfb')
# plt.legend()
# plt.show()

# plt.xlabel('Vgb, В')
# plt.ylabel('Ошибка, %')
# plt.title('Заряд в толще проводника')
# plt.plot(Vgb, np.abs(100 * ((QbA_Ψs(Ψs) - Qb) / (QbA_Ψs(Ψs)))))
# plt.axvline(x = Vfb, color = 'green', label = 'Vfb')
# plt.semilogy()
# plt.legend()
# plt.show()

# plt.xlabel('Vgb, В')
# plt.ylabel('Ошибка, %')
# plt.title('Заряд в инверсном слое проводника')
# plt.plot(Vgb, np.abs(100 * (QiA_Ψs(Ψs) - Qi) / (QiA_Ψs(Ψs) + 1e-12)))
# plt.axvline(x = Vfb, color = 'green', label = 'Vfb')
# plt.legend()
# plt.semilogy()
# plt.show()

# Пункт 3

# Оценка потенциалов слабой и сильной инверсии
φ_weak = 0.026 * mth.log(ni[T300] / n[T300])
φ_strong = 0.026 * mth.log((Na1 + Na2) / n[T300])
print(φ_weak, ' для слабой инверсии и ', φ_strong, ' для сильной инверсии')

Cs0 = np.sqrt(((qpos * ε * ε0) / (k * 300)) * (p[T300] + n[T300]))
df = lambda Ψs: (Na1 + Na2) - p[T300] * np.exp(-Ψs / 0.026) + n[T300] * np.exp(Ψs / 0.026)
Cs = lambda Ψs: np.sign(Ψs) * df(Ψs) * np.sqrt((qpos * ε * ε0) / (2 * f_Ψs(Ψs) + I_Ψs(Ψs))) + Cs0
Cgb = Cs(Ψs) * Coxp / (Cs(Ψs) + Coxp)

Cgb0 = np.zeros(len(Vgb))
for i in range(Vgb.size):
    Cgb0[i] = (Cs0 * Coxp) / (Cs0 + Coxp)

# plt.xlabel('Vgb, В')
# plt.ylabel('Cgb/Cox')
# plt.title('Емкость')
# plt.plot(Vgb, Cgb / Coxp, label = 'Cgb/Coxp')
# plt.plot(Vgb, Cgb0 / Coxp, label = 'Cgb0/Coxp,')
# plt.axvline(x = Vfb, color = 'red', label = 'Vfb')
# plt.legend()
# plt.show()

#Пункт 4
Ci = lambda Ψs: qpos * (n_Ψs(Ψs) - n[T300]) / E_Ψs(Ψs)
Cb = lambda Ψs: -qpos * (p_Ψs(Ψs) - p[T300]) / E_Ψs(Ψs)

Cb0 = np.sign(Ψs) * np.sqrt((qpos * ε * ε0 * p[T300]) * (0.026))
Ci0 = np.sign(Ψs) * np.sqrt((qpos * ε * ε0 * n[T300]) * (0.026))

# plt.xlabel('Vgb, В')
# plt.ylabel('C, Ф/см^2')
# plt.title('Емкости')
# plt.plot(Vgb, Ci(Ψs) + Cb(Ψs), 'x', label = 'Cs')
# plt.plot(Vgb, Ci(Ψs), label = 'Ci')
# plt.plot(Vgb, Cb(Ψs), label = 'Cb')
# plt.axvline(x = Vfb, color = 'red', label = 'Vfb')
# plt.legend()
# plt.show()

print('{---------------------------------Четвертая часть------------------------------------------}')

#из методических указаний 1 к КМ-4 - выбор координат
integral = lambda Ψs: 1 / E_Ψs(Ψs)

def YCoordCompute(Ψs):
    if Ψs == 0:
        y = np.linspace(0, 150, 101) * 1e-7 
        fis = 0 * y
        return y, Ψs 
    fis1 = np.linspace(Ψs, Ψs * 0.5, 21) 
    fis2 = np.logspace(np.log10(np.abs(Ψs * 0.5)), np.log10(np.abs(Ψs * 1e-3)), 101)
    if Ψs < 0:
        fis2 = -1 * fis2
    fis = np.hstack((fis1, fis2[1:]))
    y = np.array([])
    for value in fis:
        YTemporal, error = quad(integral, value, Ψs)
        y = np.hstack((y, YTemporal))
    return y, fis

newFis = (Ev[T300] - F[T300]) 
Vgb_new=SPE(newFis,0)
FisOx = Vgb_new - newFis - Vfb
y, newFisExtra = YCoordCompute(newFis)
y = y / 100 * 1e9
toxnm = tox / 100 * 1e9

nNew = n[T300] * np.exp(newFisExtra / 0.026)
pNnew = p[T300] * np.exp(-newFisExtra / 0.026)

QEqZero = pNnew - nNew - Na1 - Na2

# для зонной диаграммы 
fig, ax = plt.subplots() 
plt.figure(1)
ax.set_title('Зонная диаграмма(обогащение)')
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('Энергия , эВ', fontsize=14)
plt.grid(True)

# зонная диаграмма п\п
plt.plot(y, Ev[T300] - newFisExtra, 'b', label = "Ev")
plt.plot(y, Es[T300] - newFisExtra, 'g--', label = "Ei")
plt.plot(y, Ec[T300] - newFisExtra, 'orange', label = "Ec")

# уровень ферми
plt.plot(y, 0 * y + F[T300], 'k',  label = "F")
plt.legend(loc='best', prop={'size': 16})

#диаграмма оксида
plt.plot([0, 0], [Ev[T300] - newFis - VBO, Ec[T300] - newFis + CBO], 'r')
plt.plot([-toxnm, -toxnm],[Ev[T300] - newFis - VBO - FisOx, Ec[T300] - newFis + CBO - FisOx],  'r')
plt.plot([-toxnm, 0], [Ev[T300] - newFis - VBO - FisOx, Ev[T300] - newFis - VBO], 'r')
plt.plot([-toxnm, 0], [Ec[T300] - newFis + CBO - FisOx, Ec[T300] - newFis + CBO], 'r')
plt.legend(loc='best', prop={'size': 16})

# уровень ферми метала
plt.plot([-toxnm - 100, -toxnm], [F[T300] - Vfb - newFis - FisOx, F[T300] - Vfb - newFis - FisOx], 'k')

fig, ax = plt.subplots() 
plt.figure(2)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('(1 / см$^3$)', fontsize=14)
plt.grid(True)
ax.set_title('Концентрации(обогащение)')
plt.semilogy(y, nNew, label='n', linewidth=2)
plt.semilogy(y, pNnew, label='p', linewidth=2)
plt.semilogy(y, Na1+Na2 + 0 * y, label='Na')
plt.legend(loc='best', prop={'size': 16})

#обемный заряд
fig, ax = plt.subplots() 
ax.set_title('Объемный заряд(обогащение)')
plt.figure(3)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('$|\\rho/q|$ (1 / см$^3$)', fontsize=16)
plt.grid(True)
tmp = plt.semilogy(y, np.abs(QEqZero), label='$\\rho/q$')

plt.show()

newFis = 0
Vgb_new=SPE(newFis,0)
FisOx = Vgb_new - newFis - Vfb
y, newFisExtra = YCoordCompute(newFis)
y = y / 100 * 1e9
toxnm = tox / 100 * 1e9

newFisExtra = np.array([]);
newFisExtra = np.zeros(101)

nNew = n[T300] * np.exp(newFisExtra / 0.026)
pNnew = p[T300] * np.exp(-newFisExtra / 0.026)

QEqZero = pNnew - nNew - Na1 - Na2

# для зонной диаграммы 
fig, ax = plt.subplots() 
plt.figure(1)
ax.set_title('Зонная диаграмма(плоские зоны)')
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('Энергия , эВ', fontsize=14)
plt.grid(True)

# зонная диаграмма п\п
plt.plot(y, Ev[T300] - newFisExtra, 'b', label = "Ev")
plt.plot(y, Es[T300] - newFisExtra, 'g--', label = "Ei")
plt.plot(y, Ec[T300] - newFisExtra, 'orange', label = "Ec")

# уровень ферми
plt.plot(y, 0 * y + F[T300], 'k',  label = "F")
plt.legend(loc='best', prop={'size': 16})

#диаграмма оксида
plt.plot([0, 0], [Ev[T300] - newFis - VBO, Ec[T300] - newFis + CBO], 'r')
plt.plot([-toxnm, -toxnm],[Ev[T300] - newFis - VBO - FisOx, Ec[T300] - newFis + CBO - FisOx],  'r')
plt.plot([-toxnm, 0], [Ev[T300] - newFis - VBO - FisOx, Ev[T300] - newFis - VBO], 'r')
plt.plot([-toxnm, 0], [Ec[T300] - newFis + CBO - FisOx, Ec[T300] - newFis + CBO], 'r')
plt.legend(loc='best', prop={'size': 16})

# уровень ферми метала
plt.plot([-toxnm - 10, -toxnm], [F[T300] - Vfb - newFis - FisOx, F[T300] - Vfb - newFis - FisOx], 'k')

fig, ax = plt.subplots() 
plt.figure(2)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('(1 / см$^3$)', fontsize=14)
plt.grid(True)
ax.set_title('Концентрации(плоские зоны)')
plt.semilogy(y, nNew, label='n', linewidth=2)
plt.semilogy(y, pNnew, label='p', linewidth=2)
plt.semilogy(y, Na1+Na2 + 0 * y, label='Na')
plt.legend(loc='best', prop={'size': 16})

#обемный заряд
fig, ax = plt.subplots() 
ax.set_title('Объемный заряд(плоские зоны)')
plt.figure(3)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('$|\\rho/q|$ (1 / см$^3$)', fontsize=16)
plt.grid(True)
tmp = plt.semilogy(y, np.abs(QEqZero), label='$\\rho/q$')

plt.legend()
plt.show()

# psis = newFis    psi = newFisExtra
newFis = Es[T300] - F[T300]  
Vgb_new=SPE(newFis,0)
FisOx = Vgb_new - newFis - Vfb
y, newFisExtra = YCoordCompute(newFis)
y = y / 100 * 1e9
toxnm = tox / 100 * 1e9

nNew = n[T300] * np.exp(newFisExtra / 0.026)
pNnew = p[T300] * np.exp(-newFisExtra / 0.026)

QEqZero = pNnew - nNew - Na1 - Na2

# для зонной диаграммы 
fig, ax = plt.subplots() 
plt.figure(1)
ax.set_title('Зонная диаграмма(слабая инверсия)')
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('Энергия , эВ', fontsize=14)
plt.grid(True)

# зонная диаграмма п\п
plt.plot(y, Ev[T300] - newFisExtra, 'b', label = "Ev")
plt.plot(y, Es[T300] - newFisExtra, 'g--', label = "Ei")
plt.plot(y, Ec[T300] - newFisExtra, 'orange', label = "Ec")

# уровень ферми
plt.plot(y, 0 * y + F[T300], 'k',  label = "F")
plt.legend(loc='best', prop={'size': 16})

#диаграмма оксида
plt.plot([0, 0], [Ev[T300] - newFis - VBO, Ec[T300] - newFis + CBO], 'r')
plt.plot([-toxnm, -toxnm],[Ev[T300] - newFis - VBO - FisOx, Ec[T300] - newFis + CBO - FisOx],  'r')
plt.plot([-toxnm, 0], [Ev[T300] - newFis - VBO - FisOx, Ev[T300] - newFis - VBO], 'r')
plt.plot([-toxnm, 0], [Ec[T300] - newFis + CBO - FisOx, Ec[T300] - newFis + CBO], 'r')
plt.legend(loc='best', prop={'size': 16})

# уровень ферми метала
plt.plot([-toxnm - 10, -toxnm], [F[T300] - Vfb - newFis - FisOx, F[T300] - Vfb - newFis - FisOx], 'k')

fig, ax = plt.subplots() 
plt.figure(2)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('(1 / см$^3$)', fontsize=14)
plt.grid(True)
ax.set_title('Концентрации(слабая инверсия)')
plt.semilogy(y, nNew, label='n', linewidth=2)
plt.semilogy(y, pNnew, label='p', linewidth=2)
plt.semilogy(y, Na1+Na2 + 0 * y, label='Na')
plt.legend(loc='best', prop={'size': 16})

#обемный заряд
fig, ax = plt.subplots() 
ax.set_title('Объемный заряд(слабая инверсия)')
plt.figure(3)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('$|\\rho/q|$ (1 / см$^3$)', fontsize=16)
plt.grid(True)
tmp = plt.semilogy(y, np.abs(QEqZero), label='$\\rho/q$')

plt.legend();

# psis = newFis    psi = newFisExtra
newFis = (Es[T300] - F[T300])*2  
Vgb_new=SPE(newFis,0)
FisOx = Vgb_new - newFis - Vfb
y, newFisExtra = YCoordCompute(newFis)
y = y / 100 * 1e9
toxnm = tox / 100 * 1e9

nNew = n[T300] * np.exp(newFisExtra / 0.026)
pNnew = p[T300] * np.exp(-newFisExtra / 0.026)

QEqZero = pNnew - nNew - Na1 - Na2

# для зонной диаграммы 
fig, ax = plt.subplots() 
plt.figure(1)
ax.set_title('Зонная диаграмма(сильная инверсия)')
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('Энергия , эВ', fontsize=14)
plt.grid(True)

# зонная диаграмма п\п
plt.plot(y, Ev[T300] - newFisExtra, 'b', label = "Ev")
plt.plot(y, Es[T300] - newFisExtra, 'g--', label = "Ei")
plt.plot(y, Ec[T300] - newFisExtra, 'orange', label = "Ec")

# уровень ферми
plt.plot(y, 0 * y + F[T300], 'k',  label = "F")
plt.legend(loc='best', prop={'size': 16})

#диаграмма оксида
plt.plot([0, 0], [Ev[T300] - newFis - VBO, Ec[T300] - newFis + CBO], 'r')
plt.plot([-toxnm, -toxnm],[Ev[T300] - newFis - VBO - FisOx, Ec[T300] - newFis + CBO - FisOx],  'r')
plt.plot([-toxnm, 0], [Ev[T300] - newFis - VBO - FisOx, Ev[T300] - newFis - VBO], 'r')
plt.plot([-toxnm, 0], [Ec[T300] - newFis + CBO - FisOx, Ec[T300] - newFis + CBO], 'r')
plt.legend(loc='best', prop={'size': 16})

# уровень ферми метала
plt.plot([-toxnm - 10, -toxnm], [F[T300] - Vfb - newFis - FisOx, F[T300] - Vfb - newFis - FisOx], 'k')

fig, ax = plt.subplots() 
plt.figure(2)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('(1 / см$^3$)', fontsize=14)
plt.grid(True)
ax.set_title('Концентрации(сильная инверсия)')
plt.semilogy(y, nNew, label='n', linewidth=2)
plt.semilogy(y, pNnew, label='p', linewidth=2)
plt.semilogy(y, Na1+Na2 + 0 * y, label='Na')
plt.legend(loc='best', prop={'size': 16})

#обемный заряд
fig, ax = plt.subplots() 
ax.set_title('Объемный заряд(сильная инверсия)')
plt.figure(3)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('$|\\rho/q|$ (1 / см$^3$)', fontsize=16)
plt.grid(True)
tmp = plt.semilogy(y, np.abs(QEqZero), label='$\\rho/q$')

plt.legend();

# psis = newFis    psi = newFisExtra
newFis = (Es[T300] - F[T300])*2 + 3*0.026 
Vgb_new=SPE(newFis,0)
FisOx = Vgb_new - newFis - Vfb
y, newFisExtra = YCoordCompute(newFis)
y = y / 100 * 1e9
toxnm = tox / 100 * 1e9

nNew = n[T300] * np.exp(newFisExtra / 0.026)
pNnew = p[T300] * np.exp(-newFisExtra / 0.026)

QEqZero = pNnew - nNew - Na1 - Na2

# для зонной диаграммы 
fig, ax = plt.subplots() 
plt.figure(1)
ax.set_title('Зонная диаграмма(более сильная инверсия)')
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('Энергия , эВ', fontsize=14)
plt.grid(True)

# зонная диаграмма п\п
plt.plot(y, Ev[T300] - newFisExtra, 'b', label = "Ev")
plt.plot(y, Es[T300] - newFisExtra, 'g--', label = "Ei")
plt.plot(y, Ec[T300] - newFisExtra, 'orange', label = "Ec")

# уровень ферми
plt.plot(y, 0 * y + F[T300], 'k',  label = "F")
plt.legend(loc='best', prop={'size': 16})

#диаграмма оксида
plt.plot([0, 0], [Ev[T300] - newFis - VBO, Ec[T300] - newFis + CBO], 'r')
plt.plot([-toxnm, -toxnm],[Ev[T300] - newFis - VBO - FisOx, Ec[T300] - newFis + CBO - FisOx],  'r')
plt.plot([-toxnm, 0], [Ev[T300] - newFis - VBO - FisOx, Ev[T300] - newFis - VBO], 'r')
plt.plot([-toxnm, 0], [Ec[T300] - newFis + CBO - FisOx, Ec[T300] - newFis + CBO], 'r')
plt.legend(loc='best', prop={'size': 16})

# уровень ферми метала
plt.plot([-toxnm - 10, -toxnm], [F[T300] - Vfb - newFis - FisOx, F[T300] - Vfb - newFis - FisOx], 'k')

fig, ax = plt.subplots() 
plt.figure(2)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('(1 / см$^3$)', fontsize=14)
plt.grid(True)
ax.set_title('Концентрации(более сильная инверсия)')
plt.semilogy(y, nNew, label='n', linewidth=2)
plt.semilogy(y, pNnew, label='p', linewidth=2)
plt.semilogy(y, Na1+Na2 + 0 * y, label='Na')
plt.legend(loc='best', prop={'size': 16})

#обемный заряд
fig, ax = plt.subplots() 
ax.set_title('Объемный заряд(более сильная инверсия)')
plt.figure(3)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('$|\\rho/q|$ (1 / см$^3$)', fontsize=16)
plt.grid(True)
tmp = plt.semilogy(y, np.abs(QEqZero), label='$\\rho/q$')

plt.legend();

# psis = newFis    psi = newFisExtra
newFis = Ec[T300] - F[T300]
Vgb_new=SPE(newFis,0)
FisOx = Vgb_new - newFis - Vfb
y, newFisExtra = YCoordCompute(newFis)
y = y / 100 * 1e9
toxnm = tox / 100 * 1e9

nNew = n[T300] * np.exp(newFisExtra / 0.026)
pNnew = p[T300] * np.exp(-newFisExtra / 0.026)

QEqZero = pNnew - nNew - Na1 - Na2

# для зонной диаграммы 
fig, ax = plt.subplots() 
plt.figure(1)
ax.set_title('Зонная диаграмма(очень сильная инверсия)')
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('Энергия , эВ', fontsize=14)
plt.grid(True)

# зонная диаграмма п\п
plt.plot(y, Ev[T300] - newFisExtra, 'b', label = "Ev")
plt.plot(y, Es[T300] - newFisExtra, 'g--', label = "Ei")
plt.plot(y, Ec[T300] - newFisExtra, 'orange', label = "Ec")

# уровень ферми
plt.plot(y, 0 * y + F[T300], 'k',  label = "F")
plt.legend(loc='best', prop={'size': 16})

#диаграмма оксида
plt.plot([0, 0], [Ev[T300] - newFis - VBO, Ec[T300] - newFis + CBO], 'r')
plt.plot([-toxnm, -toxnm],[Ev[T300] - newFis - VBO - FisOx, Ec[T300] - newFis + CBO - FisOx],  'r')
plt.plot([-toxnm, 0], [Ev[T300] - newFis - VBO - FisOx, Ev[T300] - newFis - VBO], 'r')
plt.plot([-toxnm, 0], [Ec[T300] - newFis + CBO - FisOx, Ec[T300] - newFis + CBO], 'r')
plt.legend(loc='best', prop={'size': 16})

# уровень ферми метала
plt.plot([-toxnm - 1000, -toxnm], [F[T300] - Vfb - newFis - FisOx, F[T300] - Vfb - newFis - FisOx], 'k')

fig, ax = plt.subplots() 
plt.figure(2)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('(1 / см$^3$)', fontsize=14)
plt.grid(True)
ax.set_title('Концентрации(очень сильная инверсия)')
plt.semilogy(y, nNew, label='n', linewidth=2)
plt.semilogy(y, pNnew, label='p', linewidth=2)
plt.semilogy(y, Na1+Na2 + 0 * y, label='Na')
plt.legend(loc='best', prop={'size': 16})

#обемный заряд
fig, ax = plt.subplots() 
ax.set_title('Объемный заряд(очень сильная инверсия)')
plt.figure(3)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('$|\\rho/q|$ (1 / см$^3$)', fontsize=16)
plt.grid(True)
tmp = plt.semilogy(y, np.abs(QEqZero), label='$\\rho/q$')

plt.legend();

newFis = bisect(SPE, Ev[T300] - F[T300], Ec[T300] - F[T300],0)
Vgb_new=SPE(newFis,0)
FisOx = Vgb_new - newFis - Vfb
y, newFisExtra = YCoordCompute(newFis)
y = y / 100 * 1e9
toxnm = tox / 100 * 1e9

nNew = n[T300] * np.exp(newFisExtra / 0.026)
pNnew = p[T300] * np.exp(-newFisExtra / 0.026)

QEqZero = pNnew - nNew - Na1 - Na2

# для зонной диаграммы 
fig, ax = plt.subplots() 
plt.figure(1)
ax.set_title('Зонная диаграмма(нулевое смещение затвор-подложка)')
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('Энергия , эВ', fontsize=14)
plt.grid(True)

# зонная диаграмма п\п
plt.plot(y, Ev[T300] - newFisExtra, 'b', label = "Ev")
plt.plot(y, Es[T300] - newFisExtra, 'g--', label = "Ei")
plt.plot(y, Ec[T300] - newFisExtra, 'orange', label = "Ec")

# уровень ферми
plt.plot(y, 0 * y + F[T300], 'k',  label = "F")
plt.legend(loc='best', prop={'size': 16})

#диаграмма оксида
plt.plot([0, 0], [Ev[T300] - newFis - VBO, Ec[T300] - newFis + CBO], 'r')
plt.plot([-toxnm, -toxnm],[Ev[T300] - newFis - VBO - FisOx, Ec[T300] - newFis + CBO - FisOx],  'r')
plt.plot([-toxnm, 0], [Ev[T300] - newFis - VBO - FisOx, Ev[T300] - newFis - VBO], 'r')
plt.plot([-toxnm, 0], [Ec[T300] - newFis + CBO - FisOx, Ec[T300] - newFis + CBO], 'r')
plt.legend(loc='best', prop={'size': 16})

# уровень ферми метала
plt.plot([-toxnm - 100, -toxnm], [F[T300] - Vfb - newFis - FisOx, F[T300] - Vfb - newFis - FisOx], 'k')

fig, ax = plt.subplots() 
plt.figure(2)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('(1 / см$^3$)', fontsize=14)
plt.grid(True)
ax.set_title('Концентрации(нулевое смещение затвор-подложка)')
plt.semilogy(y, nNew, label='n', linewidth=2)
plt.semilogy(y, pNnew, label='p', linewidth=2)
plt.semilogy(y, Na1+Na2 + 0 * y, label='Na')
plt.legend(loc='best', prop={'size': 16})

#обемный заряд
fig, ax = plt.subplots() 
ax.set_title('Объемный заряд(нулевое смещение затвор-подложка)')
plt.figure(3)
plt.xlabel('x (нм)', fontsize=14)
plt.ylabel('$|\\rho/q|$ (1 / см$^3$)', fontsize=16)
plt.grid(True)
tmp = plt.semilogy(y, np.abs(QEqZero), label='$\\rho/q$')
plt.legend()
fig.show()