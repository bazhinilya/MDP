import math as mth
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import scipy as sci
from scipy.optimize import bisect
from scipy.integrate import quad
from Graphic import graphics as graph

materialName = "InSb"
Tmax = 400
Tmin = 70
Nd1 = 0
Nd2 = 0 
Ed1 = 0
Ed2 = 0

Na1 = 1e10
Na2 = 1e12
Ea1 = 0.05
Ea2 = 0.1

Nt = 1e14
σn = 1e-14
σp = 1e-16
Et = 0.08
S = 2.5e4
h = 0.05
a = 0.3
l = 0.6
λ = 1.8e-4
Ps = 0.2
R = 0.36
α = 2e4
Fm = 4.25

Eg0 = 0.235
bt = 650
at  = 4e-4
eps = 17.7
Cl = 8.1e10
Eact = 20
k = 8.62e-5
m0 = 9.11e-31
mp = 0.18
mn = 0.013
Xsame = 4.57
eps0 = 8.85e-14
workMe = 5.25 
tox = 10e-7
epsOx = 3.725

bb = 2.13
aa  = 9.6
cc = 12/5
q = -1.6e-19

T_range = np.arange(Tmin, Tmax + 1, 1)
nRange = np.arange(-10, 10, (20/(T_range.size)))

T300 = 230

plt.rcParams['figure.dpi'] = 100
plt.rcParams["figure.figsize"] = (9.6,5.4) 
plt.rcParams["axes.grid"] = True

#расчет Eg 
Eg = Eg0 - (at*(T_range**2))/(bt+T_range)
# graph.print(T_range, Eg, title = 'Ширина запрещённой зоны')

#расчет Nc, расчет Nv 
Nc = 2.5e19 * mn**(3/2) * (T_range/300)**(3/2)
Nv = 2.5e19 * mp**(3/2) * (T_range/300)**(3/2)
# graph.print(T_range, Nc, T_range, Nv, title = 'Плотность квантовых состояний', log = True)

#расчет интеграла Ферми порядка 1/2 
fermi = ((np.sqrt(np.pi/2))**(-1))*((((1.5*(2**1.5))/(bb+nRange+(((np.fabs(nRange-bb))**(1/cc))+aa)**1.5))+((np.exp(-nRange))/((np.sqrt(np.pi))/(2))))**(-1))
# graph.print(nRange, fermi, title = 'График проверки интеграла Ферми порядка 1/2', log = True)

# Зонная диаграмма
# Дно зоны проводимости, Потолок валентной зоны
Ec = Eg.copy()
Ev = np.zeros(T_range.size)

Ferm =  lambda nn: (((np.sqrt(np.pi/2))**(-1))*((((1.5*(2**1.5))/(bb+nn+(((np.fabs(nn-bb))**(1/cc))+aa)**1.5))+((np.exp(-nn))/((np.sqrt(np.pi))/(2))))**(-1)))
Ei = Eg / (k * T_range)

def f(nn, i):
    p = Nv[i] * Ferm(-nn - Ei[i])
    n = Nc[i] * Ferm(nn)
    na1 = Na1 / (1 + 2 * np.exp(-nn + (Ea1-Eg[i])/(k*T_range[i])))
    na2 = Na2 / (1 + 2 * np.exp(-nn + (Ea2-Eg[i])/(k*T_range[i])))
    return p-n-na1-na2

Fimax = Ec[330]/2 + 0.75 * k * T_range[330] * mth.log((mp/mn),(mth.exp(1)))
nn_start = (Fimax - Eg[T_range.size-1]) / (k*T_range[T_range.size-1])
ETA = [1]
ETA[0] = sci.optimize.fsolve(f, nn_start , args = (330),xtol=1e-11)[0]
for i in range(1,T_range.size):
    ETA.append(sci.optimize.fsolve(f, ETA[i-1] , args = (330-i),xtol=1e-11)[0])

# print(T_range[330],nn_start,Fimax)

ETAn = np.array(ETA [::-1])

# print(ETA[0], ETAn[330])

#Собственный уровенль Ферми
Es = Eg/2 + 0.75 * k * T_range * mth.log((mp/mn), (mth.exp(1)))
#Уровень ферми
F = k * ETAn * T_range + Eg
ea1=[]
ea2=[]
for i in range(T_range.size):
    ea1.append(Ea1)
    ea2.append(Ea2)

# graph.print(T_range, Ev, T_range, Ec, T_range, Es, T_range, ea1, T_range, ea2, T_range, F, title = 'Зонная диаграмма')

#Расчет концентраций электронов, дырок и собственной концентрации
p = Nv * Ferm(-ETAn - (Eg/(k * T_range)))
n = Nc * Ferm(ETAn)
ni = np.sqrt(n*p)

# graph.print(1/T_range, p, 1/T_range, n, 1/T_range, ni, title = 'Концентрация', log = True)
#Расчет концентраций ионов акцепторов
na1 = Na1 / (1 + 2 * np.exp(-ETAn + (Ea1-Eg)/(k*T_range)))
na2 = Na2 / (1 + 2 * np.exp(-ETAn + (Ea2-Eg)/(k*T_range)))

# graph.print(1/T_range, na1,1/T_range, na2, title = 'Концентрации ионов акцепторов', log = True)

ff = np.fabs(p-n-na1-na2)
ff[ff == 0] = 1e-5

# graph.print(T_range, ff, title = 'электронейтральность', log = True)

print('{---------------------------------Вторая часть------------------------------------------}')

#Расчет положения уровня Ферми от зарядового уравнения
FermiLevelQeq = k * T_range[T300] * np.log(Nv[T300]/(na1[T300]+na2[T300])) + Ev[T300]
#Расчет положения уровня Ферми от потенциала Ферми
φF = k*T_range[T300]*np.log(((na1[T300]+na2[T300])/ni[T300]))

# print(F[T300]," рассчитанынй уровень Ферми при 300K")
# print(Es[T300] - q * φF," уровень Ферми от Потенциала Ферми при 300K")
# print(FermiLevelQeq," уровень Ферми от основных носителей зарядов при 300K")
#Работа выхода полупроводника
WorkOut = Xsame + Eg[T300] - F[T300]
# print(WorkOut," работа выхода полупроводника при 300K")
#Расчет напряжения плоских зон
Vfb = workMe - WorkOut
# print(Vfb," напряжение плоских зон полупроводника при 300K")

#Расчет зависимости удельного поверхностного заряда от поверхностного потенциала
fsFunc = lambda fis: (0.026*p[T300]*(np.exp(-(fis / 0.026)) - 1)) + (0.026*n[T300]*(np.exp(fis/0.026) - 1)) + (fis * (na1[T300]+na2[T300]))
QsFunc = lambda fis: (-np.sign(fis) * ((2 * -q * eps*eps0* fsFunc(fis)) ** (1 / 2)))

fisArray = np.arange(-0.3, 0.3, 2 * (0.3/T_range.size))
fs = fsFunc(fisArray)
Qs = QsFunc(fisArray)

# graph.print(fisArray, Qs, title = 'Поверхностный заряд')

#Расчет зависимости поверхностного потенциала ψs от напряжения затвор—подложка Vgb, используя численные методы.
Coxp = epsOx * eps0 / tox
SPE = lambda fisNew, Vgb: Vfb+fisNew - QsFunc(fisNew) / Coxp - Vgb
Vgb = np.linspace(SPE(Ev[T300] - F[T300] - 0.181, 0), SPE(Ec[T300] - F[T300] + 0.28, 0), 331)

fisss = np.array([])
anew=Ev[T300]-F[T300]-0.181
bnew=Ec[T300]-F[T300]+0.28
for val in Vgb:
    fisss = np.hstack((fisss, bisect(f=SPE,a=anew,b=bnew, args=val, xtol=1e-12)))

# plt.xlabel("Vgb, В")
# plt.ylabel("Ψs, эВ")
# plt.title('Поверхностный потенциал')
# plt.plot(Vgb,fisss, label = 'Vgb');
# plt.axvline(x=Vfb,color='red', label = 'Vfb')
# plt.axvline(x=SPE(Ev[T300]-F[T300]-0.181,0),color='green', label = 'SPE(Ev-F)')
# plt.axvline(x=SPE(Ec[T300]-F[T300]+0.28,0),color='yellow', label = 'SPE(Ec-F)')
# plt.show()

Qnew = (fisss + Vfb - Vgb) * Coxp

# plt.xlabel("Vgb, В")
# plt.ylabel("Qs, Кл/см^2")
# plt.title('Поверхностный заряд')
# plt.plot(Vgb, Qnew, label = 'Qs')
# plt.axvline(x = Vfb, color='red', label = 'Vfb')
# plt.axvline(x = SPE(Ev[T300] - F[T300] - 0.181, 0), color = 'green', label = 'SPE(Ev-F)')
# plt.axvline(x = SPE(Ec[T300] - F[T300] + 0.28, 0), color = 'yellow', label = 'SPE(Ec-F)')
# plt.show()

print('{---------------------------------Третья часть------------------------------------------}')

qpos = -q

#Пункт 1
nfis = lambda fisss: n[T300] * np.exp(fisss / 0.026)
pfis = lambda fisss: p[T300] * np.exp(-fisss / 0.026)
Efis = lambda fisss: np.sign(fisss) * np.sqrt(2 * (qpos) * fsFunc(fisss)/ (eps*eps0))

Ifis = lambda fisss: 1 - np.abs(np.sign(fisss))
Sp = lambda f, ep: (f + np.sqrt(f**2 + 4 * ep**2)) / 2
Sn = lambda f, ep: (f - np.sqrt(f**2 + 4 * ep**2)) / 2

integr_inv =  lambda fisss: (qpos * n[T300] * (np.exp(fisss/0.026) - 1))/((np.sign(fisss) * np.sqrt(2 * (qpos) * fsFunc(fisss)/ (eps*eps0)))+Ifis(fisss))
integr_bulk =  lambda fisss: (-qpos * p[T300] * (np.exp(-fisss/0.026) - 1))/((np.sign(fisss) * np.sqrt(2 * (qpos) * fsFunc(fisss)/ (eps*eps0)))+Ifis(fisss))

Qi = np.array([])
Qi = np.zeros(331) 
Qb = np.array([])
Qb = np.zeros(331)

for i in range(331):
    result_i, error = quad(integr_inv, fisss[i], 0)
    Qi[i] = Sn(result_i, 1e-14)

for i in range(331):
    result_i, error = quad(integr_bulk, fisss[i], 0)
    Qb[i]= Sp(result_i, 1e-14)

plt.xlabel("Vgb, В")
plt.ylabel("Qi, Кл")
plt.title('Заряд в инверсном слое')
plt.plot(Vgb, Qi)
plt.axvline(x=Vfb,color='red', label = 'Vfb')
plt.show()

plt.xlabel("Vgb, В")
plt.ylabel("Qb, Кл")
plt.title('Заряд в Толщине полупроводника')
plt.plot(Vgb, Qb)
plt.axvline(x=Vfb,color='red', label = 'Vfb')
plt.show()

plt.xlabel("Vgb, В")
plt.ylabel("Qs, Кл/см^2")
plt.title('Поверхностный заряд')
plt.plot(Vgb, Qb+Qi, 'x',  label = 'Qb+Qi - расчет численными методами')
plt.plot(Vgb, Qnew, label = 'Qs - расчет из второй части РЗ')
plt.axvline(x=Vfb,color='red', label = 'Vfb');
plt.show()

#Пункт 2
# возможно ошибка в экспоненте, проверить к преподавателя или использовать сглаживающие функции
# стоял + и все работало, стоит минус и все перестало работать, поэтому поставил модуль.
# QbA = lambda fisss: -np.sign(fisss) * np.sqrt(2 * qpos * (eps * eps0)  * fA(fisss)) - для копирования
fA = lambda fisss: fisss * (Na1+Na2) + 0.026 * p[T300] * (np.exp(-fisss/0.026) - 1)
QbA = lambda fisss: -np.sign(fisss) * np.sqrt(Sp(2 * qpos * (eps * eps0)  * fA(fisss), 1e-20))
QiA = lambda fisss: 2 * (qpos) * (eps*eps0) * 0.026 * n[T300] * (np.exp(fisss / 0.026) - 1) /(Qnew + QbA(fisss)+ Ifis(fisss))

QbA_mas = np.array([])
QbA_mas = np.zeros(331)
QbA_mas = QbA(fisss)

QiA_mas = np.array([])
QiA_mas = np.zeros(331)
QiA_mas = QiA(fisss)

plt.xlabel("Vgb, В")
plt.ylabel("QbA, Кл/см^2")
plt.title('Заряд в толще полупроводника аналитически')
plt.plot(Vgb, QbA_mas, '*', label = 'QbA - аналитически')
plt.plot(Vgb, Qb, '-', color = 'r', label = 'Qb - численно')
plt.axvline(x=Vfb,color='green', label = 'Vfb')
plt.semilogy()
plt.show()

plt.xlabel("Vgb, В")
plt.ylabel("QiA, Кл/см^2")
plt.title('Заряд в инверсном аналитически')
plt.plot(Vgb, np.abs(QiA_mas), '*', label = 'QiA - аналитически')
plt.plot(Vgb, np.abs(Qi),'-', color = 'r', label = 'Qi - численно')
plt.legend()
plt.semilogy()
plt.axvline(x=Vfb,color='green', label = 'Vfb')
plt.show()

plt.xlabel("Vgb, В")
plt.ylabel("QbA, Кл/см^2")
plt.title('Поверхностый заряд аналитически')
plt.plot(Vgb, QiA_mas + QbA_mas, '*', label = 'QiA')
plt.plot(Vgb, Qnew, label = 'Qs')
plt.axvline(x=Vfb,color='green', label = 'Vfb')
plt.show()

plt.xlabel("Vgb, В")
plt.ylabel("Ошибка, %")
plt.title('Заряд в толще проводника')
plt.plot(Vgb, np.abs(100 * ((QbA(fisss) - Qb) / (QbA(fisss)))));
plt.axvline(x=Vfb,color='green', label = 'Vfb')
plt.semilogy()
plt.show()

plt.xlabel("Vgb, В")
plt.ylabel("Ошибка, %")
plt.title('Заряд в инверсном слое проводника')
plt.plot(Vgb, np.abs(100 * (QiA(fisss) - Qi) / (QiA(fisss)+1e-12)))
plt.axvline(x=Vfb,color='green', label = 'Vfb')
plt.semilogy()
plt.show()

# Пункт 3
Vgb = np.linspace(SPE(Ev[T300] - F[T300] - 0.181, 0), SPE(Ec[T300] - F[T300] + 0.28, 0), 84736)
fisss = np.array([])
# anew = Ev[T300]-F[T300]-0.181
# bnew=Ec[T300]-F[T300]+0.28
for val in Vgb:
   fisss = np.hstack((fisss, bisect(f=SPE,a=Ev[T300]-F[T300]-0.181,b=Ec[T300]-F[T300]+0.28, args=val, xtol=1e-12)))

fsFunc = lambda fis: (0.026*p[T300]*(np.exp(-(fis / 0.026)) - 1)) + (0.026*n[T300]*(np.exp(fis/0.026) - 1)) + (fis * (na1[T300]+na2[T300]))

fisArray = np.arange(-0.3, 0.3, 0.0078125 * (0.3/T_range.size))
fs = fsFunc(fisArray)

# Оценка потенциалов слабой и сильной инверсии
phi_WI = 0.026 * mth.log(ni[T300]/n[T300])
phi_SI = 0.026 * mth.log((Na1+Na2)/n[T300])
print(phi_WI, ' для слабой инверсии и ', phi_SI, ' для сильной инверсии')

# Возможно ошибка в том, что я строю заряд в кулонах, температуру в Кельвинах, а постоянная больцмана в эВ
Cs0 = np.sqrt(((qpos*eps * eps0) / (k*300)) * (p[T300] + n[T300]))
df = lambda fisss: (Na2+Na1) - p[T300] * np.exp(-fisss / 0.026) + n[T300] * np.exp(fisss / 0.026)
Cs = lambda fisss: np.sign(fisss) * df(fisss) * np.sqrt((qpos*eps*eps0)/(2*fs+Ifis(fisss))) + Cs0
Cgb = Cs(fisss) * Coxp / (Cs(fisss) + Coxp)
#Cgb0 = (Cs0 * Coxp) / (Cs0 + Coxp)

Cgb0 = np.array([])
Cgb0 = np.zeros(84736)
for iii in range(84736):
        Cgb0[iii] = (Cs0 * Coxp) / (Cs0 + Coxp)


plt.xlabel("Vgb, В")
plt.ylabel("Cgb/Cox")
plt.title('Емкость')
plt.plot(Vgb, Cgb/Coxp, label = "Cgb/Coxp")
plt.plot(Vgb, Cgb0/Coxp, label = "Cgb0/Coxp,")
plt.axvline(x=Vfb,color='red', label = 'Vfb')
plt.xlim(0.1,1.2)
plt.ylim(0,1.1)
plt.show()

Vgb = np.linspace(SPE(Ev[T300]-F[T300]-0.181,0),SPE(Ec[T300]-F[T300]+0.28,0),331)
fisss = np.array([])
# anew = Ev[T300] - F[T300] - 0.181
# bnew=Ec[T300]-F[T300]+0.28
for val in Vgb:
    fisss = np.hstack((fisss, bisect(f=SPE,a=Ev[T300] - F[T300] - 0.181,b=Ec[T300]-F[T300]+0.28, args=val, xtol=1e-12)))

fsFunc = lambda fis: (0.026*p[T300]*(np.exp(-(fis / 0.026)) - 1)) + (0.026*n[T300]*(np.exp(fis/0.026) - 1)) + (fis * (na1[T300]+na2[T300]))
fisArray = np.arange(-0.3, 0.3, 2 * (0.3 / T_range.size))
fs = fsFunc(fisArray)

#Пункт 4
Ci = lambda fis: qpos*(nfis(fis)-n[T300])/Efis(fis)
Cb = lambda fis: -qpos*(pfis(fis)-p[T300])/Efis(fis)

Cb0 = np.sign(fisss)*np.sqrt((qpos*eps*eps0*p[T300])*(0.026))
Ci0 = np.sign(fisss)*np.sqrt((qpos*eps*eps0*n[T300])*(0.026))

Ci_mas = np.zeros(331)
for i in range(0,331):
    Ci_mas[i] = Ci(fisss[i])
Cb_mas = np.zeros(331)
for i in range(0,331):
    Cb_mas[i] = Cb(fisss[i]) 

plt.xlabel("Vgb, В")
plt.ylabel("C, Ф/см^2")
plt.title('Емкости')
plt.plot(Vgb, Ci_mas + Cb_mas, 'x', label = "Cs")
plt.plot(Vgb, Ci_mas, label = "Ci")
plt.plot(Vgb, Cb_mas, label = "Cb")
plt.axvline(x=Vfb,color='red', label = 'Vfb')
plt.show()