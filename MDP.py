import math as mth
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
from scipy.optimize import bisect
from scipy.integrate import quad

materialName = "InSb"
Tmax = 400
Tmin = 70

Na1 = 1e10
Na2 = 1e12
Ea1 = 0.05
Ea2 = 0.1

Eg0 = 0.235
bt = 650
at  = 4e-4
eps = 17.7

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
f_Ψs = lambda Ψs: 0.026 * p[T300] * (np.exp(-Ψs/0.026) - 1) + 0.026 * n[T300] * (np.exp(Ψs/0.026) - 1) + Ψs * (Na1+Na2)
QsFunc = lambda Ψs: (-np.sign(Ψs) * ((2 * -q * eps*eps0* f_Ψs(Ψs)) ** (1 / 2)))

Ψs_range = np.arange(-0.3, 0.3, 2 * (0.3/T_range.size))
fs = f_Ψs(Ψs_range)
Qs = QsFunc(Ψs_range)

# graph.print(Ψs_range, Qs, title = 'Поверхностный заряд')

#Расчет зависимости поверхностного потенциала ψs от напряжения затвор—подложка Vgb, используя численные методы.
Coxp = epsOx * eps0 / tox
SPE = lambda ΨsNew, Vgb: Vfb+ΨsNew - QsFunc(ΨsNew) / Coxp - Vgb
Vgb = np.linspace(SPE(Ev[T300] - F[T300] - 0.181, 0), SPE(Ec[T300] - F[T300] + 0.28, 0), 331)

Ψs = np.array([])
for val in Vgb:
    Ψs = np.hstack((Ψs, bisect(f = SPE, a = Ev[T300] - F[T300] - 0.181, b = Ec[T300]-F[T300]+0.28, args = val, xtol = 1e-12)))

# plt.xlabel("Vgb, В")
# plt.ylabel("Ψs, эВ")
# plt.title('Поверхностный потенциал')
# plt.plot(Vgb,Ψs, label = 'Vgb');
# plt.axvline(x=Vfb,color='red', label = 'Vfb')
# plt.axvline(x=SPE(Ev[T300]-F[T300]-0.181,0),color='green', label = 'SPE(Ev-F)')
# plt.axvline(x=SPE(Ec[T300]-F[T300]+0.28,0),color='yellow', label = 'SPE(Ec-F)')
# plt.show()

Qnew = (Ψs + Vfb - Vgb) * Coxp

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
n_Ψs = lambda Ψs: n[T300] * np.exp(Ψs / 0.026)
p_Ψs = lambda Ψs: p[T300] * np.exp(-Ψs / 0.026)
E_Ψs = lambda Ψs: np.sign(Ψs) * np.sqrt(2 * (qpos) * f_Ψs(Ψs)/ (eps*eps0))

I_Ψs = lambda Ψs: 1 - np.abs(np.sign(Ψs))
Sp = lambda f, ep: (f + np.sqrt(f**2 + 4 * ep**2)) / 2
Sn = lambda f, ep: (f - np.sqrt(f**2 + 4 * ep**2)) / 2

integr_inv =  lambda Ψs: (qpos * n[T300] * (np.exp(Ψs/0.026) - 1))/((np.sign(Ψs) * np.sqrt(2 * (qpos) * f_Ψs(Ψs)/ (eps*eps0)))+I_Ψs(Ψs))
integr_bulk =  lambda Ψs: (-qpos * p[T300] * (np.exp(-Ψs/0.026) - 1))/((np.sign(Ψs) * np.sqrt(2 * (qpos) * f_Ψs(Ψs)/ (eps*eps0))) + I_Ψs(Ψs))

Qi = np.zeros(331) 
Qb = np.zeros(331)

for i in range(331):
    result_i, error = quad(integr_inv, Ψs[i], 0)
    Qi[i] = Sn(result_i, 1e-11)

for i in range(331):
    result_i, error = quad(integr_bulk, Ψs[i], 0)
    Qb[i] = result_i
    # Sp(result_i, 1e-11)

plt.xlabel("Vgb, В")
plt.ylabel("Qi, Кл")
plt.title('Заряд в инверсном слое')
plt.plot(Vgb, np.abs(Qi))
plt.axvline(x = Vfb,color='red', label = 'Vfb')
plt.semilogy()
plt.show()

plt.xlabel("Vgb, В")
plt.ylabel("Qb, Кл")
plt.title('Заряд в Толщине полупроводника')
plt.plot(Vgb, np.abs(Qb))
plt.axvline(x = Vfb, color = 'red', label = 'Vfb')
plt.semilogy()
plt.show()

plt.xlabel("Vgb, В")
plt.ylabel("Qs, Кл/см^2")
plt.title('Поверхностный заряд')
plt.plot(Vgb, Qb + Qi, 'x',  label = 'Qb+Qi - расчет численными методами')
plt.plot(Vgb, Qnew, label = 'Qs - расчет из второй части РЗ')
plt.axvline(x = Vfb, color = 'red', label = 'Vfb')
plt.show()

#Пункт 2
fA_Ψs = lambda Ψs: 0.026 * p[T300] * (np.exp(-Ψs/0.026) - 1) + Ψs * (Na1 + Na2)
QbA = lambda Ψs: -np.sign(Ψs) * np.sqrt(2 * qpos * (eps * eps0) * fA_Ψs(Ψs))
QiA = lambda Ψs: 2 * (qpos) * (eps * eps0) * 0.026 * n[T300] * (np.exp(Ψs / 0.026) - 1) /(Qnew + QbA(Ψs)+ I_Ψs(Ψs))
print(np.exp(-Ψs/0.026) - 1)

QiA_mas = QiA(Ψs)
QbA_mas = QbA(Ψs)

#TODO: поднять красную
plt.xlabel("Vgb, В")
plt.ylabel("QiA, Кл/см^2")
plt.title('Заряд в инверсном аналитически')
plt.plot(Vgb, np.abs(QiA_mas), '*', label = 'QiA - аналитически')
plt.plot(Vgb, np.abs(Qi),'-', color = 'r', label = 'Qi - численно')
plt.legend()
plt.semilogy()
plt.axvline(x = Vfb, color = 'green', label = 'Vfb')
plt.show()

plt.xlabel("Vgb, В")
plt.ylabel("QbA, Кл/см^2")
plt.title('Заряд в толще полупроводника аналитически')
plt.plot(Vgb, np.abs(QbA_mas), '*', label = 'QbA - аналитически')
plt.plot(Vgb, np.abs(Qb), '-', color = 'r', label = 'Qb - численно')
plt.axvline(x = Vfb, color = 'green', label = 'Vfb')
plt.semilogy()
plt.legend()
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
plt.plot(Vgb, np.abs(100 * ((QbA(Ψs) - Qb) / (QbA(Ψs)))))
plt.axvline(x=Vfb,color='green', label = 'Vfb')
plt.semilogy()
plt.show()

plt.xlabel("Vgb, В")
plt.ylabel("Ошибка, %")
plt.title('Заряд в инверсном слое проводника')
plt.plot(Vgb, np.abs(100 * (QiA(Ψs) - Qi) / (QiA(Ψs)+1e-12)))
plt.axvline(x=Vfb,color='green', label = 'Vfb')
plt.semilogy()
plt.show()

# Пункт 3
Vgb = np.linspace(SPE(Ev[T300] - F[T300] - 0.181, 0), SPE(Ec[T300] - F[T300] + 0.28, 0), 84736)
Ψs = np.array([])
# anew = Ev[T300]-F[T300]-0.181
# bnew=Ec[T300]-F[T300]+0.28
for val in Vgb:
   Ψs = np.hstack((Ψs, bisect(f=SPE,a=Ev[T300]-F[T300]-0.181,b=Ec[T300]-F[T300]+0.28, args=val, xtol=1e-12)))

Ψs_range = np.arange(-0.3, 0.3, 0.0078125 * (0.3/T_range.size))
fs = f_Ψs(Ψs_range)

# Оценка потенциалов слабой и сильной инверсии
phi_WI = 0.026 * mth.log(ni[T300]/n[T300])
phi_SI = 0.026 * mth.log((Na1+Na2)/n[T300])
print(phi_WI, ' для слабой инверсии и ', phi_SI, ' для сильной инверсии')

Cs0 = np.sqrt(((qpos*eps * eps0) / (k*300)) * (p[T300] + n[T300]))
df = lambda Ψs: (Na2+Na1) - p[T300] * np.exp(-Ψs / 0.026) + n[T300] * np.exp(Ψs / 0.026)
Cs = lambda Ψs: np.sign(Ψs) * df(Ψs) * np.sqrt((qpos*eps*eps0)/(2*fs+I_Ψs(Ψs))) + Cs0
Cgb = Cs(Ψs) * Coxp / (Cs(Ψs) + Coxp)

Cgb0 = np.zeros(84736)
for i in range(84736):
    Cgb0[i] = (Cs0 * Coxp) / (Cs0 + Coxp)


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
Ψs = np.array([])
# anew = Ev[T300] - F[T300] - 0.181
# bnew=Ec[T300]-F[T300]+0.28
for val in Vgb:
    Ψs = np.hstack((Ψs, bisect(f=SPE,a=Ev[T300] - F[T300] - 0.181,b=Ec[T300]-F[T300]+0.28, args=val, xtol=1e-12)))

Ψs_range = np.arange(-0.3, 0.3, 2 * (0.3 / T_range.size))
fs = f_Ψs(Ψs_range)

#Пункт 4
Ci = lambda Ψs: qpos*(n_Ψs(Ψs)-n[T300])/E_Ψs(Ψs)
Cb = lambda Ψs: -qpos*(p_Ψs(Ψs)-p[T300])/E_Ψs(Ψs)

Cb0 = np.sign(Ψs)*np.sqrt((qpos*eps*eps0*p[T300])*(0.026))
Ci0 = np.sign(Ψs)*np.sqrt((qpos*eps*eps0*n[T300])*(0.026))

plt.xlabel("Vgb, В")
plt.ylabel("C, Ф/см^2")
plt.title('Емкости')
plt.plot(Vgb, Ci(Ψs) + Cb(Ψs), 'x', label = "Cs")
plt.plot(Vgb, Ci(Ψs), label = "Ci")
plt.plot(Vgb, Cb(Ψs), label = "Cb")
plt.axvline(x=Vfb,color='red', label = 'Vfb')
plt.show()

print('{---------------------------------Четвертая часть------------------------------------------}')