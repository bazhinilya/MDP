from Const import const
from Graphic import graphics as graph

from numpy import array, arange, exp, fabs, sqrt, log, sign, linspace, hstack
from scipy.optimize import fsolve, bisect
from math import pi
from varname import nameof

T_range = arange(const.Tmin, const.Tmax + 1)

#1. Зависимость ширины запрещённой зоны Eg от температуры.
Eg_T = array(const.Eg0 - (const.alpha_T / (const.beta_T + T_range)) * T_range ** 2)
graph.print(T_range, Eg_T, xlabel = nameof(T_range), ylabel = nameof(Eg_T))

#2. Зависимость эффективной плотности квантовых состояний Nc, Nv от температуры.
Nc_T = array(2.5e19 * (const.mn)**(3/2) * (T_range/300)**(3/2)) 
Nv_T = array(2.5e19 * (const.mp)**(3/2) * (T_range/300)**(3/2))
graph.print(T_range, Nc_T, T_range, Nv_T, log = True, xlabel = nameof(T_range), ylabel = nameof(Nc_T, Nv_T))

#3. График проверки интеграла Ферми порядка ½.
eta_range = arange(-10, 9.82, 0.06)

def F_12_eta(eta): 
    return sqrt(pi/2)**(-1) * (((1.5*2**(1.5))/(2.13+eta+((fabs(eta-2.13))**(1/2.4)+9.6)**(1.5)))+exp(-eta)/(sqrt(pi)/2))**(-1)
graph.print(eta_range, F_12_eta(eta_range), log = True, xlabel = nameof(eta_range), ylabel = nameof(F_12_eta))

#4. Зависимость положения энергетических уровней на зонной диаграмме от T (Ev, Ec, F, Fi, Ea1, Ea2).
Ei_T = Eg_T / (const.k * T_range)
Ev_T = 0 * T_range
Ec_T = Eg_T
Fi_T = Ec_T/2 + (3/4) * const.k * T_range * log(const.mp / const.mn)

def Na1_eta(eta, i = None): 
    if i is None: 
        return const.Na1 / (1 + 2*exp(-eta + (const.Ea1 - Eg_T) / (const.k * T_range)))    
    return const.Na1 / (1 + 2*exp(-eta + (const.Ea1 - Eg_T[i]) / (const.k * T_range[i])))

def Na2_eta(eta, i = None): 
    if i is None:
        return const.Na2 / (1 + 2*exp(-eta + (const.Ea2 - Eg_T) / (const.k * T_range)))
    return const.Na2 / (1 + 2*exp(-eta + (const.Ea2 - Eg_T[i]) / (const.k * T_range[i])))

def f_eta_T(eta, i)->float: 
    na1 = Na1_eta(eta, i)
    na2 = Na2_eta(eta, i)
    n = Nc_T[i] * F_12_eta(eta)
    p = Nv_T[i] * F_12_eta(-eta-Ei_T[i])
    return p - n - na1 - na2

eta_static = (Fi_T[0] - Eg_T[0]) / (const.k * const.Tmax)
ETA = [1]
ETA[0] = fsolve(f_eta_T, eta_static, args = 0, xtol = 1e-12)[0]

for i in range(1, T_range.size):
    ETA.append(fsolve(f_eta_T, ETA[i-1], args = i, xtol=1e-12)[0])

ETA = array(ETA)
F_T = ETA * const.k * T_range + Eg_T

graph.print(T_range, Ev_T, T_range, Ec_T, T_range, F_T, T_range, Fi_T, T_range, Ev_T + const.Ea1, T_range, Ev_T + const.Ea2, 
                 xlabel = nameof(T_range),
                 ylabel = nameof(Ev_T, Ec_T, F_T, Fi_T) + tuple({"Ea1", "Ea2"}),
                 title = "Зонная диаграмма")

# 5. Зависимость концентраций электронов n, дырок p, собственной концентрации ni от температуры.
n_T = Nc_T * F_12_eta(ETA)
p_T = Nv_T * F_12_eta(-ETA - Eg_T / (const.k * T_range))
ni_T = sqrt(n_T * p_T)
graph.print(1/T_range, n_T, 1/T_range, p_T, 1/T_range, ni_T, log = True, xlabel = "1/" + nameof(T_range), ylabel = nameof(n_T, p_T, ni_T))

# 6. Зависимость концентрации ионов доноров и акцепторов для каждой примеси Na1, Na2 от T
na1 = Na1_eta(ETA)
na2 = Na2_eta(ETA)
graph.print(1/T_range, na1, 1/T_range, na2, log = True, xlabel = "1/" + nameof(T_range), ylabel = nameof(na1, na2))

# 7. График величины отклонения суммы зарядов от нуля (проверка условия электронейтральности) для каждой T
Pr_T = fabs(p_T - n_T - na1 - na2)
graph.print(T_range, Pr_T, log = True, xlabel = nameof(T_range), title = "Проверка условия электронейтральности")

# 8. Оценка уровня Ферми
Na_T = p_T
Ef_T = const.k * T_range * log(Nv_T/Na_T) + Ev_T
graph.print(T_range, F_T, T_range, Ef_T, xlabel = nameof(T_range), ylabel = nameof(F_T, Ef_T), title = "Оценка уровня Ферми", log = True)

#TODO: Значения уходят в некорректный диапазон, проверить формулы и параметры

# 9. Работа выхода полупроводника
#TODO: psi=const ?
psi = 4.59
Fs = psi + Ec_T[0] - F_T[0]
print("Работа выхода пп = ", Fs)

# 10. Напряжение плоских зон
Vfb = const.Fm - Fs
print("Напряжение плоских зон", Vfb)

# 11. Зависимость удельного поверхностного заряда от поверхностного потенциала
fis = arange(-1, 1, 0.001)

def f_fis(fis):
    return const.fiT * p_T[0] * (exp(-fis/const.fiT) - 1) + const.fiT * n_T[0] * (exp(fis/const.fiT) - 1) + fis * (Na_T[0])
def Qs_fis(fis):
    return -sign(fis) * (2 * const.q * const.Es * const.E0 * f_fis(fis))**(1/2)
graph.print(fis, Qs_fis(fis))

# 12. Зависимость поверхностного потенциала ψs от напряжения затвор — подложка Vgb
Coxp = const.Eox * const.E0 / const.tox
def SPE(fis, Vgb):
    return Vfb + fis - Qs_fis(fis) / Coxp - Vgb

Vgb = linspace(SPE(Ev_T[0] - F_T[0], 0), SPE(Ec_T[0] - F_T[0], 0), 201)
fis = array([])

for item in Vgb:
    fis = hstack((fis, bisect(SPE, a = Ev_T[0] - F_T[0] - 0.2, b = Ec_T[0] - F_T[0] + 0.2, args = item, xtol = 1e-12)))
graph.print(Vgb, fis)
#TODO: отметить значения напряжения плоских зон

# 13. Зависимость поверхностного заряда от напряжения затвор — подложка
graph.print(Vgb, Qs_fis(fis))
#TODO: отметить значения напряжения плоских зон