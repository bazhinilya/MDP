from Const import const
from Graphic import graphics as graph

from numpy import array, arange, exp, fabs, sqrt, log, sign, linspace, hstack, where, zeros
from scipy.optimize import fsolve, bisect
from scipy.integrate import quad
from math import pi
from varname import nameof

T_range = arange(const.Tmin, const.Tmax + 1)

#1. Зависимость ширины запрещённой зоны Eg от температуры.
Eg_T = array(const.Eg0 - (const.alpha_T / (const.beta_T + T_range)) * T_range ** 2)
# graph.print(T_range, Eg_T, xlabel = nameof(T_range), ylabel = nameof(Eg_T))

#2. Зависимость эффективной плотности квантовых состояний Nc, Nv от температуры.
Nc_T = array(2.5e19 * (const.mn)**(3/2) * (T_range/300)**(3/2)) 
Nv_T = array(2.5e19 * (const.mp)**(3/2) * (T_range/300)**(3/2))
# graph.print(T_range, Nc_T, T_range, Nv_T, log = True, xlabel = nameof(T_range), ylabel = nameof(Nc_T, Nv_T))

#3. График проверки интеграла Ферми порядка ½.
eta_range = arange(-10, 9.82, 0.06)

F_12_eta = lambda eta: sqrt(pi/2)**(-1) * (((1.5*2**(1.5))/(2.13+eta+((fabs(eta-2.13))**(1/2.4)+9.6)**(1.5)))+exp(-eta)/(sqrt(pi)/2))**(-1)
# graph.print(eta_range, F_12_eta(eta_range), log = True, xlabel = nameof(eta_range), ylabel = nameof(F_12_eta))

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

# graph.print(T_range, Ev_T, T_range, Ec_T, T_range, F_T, T_range, Fi_T, T_range, Ev_T + const.Ea1, T_range, Ev_T + const.Ea2, 
#                  xlabel = nameof(T_range),
#                  ylabel = nameof(Ev_T, Ec_T, F_T, Fi_T) + tuple({"Ea1", "Ea2"}),
#                  title = "Зонная диаграмма")

# 5. Зависимость концентраций электронов n, дырок p, собственной концентрации ni от температуры.
n_T = Nc_T * F_12_eta(ETA)
p_T = Nv_T * F_12_eta(-ETA - Eg_T / (const.k * T_range))
ni_T = sqrt(n_T * p_T)
# graph.print(1/T_range, n_T, 1/T_range, p_T, 1/T_range, ni_T, log = True, xlabel = "1/" + nameof(T_range), ylabel = nameof(n_T, p_T, ni_T))

# 6. Зависимость концентрации ионов доноров и акцепторов для каждой примеси Na1, Na2 от T
na1 = Na1_eta(ETA)
na2 = Na2_eta(ETA)
# graph.print(1/T_range, na1, 1/T_range, na2, log = True, xlabel = "1/" + nameof(T_range), ylabel = nameof(na1, na2))

# 7. График величины отклонения суммы зарядов от нуля (проверка условия электронейтральности) для каждой T
Pr_T = fabs(p_T - n_T - na1 - na2)
# graph.print(T_range, Pr_T, log = True, xlabel = nameof(T_range), title = "Проверка условия электронейтральности")

# ЧАСТЬ 2

# 1. Оценка уровня Ферми
T300 = const.Tmax - 70 - 1

Ef_T = Fi_T - const.k * T_range * log((const.Na1 + const.Na2)/ni_T)
#Аналитический Ef
Ef_T_altr = const.k * T_range * log(Nv_T/p_T) + Ev_T
# graph.print(T_range, F_T, T_range, Ef_T, T_range, Ef_T_altr, 
#             xlabel = nameof(T_range), 
#             ylabel = nameof(F_T, Ef_T_altr, Ef_T), 
#             title = "Оценка уровня Ферми", 
#             log = True)

# 2. Работа выхода полупроводника
Фs = const.psi + Eg_T[T300] - F_T[T300]
print("Работа выхода пп = ", Фs)

# 3. Напряжение плоских зон
Vfb = const.Фm - Фs
print("Напряжение плоских зон", Vfb)

# 4. Зависимость удельного поверхностного заряда от поверхностного потенциала
Ψs = arange(-1, 1, 0.001)
f_Ψs = lambda Ψs: const.φT * p_T[T300] * (exp(-Ψs/const.φT) - 1) + const.φT * n_T[T300] * (exp(Ψs/const.φT) - 1) + Ψs * (p_T[T300])
Qs_Ψs = lambda Ψs: -sign(Ψs) * (2 * const.q * const.εs * const.ε0 * f_Ψs(Ψs))**(1/2)
# graph.print(Ψs, Qs_Ψs(Ψs), xlabel = nameof(Ψs), ylabel = nameof(Qs_Ψs))

# 5. Зависимость поверхностного потенциала Ψss от напряжения затвор — подложка Vgb
Coxp = const.εox * const.ε0 / const.tox
SPE = lambda Ψs, Vgb: Vfb + Ψs - Qs_Ψs(Ψs) / Coxp - Vgb

Vgb = linspace(SPE(Ev_T[T300] - F_T[T300], 0), SPE(Ec_T[T300] - F_T[T300], 0), 201)
Ψs = array([])
for item in Vgb:
    Ψs = hstack((Ψs, bisect(SPE, a = Ev_T[T300] - F_T[T300] - 0.2, b = Ec_T[T300] - F_T[T300] + 0.2, args = item, xtol = 1e-12)))
# graph.print(Vgb, Ψs, xlabel = nameof(Vgb), ylabel = nameof(Ψs), xline = Vfb, yline_min = Ψs.min(), yline_max = Ψs.max())

# 6. Зависимость поверхностного заряда от напряжения затвор — подложка
# graph.print(Vgb, Qs_Ψs(Ψs), xlabel = nameof(Vgb), ylabel = nameof(Qs_Ψs), xline = Vfb, yline_min = -5e-7, yline_max = 3.5e-6)

# ЧАСТЬ 3

# 1. Рассчитать Qi и Qb. Зависимость этих зарядов от Vgb. Построить график суммы заряда
# Qi и Qb от Vgb и сравнить с зависимостью Qs(Vgb).

I = lambda Ψs: 1 - abs(sign(Ψs))
n = lambda Ψs: n_T[T300] * exp(Ψs / const.φT)
p = lambda Ψs: p_T[T300] * exp(-Ψs / const.φT)
E = lambda Ψs: sign(Ψs) * sqrt(2 * const.q / (const.ε0 * const.εs) * f_Ψs(Ψs))
Sp = lambda f, ep: (f + sqrt(f**2 + 4 * ep**2))/2
Qi = array([])
Qb = array([])
integrand =  lambda Ψs: - const.q * (p(Ψs) - p_T[T300]) / (E(Ψs) + I(Ψs))
integrand2 = lambda Ψs: const.q * (n(Ψs) - n_T[T300]) / (E(Ψs) + I(Ψs))
Qi = zeros(201) 
for i in range(201):
        result_i, error = quad(integrand, Ψs[i], 0)
        Qi[i]= Sp(result_i, 1e-19)

graph.print(Vgb, Qi, title = 'Заряд в инверсном слое', log = True)

Qb = zeros(201)
for i in range(201):
        result_i, error = quad(integrand2, Ψs[i], 0)
        Qb[i]= (result_i)

graph.print(Vgb, Qb, title = 'Заряд в толще полупроводника')

graph.print(Vgb, Qb + Qi, Vgb, Qs_Ψs(Ψs), title = 'Поверхностный заряд')

# 2 В соответствии с зарядовой моделью МДП — структуры рассчитать аналитически заряды Qb и Qi, построить их зависимости от Vgb, сравнить
#результат с численным расчетом в предыдущем пункте. Рассчитать в процентах разницу между численным и аналитическим решением. Провести анализ
#ошибки вычисления заряда в инверсном слое в режиме слабой и сильной инверсии.
fA = lambda Ψs: Ψs * (const.Na1 + const.Na2) + const.φT * n_T[T300] * (exp(Ψs / const.φT) - 1)
QbA = lambda Ψs: -sign(Ψs) * sqrt(2 * const.q * const.ε0 * const.εs * fA(Ψs))
QiA = lambda Ψs: 2 * const.q * const.ε0 * const.εs * const.φT * p_T[T300] * (exp(-Ψs / 0.026) - 1) /(Qs_Ψs(Ψs) + QbA(Ψs)+ I(Ψs))
QbA_mas = zeros(201)
for i in range(0,201):
    QbA_mas[i] = QbA(Ψs[i])
graph.print(Vgb, QbA_mas, Vgb, Qb,title = 'Заряд в толще полупроводника')

QiA_mas = zeros(201)
for i in range(0,201):
    QiA_mas[i] = QiA(Ψs[i])
QiA_mas = QiA(Ψs)
graph.print(Vgb, QiA_mas, Vgb, Qi, title = 'Заряд в инверсном слое', log = True)

graph.print(Vgb, 100 * (QbA(Ψs) - Qb) / Qb, title = 'Заряд в толще полупроводника ')

graph.print(Vgb, 100 * (QiA(Ψs) - Qi) / Qi, title = 'Заряд в инверсном слое')

# selector = where(Ψs <= phif)[0]
Wi = Ei_T[T300] - F_T[T300]
graph.print(Vgb, 100 * (QiA(Ψs) - Qi) / Qi, title = 'график ошибок инверсного заряда', xline = Wi, yline_min = Vgb.min(), yline_max = Vgb.max()) 
# ax.set_xlim(-6,1)
# ax.set_ylim(-20,1);
Si = 2 * (Ei_T[T300] - F_T[T300]) 
# ax.vlines(Si, 10, -20,color = 'red')
# ax.legend(labels = ("ошибки инверсного заряда","сильная инверсия", "слабая инверсия"), loc="lower left");

# 3. Рассчитать и построить график зависимости ёмкости затвор — подложка
#(ёмкости всей МДП — структуры) Сgb от напряжения затвор — подложка Vgb .
#Отметить на получившемся графике точку, соответствующую плоским зонам.
#Рекомендуется по оси ординат откладывать ёмкость в долях от ёмкости
#диэлектрика.
Cs0 = sqrt(const.q * const.εs * const.ε0 / const.φT * (p_T[T300] + n_T[T300]))
df = lambda Ψs: const.Na1 + const.Na2 - p_T[T300] * exp(-Ψs / const.φT) + n_T[T300] * exp(Ψs / const.φT)
Cs = lambda Ψs: sign(Ψs) * df(Ψs) * sqrt(const.q * const.εs * const.ε0 / 2 / (f_Ψs(Ψs) + I(Ψs))) + I(Ψs) * Cs0
Vgb = linspace(SPE(Ev_T[T300] - F_T[T300], 0), SPE(Ec_T[T300] - F_T[T300], 0), 201)
Cgb = Cs(Ψs) * Coxp / (Cs(Ψs) + Coxp)
Cgb0 = Cs(0) * Coxp / (Cs(0) + Coxp)
Cs_mas1 = zeros(201)
for i in range(0,201):
    Cs_mas1[i] = Cs(Ψs[i])
Cg_mas1 = zeros(201)  
for i in range(0,201):
    Cg_mas1[i] = Cs_mas1[i] * Coxp / (Cs_mas1[i] + Coxp)
Cs0 = sqrt(const.q * const.εs * const.ε0 / const.φT * (p_T[T300] + n_T[T300]))
Cgb0 = Cs0* Coxp / (Cs0 + Coxp)/Coxp
graph.print(Vgb, Cg_mas1/Coxp, title = 'Ёмкость затовора')

# ax.vlines(Vfb, Vgb.min(), Vgb.max(),color = 'black');
# plt.text(3, 0.2, "Vfb", fontsize=14);
# ax.set_ylim(0,1.1);
# ax.hlines(Cgb0, -10, 20,color = 'red')
# plt.text(-10, 0.35, "Cgb0/Cox", fontsize=14);

# 4. Рассчитать и построить зависимость ёмкости полупроводника Cs ,
#инверсного слоя Ci , подложки Cb от напряжения затвор — подложка Vgb.
#Рекомендуется построить все три ёмкости на одном графике.
n1 = lambda Ψs: n_T[T300] * exp(Ψs/const.φT)
p1 = lambda Ψs: p_T[T300] * exp(-Ψs/const.φT)
Ess = lambda Ψs: sign(Ψs) * sqrt(2 * const.q * f_Ψs(Ψs) / (const.ε0 * const.εs))

Ci = lambda Ψs: const.q * (n1(Ψs) - n_T[T300]) / Ess(Ψs)
Cb = lambda Ψs: -const.q * (p1(Ψs) - p_T[T300]) / Ess(Ψs)
Ci_mas = zeros(201)
for i in range(0,201):
    Ci_mas[i] = Ci(Ψs[i])
Cb_mas = zeros(201)
for i in range(0,201):
    Cb_mas[i] = Cb(Ψs[i])    
graph.print(Vgb, Ci_mas, Vgb, Cb_mas, Vgb,Ci_mas + Cb_mas, title = 'Ёмкости')