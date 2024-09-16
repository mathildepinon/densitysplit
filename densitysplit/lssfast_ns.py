import sympy as sp
from scipy.integrate import quad
from scipy.optimize import fsolve

# Define the symbols
x, y, nu, ns, alpha, r, s, beta, sigma, R, k = sp.symbols('x y nu ns alpha r s beta sigma R k')

# Define the functions
Zeta = 1 / (1 - x / nu)**nu
Tau = sp.solve(y - Zeta, x)[0]
Sigma = sp.sqrt(2 / ((r)**(3 + ns + alpha) + (r)**(3 + ns - alpha)))
Psi = (1 / 2) * Tau**2 / Sigma.subs(r, y**(1 / 3))**2
dPsi = sp.diff(Psi, y)
ddPsi = sp.diff(Psi, y, y)
Lowlogrho = (1 / sp.sqrt(2 * sp.pi)) * (1 / s) * sp.sqrt(ddPsi + dPsi / x) * sp.exp(-1 / s**2 * Psi)

def logs(order, s_val, nu_val, ns_val, alpha_val):
    lowlogrho = Lowlogrho.subs({s: s_val, nu: nu_val, ns: ns_val, alpha: alpha_val})
    func = sp.utilities.lambdify(y, y**order * lowlogrho, 'numpy') # returns a numpy-ready function
    return quad(func, 0.1, 10)[0]

def effsiglog(sig_val, nu_val, ns_val, alpha_val):
    def func(ss):
        ss = ss[0] if len(ss) else float(ss)
        return logs(0, float(ss), nu_val, ns_val, alpha_val) * logs(2, float(ss), nu_val, ns_val, alpha_val) / logs(1, float(ss), nu_val, ns_val, alpha_val)**2 - 1 - sig_val**2
    solution = fsolve(func, sig_val)
    return solution[0]

# PDF
def RhoPDFns(rho, sigma_val, ns_val, nu_val=21/13, alpha_val=0):
    eff_s = effsiglog(sigma_val, nu_val, ns_val, alpha_val)
    return logs1(eff_s, nu_val, ns_val, alpha_val) / logs0(eff_s, nu_val, ns_val, alpha_val)**2 * Lowlogrho.subs({s: eff_s, nu: nu_val, ns: ns_val, alpha: alpha_val, x: rho * logs1(eff_s, nu_val, ns_val, alpha_val) / logs0(eff_s, nu_val, ns_val, alpha_val)})

def LowRhoBias(rho, slog, nu_val, ns_val, alpha_val):
    return 1 / (Sigma.subs({r: rho**(1 / 3), ns: ns_val, alpha: alpha_val})**2 * slog**2) * Tau.subs({y: rho, nu: nu_val})

def expbiasns(slog, nu_val, ns_val, alpha_val):
    func = sp.utilities.lambdify(x, LowRhoBias(x, slog, nu_val, ns_val, alpha_val) * RhoPDFns(x, slog, ns_val), 'numpy')
    return quad(func, 0.1, 10)[0]

def exprhobiasns(slog, nu_val, ns_val, alpha_val):
    exp_bias = expbiasns(slog, nu_val, ns_val, alpha_val)
    func = sp.utilities.lambdify(x, x * (LowRhoBias(x, slog, nu_val, ns_val, alpha_val) - exp_bias) * RhoPDFns(x, slog, ns_val), 'numpy')
    return quad(func, 0.1, 10)[0]

# bias function
def RhoBiasns(rho, sigma_val, ns_val, nu_val=21/13, alpha_val=0):
    eff_s = effsiglog(sigma_val, nu_val, ns_val, alpha_val)
    exp_bias = expbiasns(eff_s, nu_val, ns_val, alpha_val)
    rho_bias = exprhobiasns(eff_s, nu_val, ns_val, alpha_val)
    return (LowRhoBias(rho, eff_s, nu_val, ns_val, alpha_val) - exp_bias) / rho_bias