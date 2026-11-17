"""46711 Assignment 2 - Part 2.1: P-V curves (voltage stability)
Lossless line: E -- jX -- V , load P+jQ with Q = P*tan(phi).
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 2.1.1
def Pmax_lossless(X, E, phi):
    """Maximum deliverable power for a lossless line (phi in rad).
    Discriminant of (23) = 0 with Q = P tan(phi):
        Pmax = E^2 (1 - sin phi) / (2 X cos phi)"""
    return E**2 * (1 - np.sin(phi)) / (2 * X * np.cos(phi))

def Vcrit_lossless(E, phi):
    """Voltage at the nose point: Vcr = E / sqrt(2 (1 + sin phi))"""
    return E / np.sqrt(2 * (1 + np.sin(phi)))

# ---------------------------------------------------------------- 2.1.2
def PlotPV(X, E, phi, Pbase, Vbase, fig):
    ax = fig.gca()
    for phi_i in np.atleast_1d(phi):
        ph = np.deg2rad(phi_i)
        # 1 - maximum power
        Pmax = Pmax_lossless(X, E, ph)
        # 2 - P vector, denser near the nose
        P1 = np.linspace(0, Pmax * 9 / 10, 10000, endpoint=False)
        P2 = np.linspace(Pmax * 9 / 10, Pmax * (1.0 - 1e-15), 10000, endpoint=True)
        P = np.concatenate((P1, P2))
        Q = P * np.tan(ph)
        # 3 - upper / lower solutions of eq. (23)
        disc = np.maximum(E**4 / 4 - X**2 * P**2 - E**2 * X * Q, 0)
        Vu = np.sqrt(-X * Q + E**2 / 2 + np.sqrt(disc))
        Vl = np.sqrt(np.maximum(-X * Q + E**2 / 2 - np.sqrt(disc), 0))
        # 4 - plot (one curve: lower branch reversed)
        Pp = np.concatenate((P, P[::-1])) / Pbase
        Vp = np.concatenate((Vu, Vl[::-1])) / Vbase
        ax.plot(Pp, Vp, 'b-', lw=1.5)
        # 5 - label at the nose
        ax.text(Pmax / Pbase * 1.005, Vcrit_lossless(E, ph) / Vbase,
                r'$\phi=%g^\circ$' % phi_i, rotation=90, fontsize=8,
                ha='left', va='center')
    # 6 - locus of critical points, axes
    ph_l = np.deg2rad(np.linspace(-30, 40, 200))
    ax.plot(Pmax_lossless(X, E, ph_l) / Pbase, Vcrit_lossless(E, ph_l) / Vbase,
            'b--', lw=0.9, label='Locus of critical points')
    ax.set_xlabel('Active power, $P$ [pu]')
    ax.set_ylabel('Voltage, $V$ [pu]')
    ax.set_title('PV curves')
    fig.tight_layout()
    ax.margins(x=0, y=0)
    ax.set_ylim([0, E / Vbase * 1.1])
    ax.tick_params(labelsize=10)
    ax.grid(linestyle='-.', linewidth=0.6)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    return fig

E, X, Pbase, Vbase = 135e3, 100.0, 80e6, 135e3
phis = [15, 5, 0, -5, -10]
fig = plt.figure(1, figsize=(6, 5), dpi=300)
PlotPV(X, E, phis, Pbase, Vbase, fig)
ax = fig.gca()
ax.set_xlim(0, 1.5)
# operating points with Z_LD = 150 ohm and 66.67 ohm (phi = 0)
print('--- 2.1.2 operating points (phi = 0) ---')
for Z, col in [(150.0, 'red'), (66.67, 'green')]:
    V = Z * E / np.sqrt(X**2 + Z**2)       # eq. (8), phi=0 -> F = 1+(Z/X)^2
    P = V**2 / Z
    print('ZLD=%.2f ohm: V=%.2f kV = %.4f pu, P=%.2f MW = %.4f pu' % (Z, V/1e3, V/Vbase, P/1e6, P/Pbase))
    ax.plot(P/Pbase, V/Vbase, 'o', color=col, ms=6)
    ax.annotate('$Z_{LD}$=%s$\\Omega\\angle0$\n(P=%.3f, V=%.3f pu)' % (Z, P/Pbase, V/Vbase),
                (P/Pbase, V/Vbase), xytext=(-95 if Z > 100 else -95, 25 if Z > 100 else -35),
                textcoords='offset points', fontsize=7, color=col,
                arrowprops=dict(arrowstyle='-', color=col))
for p in phis:
    print('phi=%4d deg: Pmax=%.4f pu, Vcr=%.4f pu' % (p, Pmax_lossless(X, E, np.deg2rad(p))/Pbase,
          Vcrit_lossless(E, np.deg2rad(p))/Vbase))
ax.legend(loc='lower right', fontsize=8)
fig.savefig('PV_curves_2_1_2.png', dpi=300)

# ---------------------------------------------------------------- 2.1.6
def PV_general(Zline, E, P):
    """Unity-power-factor load (phi=0) behind complex impedance R+jX:
       V^4 + (2RP - E^2)V^2 + |Z|^2 P^2 = 0"""
    R, Xl = Zline.real, Zline.imag
    b = E**2 - 2 * R * P
    d = np.maximum(b**2 - 4 * abs(Zline)**2 * P**2, 0)
    return np.sqrt((b + np.sqrt(d)) / 2), np.sqrt(np.maximum((b - np.sqrt(d)) / 2, 0))

def Pmax_general(Zline, E):
    return E**2 / (2 * (Zline.real + abs(Zline)))

Zb = Vbase**2 / Pbase
cases = [
    ('1) Base case', 1.00 * Vbase, 100j, 'k', '-'),
    ('2) E = 1.05 pu (AVR)', 1.05 * Vbase, 100j, 'tab:blue', '--'),
    ('3) OXL: $E_q$=1.1 pu, $X_s$=30 $\\Omega$', 1.10 * Vbase, 100j + 30j, 'tab:red', '-.'),
    ('4) OXL + lossy line (100$\\Omega\\angle85^\\circ$)', 1.10 * Vbase,
     100 * np.exp(1j * np.deg2rad(85)) + 30j, 'tab:green', ':'),
]
fig2, ax2 = plt.subplots(figsize=(6.5, 5), dpi=300)
print('--- 2.1.6 ---')
offs = {0: [(-135, -75), (25, 12)], 1: [(-110, 25), (15, 28)], 2: [(-135, -45), (30, -30)], 3: [(0, 0), (-10, -120)]}
for i, (name, Ec, Zl, col, ls) in enumerate(cases):
    Pm = Pmax_general(Zl, Ec)
    P = np.concatenate((np.linspace(0, Pm*0.9, 5000, endpoint=False), np.linspace(Pm*0.9, Pm*(1-1e-12), 5000)))
    Vu, Vl = PV_general(Zl, Ec, P)
    ax2.plot(np.concatenate((P, P[::-1]))/Pbase, np.concatenate((Vu, Vl[::-1]))/Vbase,
             color=col, ls=ls, lw=1.6, label=name)
    Vcr = np.sqrt(abs(Zl) * Pm)
    dP = Pm/Pbase - 1
    print('%s: Pmax=%.4f pu, Vcr=%.4f pu, dP=%.4f pu' % (name, Pm/Pbase, Vcr/Vbase, dP))
    ax2.plot(Pm/Pbase, Vcr/Vbase, 's', color=col, ms=6)
    ax2.annotate('Pmax=%.3f, V=%.3f' % (Pm/Pbase, Vcr/Vbase), (Pm/Pbase, Vcr/Vbase),
                 xytext=offs[i][1], textcoords='offset points', fontsize=7, color=col, bbox=dict(fc='white', ec='none', pad=0.5),
                 arrowprops=dict(arrowstyle='-', color=col))
    if Pm >= Pbase:
        V1, _ = PV_general(Zl, Ec, Pbase)
        print('   P=1pu: V=%.4f pu' % (V1/Vbase))
        ax2.plot(1.0, V1/Vbase, 'o', color=col, ms=6)
        ax2.annotate('P=1, V=%.3f' % (V1/Vbase), (1.0, V1/Vbase), xytext=offs[i][0],
                     textcoords='offset points', fontsize=7, color=col, bbox=dict(fc='white', ec='none', pad=0.5),
                     arrowprops=dict(arrowstyle='-', color=col))
    else:
        print('   P=1pu cannot be supplied (Pmax < 1 pu)')
        ax2.plot(1.0, 0.02, 'x', color=col, ms=7)
        ax2.annotate('P=1 pu: no solution', (1.0, 0.02), xytext=(30, 25),
                     textcoords='offset points', fontsize=7, color=col, bbox=dict(fc='white', ec='none', pad=0.5),
                     arrowprops=dict(arrowstyle='-', color=col))
ax2.set_xlabel('Active power, $P$ [pu]'); ax2.set_ylabel('Voltage, $V$ [pu]')
ax2.set_title('PV curves, unity power factor load')
ax2.set_xlim(0, 1.4); ax2.set_ylim(0, 1.25)
ax2.grid(linestyle='-.', linewidth=0.6); ax2.legend(loc='lower left', fontsize=8)
fig2.tight_layout(); fig2.savefig('PV_curves_2_1_6.png', dpi=300)
