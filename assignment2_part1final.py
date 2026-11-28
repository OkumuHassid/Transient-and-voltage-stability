
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.integrate import quad

# =====================================================================
# Parameters (Tables 1-3)
# =====================================================================
X_tr = 0.6                              # transformer reactance [pu]
X_1 = 1.0                               # TL1 [pu]
X_21, X_22, X_23 = 2/12, 5/12, 3/12     # TL2 sections [pu]
Xd_p = 0.3                              # transient reactance X'd [pu]
H = 4.5                                 # inertia constant [s]
f0 = 50.0                               # nominal frequency [Hz]
w0 = 2 * np.pi * f0                     # [rad/s]
E_B = 1.0                               # infinite bus voltage [pu]
E_t = 1.0 * np.exp(1j * np.deg2rad(25)) # terminal voltage (pre-fault) [pu]

# =====================================================================
# 1.2.1  Pre-fault operating point
# =====================================================================
X_2 = X_21 + X_22 + X_23
X_line = X_1 * X_2 / (X_1 + X_2)        # TL1 || TL2
X_tE = X_tr + X_line                    # terminal -> infinite bus
I = (E_t - E_B) / (1j * X_tE)           # line current
E_p = E_t + 1j * Xd_p * I               # internal transient voltage E'
Ep_mag = abs(E_p)
delta0 = np.angle(E_p)                  # [rad]
Pm = (E_t * np.conj(I)).real            # mechanical power = Pe pre-fault

# =====================================================================
# Helper functions 
# =====================================================================
def transfer_reactances(Xa):
    """Transfer reactance pre / during / post fault for a source behind Xa."""
    X_pre = Xa + X_line                             
    X_dur = Xa + X_1 + Xa * X_1 / (X_21 + X_22)      
    X_post = Xa + X_1                               
    return X_pre, X_dur, X_post


def critical_angle(d0, P1, P2):
    """Equal-area criterion: returns (delta_cr, delta_max) in rad."""
    dmax = np.pi - np.arcsin(Pm / P2)
    dcr = np.arccos((Pm * (dmax - d0) + P2 * np.cos(dmax) - P1 * np.cos(d0)) / (P2 - P1))
    return dcr, dmax


def tcr_analytic(d0, dcr, P1):
    """Critical clearing time assuming constant Pa = Pm - P1 sin(d0) during the fault."""
    Pa = Pm - P1 * np.sin(d0)
    return np.sqrt(4 * H * (dcr - d0) / (w0 * Pa))


def simulate(Pmax3, d0, t_fault, t_c, D, t_end=20.0, dt=1e-3):
    """Integrate the swing equation (structure of Listing 1)."""
    P_pre, P_dur, P_post = Pmax3

    def equation_motion(t, x):
        if t <= t_fault:
            Pemax = P_pre                 # pre-fault
        elif t <= t_c:
            Pemax = P_dur                 # during fault
        else:
            Pemax = P_post                # post-fault
        eq = np.zeros(2)
        eq[0] = (1 / (2 * H)) * (Pm - Pemax * np.sin(x[1]) - D * x[0])  # d(dw)/dt
        eq[1] = w0 * x[0]                                                # d(delta)/dt
        return eq

    r = ode(equation_motion).set_integrator("dopri5", max_step=dt)
    r.set_initial_value([0.0, d0], 0.0)
    t, x = [0.0], [[0.0, d0]]
    while r.successful() and r.t < t_end:
        r.integrate(r.t + dt)
        t.append(r.t)
        x.append(r.y.copy())
        if r.y[1] > np.pi:                # beyond 180 deg -> lost synchronism
            break
    x = np.array(x)
    return np.array(t), x[:, 0], x[:, 1]


def tcr_simulation(Pmax3, d0, t_fault, lo=0.0, hi=1.0, tol=1e-4):
    """Critical fault duration by bisection (D = 0)."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        _, _, d = simulate(Pmax3, d0, t_fault, t_fault + mid, 0.0, t_end=3.0)
        if d.max() < np.pi:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =====================================================================
# 1.2.2  Pe,max for the three conditions
# =====================================================================
X_a = Xd_p + X_tr
X_pre, X_dur, X_post = transfer_reactances(X_a)
Pmax = {
    "Pre-fault": Ep_mag * E_B / X_pre,
    "During fault": Ep_mag * E_B / X_dur,
    "Post-fault": Ep_mag * E_B / X_post,
}
Pmax3 = tuple(Pmax.values())
P1, P2 = Pmax["During fault"], Pmax["Post-fault"]

# =====================================================================
# 1.2.3  Critical clearing angle
# =====================================================================
delta_cr, delta_max = critical_angle(delta0, P1, P2)
A_acc = Pm * (delta_cr - delta0) + P1 * (np.cos(delta_cr) - np.cos(delta0))
A_dec = P2 * (np.cos(delta_cr) - np.cos(delta_max)) - Pm * (delta_max - delta_cr)

print("---- 1.2.1 ----")
print(f"E' = {Ep_mag:.3f} pu, delta0 = {np.rad2deg(delta0):.2f} deg, Pm = {Pm:.3f} pu")
print("---- 1.2.2 ----")
for name, X in zip(Pmax, (X_pre, X_dur, X_post)):
    print(f"{name:13s}: X_T = {X:.3f} pu, Pe,max = {Pmax[name]:.3f} pu")
print("---- 1.2.3 ----")
print(f"delta_max = {np.rad2deg(delta_max):.2f} deg, delta_cr = {np.rad2deg(delta_cr):.2f} deg")
print(f"A_acc = {A_acc:.4f}, A_dec = {A_dec:.4f} pu*rad")

# =====================================================================
# Figures for 1.2.2 and 1.2.3
# =====================================================================
delta = np.linspace(0, np.pi, 500)
colors = {"Pre-fault": "tab:blue", "During fault": "tab:red", "Post-fault": "tab:green"}


def draw_curves(ax):
    """Draw the three P-delta curves and the Pm line on ax."""
    for name, P in Pmax.items():
        ax.plot(np.rad2deg(delta), P * np.sin(delta), color=colors[name],
                label=rf"{name}: $P_{{e,max}}$ = {P:.3f} pu")
    ax.axhline(Pm, color="k", linestyle="--", linewidth=1,
               label=rf"$P_m$ = {Pm:.3f} pu")
    ax.set_xlabel(r"Rotor angle $\delta$ [deg]")
    ax.set_ylabel(r"Electrical power $P_e$ [pu]")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.1 * max(Pmax.values()))
    ax.set_xticks(range(0, 181, 30))
    ax.grid(alpha=0.3)


# Figure 1: P-delta curves (1.2.2)
fig1, ax1 = plt.subplots(figsize=(8, 5))
draw_curves(ax1)
ax1.set_title(r"$P$-$\delta$ curves")
ax1.legend(loc="upper right")
fig1.tight_layout()
fig1.savefig("p_delta_curves.png", dpi=300)

# Figure 2: equal-area criterion (1.2.3)
fig2, ax2 = plt.subplots(figsize=(8, 5))
draw_curves(ax2)
d_acc = np.linspace(delta0, delta_cr, 200)
d_dec = np.linspace(delta_cr, delta_max, 200)
ax2.fill_between(np.rad2deg(d_acc), P1 * np.sin(d_acc), Pm,
                 color="tab:red", alpha=0.25, label=rf"$A_\mathrm{{acc}}$ = {A_acc:.3f}")
ax2.fill_between(np.rad2deg(d_dec), Pm, P2 * np.sin(d_dec),
                 color="tab:green", alpha=0.25, label=rf"$A_\mathrm{{dec}}$ = {A_dec:.3f}")
for d, lbl in [(delta0, r"$\delta_0$"), (delta_cr, r"$\delta_{cr}$"),
               (delta_max, r"$\delta_{max}$")]:
    ax2.axvline(np.rad2deg(d), color="gray", linestyle=":", linewidth=1)
    ax2.text(np.rad2deg(d) + 1, 0.02, lbl)
ax2.set_title(rf"Equal-area criterion: $\delta_{{cr}}$ = {np.rad2deg(delta_cr):.1f}$^\circ$")
ax2.legend(loc="upper right", fontsize=9)
fig2.tight_layout()
fig2.savefig("equal_area_criterion.png", dpi=300)

# =====================================================================
# 1.3.1  Simulation, t_clear = 200 ms, D = 0 and D = 2.5
# =====================================================================
t_fault = 0.1                           # fault applied at t = 0.1 s
t_clear = 0.2                           # fault duration [s]

print("---- 1.3.1 ----")
fig3, ax3 = plt.subplots(figsize=(8, 4.5))
for D, c in [(0.0, "tab:blue"), (2.5, "tab:orange")]:
    t, dw, d = simulate(Pmax3, delta0, t_fault, t_fault + t_clear, D)
    ax3.plot(t, np.rad2deg(d), color=c, label=f"D = {D}")
    print(f"D = {D}: delta_peak = {np.rad2deg(d.max()):.2f} deg, "
          f"delta(5 s) = {np.rad2deg(d[-1]):.2f} deg")
ax3.axhline(np.rad2deg(delta_cr), color="gray", ls=":", lw=1, label=r"$\delta_{cr}$")
ax3.set_xlabel("Time [s]")
ax3.set_ylabel(r"Rotor angle $\delta$ [deg]")
ax3.set_title("Rotor angle, fault cleared after 200 ms")
ax3.grid(alpha=0.3)
ax3.legend()
fig3.tight_layout()
fig3.savefig("rotor_angle_200ms.png", dpi=300)

# =====================================================================
# 1.3.2  Critical clearing time (D = 0)
# =====================================================================
tcr_a = tcr_analytic(delta0, delta_cr, P1)
tcr_s = tcr_simulation(Pmax3, delta0, t_fault)
f = lambda d: 1 / np.sqrt(w0 / H * (Pm * (d - delta0) + P1 * (np.cos(d) - np.cos(delta0))))
tcr_exact, _ = quad(f, delta0, delta_cr, limit=200)

print("---- 1.3.2 ----")
print(f"t_cr analytic  = {tcr_a*1e3:.1f} ms")
print(f"t_cr simulated = {tcr_s*1e3:.1f} ms")
print(f"t_cr (energy integral) = {tcr_exact*1e3:.1f} ms")

fig4, ax4 = plt.subplots(figsize=(8, 4.5))
for tc, c, lbl in [(tcr_s - 0.005, "tab:green", "stable"),
                   (tcr_s + 0.005, "tab:red", "unstable")]:
    t, dw, d = simulate(Pmax3, delta0, t_fault, t_fault + tc, 0.0, t_end=10)
    ax4.plot(t, np.rad2deg(d), color=c, label=f"$t_c$ = {tc*1e3:.0f} ms ({lbl})")
ax4.axhline(np.rad2deg(delta_cr), color="gray", ls=":", lw=1, label=r"$\delta_{cr}$")
ax4.set_xlabel("Time [s]")
ax4.set_ylabel(r"Rotor angle $\delta$ [deg]")
ax4.set_title("Clearing just before and just after the critical clearing time (D = 0)")
ax4.grid(alpha=0.3)
ax4.legend()
fig4.tight_layout()
fig4.savefig("critical_clearing_time.png", dpi=300)



# =====================================================================
# 1.3.3  Ideal excitation system: |E_t| = 1 pu constant
# =====================================================================
Et_mag, theta0 = abs(E_t), np.angle(E_t)
Pmax3_avr = tuple(Et_mag * E_B / X for X in transfer_reactances(X_tr))
dcr_avr, dmax_avr = critical_angle(theta0, Pmax3_avr[1], Pmax3_avr[2])
tcr_avr_a = tcr_analytic(theta0, dcr_avr, Pmax3_avr[1])
tcr_avr_s = tcr_simulation(Pmax3_avr, theta0, t_fault)
print("---- 1.3.3 ----")
print("Pe,max pre/dur/post = " + ", ".join(f"{p:.3f}" for p in Pmax3_avr) + " pu")
print(f"delta_cr = {np.rad2deg(dcr_avr):.2f} deg, delta_max = {np.rad2deg(dmax_avr):.2f} deg")
print(f"t_cr analytic = {tcr_avr_a*1e3:.1f} ms, simulated = {tcr_avr_s*1e3:.1f} ms")

fig5, ax5 = plt.subplots(figsize=(8, 4.5))
for tc, c, lbl in [(tcr_avr_s - 0.005, "tab:green", "stable"),
                   (tcr_avr_s + 0.005, "tab:red", "unstable")]:
    t, dw, d = simulate(Pmax3_avr, theta0, t_fault, t_fault + tc, 0.0, t_end=5)
    ax5.plot(t, np.rad2deg(d), color=c, label=f"$t_c$ = {tc*1e3:.0f} ms ({lbl})")
ax5.axhline(np.rad2deg(dcr_avr), color="gray", ls=":", lw=1, label=r"$\delta_{cr}$")
ax5.set_xlabel("Time [s]")
ax5.set_ylabel(r"Angle of $E_t$, $\theta$ [deg]")
ax5.set_title("Ideal excitation: clearing just before and after the critical clearing time (D = 0)")
ax5.grid(alpha=0.3)
ax5.legend()
fig5.tight_layout()
fig5.savefig("critical_clearing_time_avr.png", dpi=300)

plt.show()
