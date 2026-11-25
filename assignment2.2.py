# reference: used the provided power flow program Simple_PowerFlow_46711.py to run the power flow

# import libraries
import numpy as np
import matplotlib.pyplot as plt

# provided power flow function from Simple_PowerFlow_46711.py
def PowerFlowNewton(Ybus,Sbus,V0,buscode,max_iter,err_tol):
    success = 0; n = 0; V = V0;  SVEC = Sbus - V * (Ybus.dot(V)).conj(); PVEC = np.real(SVEC); Delta_Q = np.imag(SVEC)
    pv_index = np.where(buscode==2)[0]; pq_index = np.where(buscode==1)[0]
    F = np.concatenate((PVEC[pv_index],PVEC[pq_index], Delta_Q[pq_index]),axis=0); pvpq_ind = np.append(pv_index,pq_index)
    success = 0 if  np.linalg.norm(F,np.inf)> err_tol else 1   
    while (not success) and (n < max_iter):
        n += 1; J_dS_dVm = np.diag(V/np.absolute(V)).dot(np.diag((Ybus.dot(V)).conj())) + \
        np.diag(V).dot(Ybus.dot(np.diag(V/np.absolute(V))).conj()); J_dS_dTheta = 1j*np.diag(V).dot((np.diag(Ybus.dot(V))-Ybus.dot(np.diag(V))).conj())
        dx = np.linalg.solve(np.block([[np.real(J_dS_dTheta[np.ix_(pvpq_ind,pvpq_ind)]) ,np.real(J_dS_dVm[np.ix_(pvpq_ind,pq_index)])],[np.imag(J_dS_dTheta[np.ix_(pq_index,pvpq_ind)]),np.imag(J_dS_dVm[np.ix_(pq_index,pq_index)])]]),F) 
        N1 = 0; Xgf = np.absolute(V);  N2 = len(pv_index); Pkf = np.angle(V); N3 = N2; N4 = N3 + len(pq_index); N5 = N4; N6 = N5 + len(pq_index)
        if len(pv_index) > 0:
            Pkf[pv_index] += dx[N1:N2]
        if len(pq_index) > 0:
            Pkf[pq_index] += dx[N3:N4]; Xgf[pq_index] += dx[N5:N6]
        V = Xgf * np.exp(1j*Pkf)
        SVEC = Sbus - V * (Ybus.dot(V)).conj(); PVEC = np.real(SVEC); Delta_Q = np.imag(SVEC)
        F = np.concatenate((PVEC[pv_index],PVEC[pq_index], Delta_Q[pq_index]),axis=0)
        success = 0 if  np.linalg.norm(F,np.inf)> err_tol else 1   
    if success:
        print('The Newton Raphson Power Flow Converged in %d iterations!\n\n' % (n,))
    else:
        print('!!! No Convergence !!!\n Stopped after %d iterations without solution...' % (n,))
    return V,success,n

# set the power flow parameters given in the provided file
max_iter = 30
err_tol = 1e-4

# define the system parameters for the PQ bus system, the line is j0.5 pu and the sending end voltage is 1 pu
X = 0.5
E = 1

# construct the Ybus
y_line = 1 / (1j * X)
Ybus = np.array([[y_line, -y_line], [-y_line, y_line]])

# define the bus types (1 = PQ, 2 = PV, 3 = slack), bus 1 is the slack and bus 2 is the load
buscode = np.array([3, 1])

# define the initial guess for the voltages, V2 = 1 pu at 0 degrees
Vo = np.array([1 + 0j, 1 + 0j])

# define the active power of the load and the three reactive powers
P_load = 1
Q_cases = [-0.15, 0, 0.15]

# Q2.2.1 for loop to run the power flow for each of the loads
print("Q2.2.1")

for Q_load in Q_cases:
    print(f"P = {P_load} pu, Q = {Q_load} pu")

    # define the bus injections, the load is a negative injection at bus 2
    Sbus = np.array([0, -(P_load + 1j * Q_load)])

    # run the power flow
    V, success, n = PowerFlowNewton(Ybus, Sbus, Vo, buscode, max_iter, err_tol)

    # print the receiving end voltage and the number of iterations if it converged
    if success:
        print(f"V2 = {np.abs(V[1])} pu\nIterations = {n}\n")

# Q2.2.2 plot the P-V curves for the three loads
print("Q2.2.2")

# define the colours of the curves
colors = ["blue", "green", "purple"]

plt.figure(figsize = (6,4))

# for loop to plot the curve of each of the loads
for i in range(3):
    # find the load angle
    phi = np.arctan(Q_cases[i] / P_load)

    # find the maximum power
    Pmax = E**2 * (1 - np.sin(phi)) / (2 * X * np.cos(phi))
    print(f"Q = {Q_cases[i]}: phi = {np.rad2deg(phi)} degrees, Pmax = {Pmax} pu")

    # create the P vector and find Q using the load angle
    P = np.linspace(0, Pmax * 0.99999, 5000)
    Q = P * np.tan(phi)

    # find the upper and lower voltage using equation (23)
    V_upper = np.sqrt(-X * Q + E**2 / 2 + np.sqrt(E**4 / 4 - X**2 * P**2 - E**2 * X * Q))
    V_lower = np.sqrt(-X * Q + E**2 / 2 - np.sqrt(E**4 / 4 - X**2 * P**2 - E**2 * X * Q))

    # plot the curve
    plt.plot(P, V_upper, color = colors[i], label = f"Q = {Q_cases[i]} pu")
    plt.plot(P, V_lower, color = colors[i])

# plot the locus of critical points, the nose (Pmax, Vcr) of the P-V curve for a range of load angles
phi_locus = np.deg2rad(np.linspace(-15, 89, 500))
P_locus = E**2 * (1 - np.sin(phi_locus)) / (2 * X * np.cos(phi_locus))
V_locus = E / np.sqrt(2 * (1 + np.sin(phi_locus)))
plt.plot(P_locus, V_locus, color = "black", linestyle = ":", label = "Locus of critical points")

# plot the value of P as a vertical line
plt.axvline(P_load, color = "red", linestyle = "--", label = "P = 1 pu")

# mark the operating point (upper solution) at P = 1 pu for Q = -0.15 pu and Q = 0 pu
for Q_op in Q_cases[:2]:
    V_op = np.sqrt(-X * Q_op + E**2 / 2 + np.sqrt(max(E**4 / 4 - X**2 * P_load**2 - E**2 * X * Q_op, 0)))
    plt.plot(P_load, V_op, "o", color = "red")

plt.xlabel("Active power, P [pu]")
plt.ylabel("Voltage, V [pu]")
# plt.title("PV curves for the three loads")
plt.grid(True)
plt.legend(fontsize = 8)
plt.savefig("PV_curves.png", dpi = 300)
# plt.show()

# Q2.2.3 the load that did not converge is P = 1 pu and Q = 0.15 pu
# it can be supplied if Pmax >= P, which is E^4/4 - X^2*P^2 - E^2*X*Q >= 0
Q_load = 0.15

# find the E needed
E_needed = np.sqrt(2 * X * (Q_load + np.sqrt(Q_load**2 + P_load**2)))

# find the X needed
X_needed = E**2 * (np.sqrt(Q_load**2 + P_load**2) - Q_load) / (2 * P_load**2)

# find the Q needed
Q_needed = (E**4 / 4 - X**2 * P_load**2) / (E**2 * X)

print("\nQ2.2.3")
print(f"E must be at least {E_needed} pu")
print(f"X must be at most {X_needed} pu")
print(f"Q must be at most {Q_needed} pu, so at least {Q_load - Q_needed} pu of reactive power compensation is needed\n")

# define the three measures to test, increase E to 1.1, decrease X to 0.4, and compensation 0.2 so the net Q is -0.05
E_measures = [1.1, 1, 1]
X_measures = [0.5, 0.4, 0.5]
Q_measures = [0.15, 0.15, -0.05]

# for loop to run the power flow with each of the measures
for i in range(3):
    print(f"E = {E_measures[i]} pu, X = {X_measures[i]} pu, Q = {Q_measures[i]} pu")

    # construct the Ybus and the bus injections
    y_measure = 1 / (1j * X_measures[i])
    Ybus_measure = np.array([[y_measure, -y_measure], [-y_measure, y_measure]])
    Sbus = np.array([0, -(P_load + 1j * Q_measures[i])])

    # the sending end voltage is changed in the initial guess
    Vo_measure = np.array([E_measures[i] + 0j, 1 + 0j])

    # run the power flow
    V, success, n = PowerFlowNewton(Ybus_measure, Sbus, Vo_measure, buscode, max_iter, err_tol)

    if success:
        print(f"V2 = {np.abs(V[1])} pu\nIterations = {n}\n")

# define the system parameters for the PV bus system, the line is 1 pu at 85 degrees
Z_line = 1 * np.exp(1j * np.deg2rad(85))

# construct the Ybus
y_lossy = 1 / Z_line
Ybus_PV = np.array([[y_lossy, -y_lossy], [-y_lossy, y_lossy]])

# define the bus types, bus 1 is the sending end (PV) and bus 2 is the receiving end (reference)
buscode = np.array([2, 3])

# define the sending and receiving end voltages of the three cases
Es_values = [1.0, 1.0, 0.9128]
Er_values = [1.0, 0.9128, 1.0]

# Q2.2.4 for loop to run the power flow for each of the cases
print("Q2.2.4")

for i in range(3):
    print(f"Es = {Es_values[i]} pu, Er = {Er_values[i]} pu")

    # define the initial guess, 0 degrees with the given voltage magnitudes
    Vo = np.array([Es_values[i] + 0j, Er_values[i] + 0j])

    # define the bus injections, the sending end produces 1 pu
    Sbus = np.array([1 + 0j, 0])

    # run the power flow
    V, success, n = PowerFlowNewton(Ybus_PV, Sbus, Vo, buscode, max_iter, err_tol)

    # the receiving end is the reference at 0 degrees so delta is the angle of the sending end
    if success:
        print(f"delta = {np.angle(V[0], deg = True)} degrees\nIterations = {n}\n")

# Q2.2.5 find the maximum sending end power of each of the cases
# Ps = Es^2*cos(85)/Z - Es*Er*cos(85 + delta)/Z, the maximum is when cos(85 + delta) = -1
print("Q2.2.5")

for i in range(3):
    Ps_max = Es_values[i]**2 * np.cos(np.deg2rad(85)) + Es_values[i] * Er_values[i]

    print(f"Es = {Es_values[i]}, Er = {Er_values[i]}: Ps,max = {Ps_max} pu")

# plot the P-delta curves of the three cases
# Ps = Es^2*cos(theta)/Z - Es*Er*cos(theta + delta)/Z, where Z and theta are the magnitude and angle of the line impedance
Z = np.abs(Z_line)
theta = np.angle(Z_line)
Ps = 1
delta = np.deg2rad(np.linspace(0, 180, 1000))

plt.figure(figsize = (6,4))

for i in range(3):
    Es, Er = Es_values[i], Er_values[i]
    P_delta = Es**2 * np.cos(theta) / Z - Es * Er * np.cos(theta + delta) / Z
    plt.plot(np.rad2deg(delta), P_delta, color = colors[i], label = f"Es = {Es} pu, Er = {Er} pu")

    # mark the intersection with Ps = 1 pu on the rising side of the curve (the solution found by the power flow)
    # the curve counts as reaching Ps if Ps,max is within the power flow tolerance of Ps
    Ps_max = Es**2 * np.cos(theta) / Z + Es * Er / Z
    if Ps_max >= Ps - err_tol:
        delta_op = np.arccos(np.clip((Es**2 * np.cos(theta) - Ps * Z) / (Es * Er), -1, 1)) - theta
        plt.plot(np.rad2deg(delta_op), Ps, "o", color = "red")

# plot the value of Ps as a horizontal line
plt.axhline(Ps, color = "red", linestyle = "--", label = "Ps = 1 pu")
plt.xlabel(r"Power angle, $\delta$ [degrees]")
plt.ylabel("Active power, Ps [pu]")
plt.grid(True)
plt.legend(fontsize = 8)
plt.savefig("P_delta_curves_q2_2.png", dpi = 300)