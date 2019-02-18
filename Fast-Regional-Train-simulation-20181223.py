import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#-------------------------------------------------------------------------------------
# Constants

# Number of passenger carriages
N_CARRIAGES = 5
# Passenger carriages gross mass [t]
CARRIAGE_GROSS_WEIGHT = 39
# Number of seats for each carriage
CARRIAGE_SEATS = 84
# Passenger gross mass [t]
PASSENGER_GROSS_WEIGHT = 0.070
# Locomotive gross mass [t]
LOCOMOTIVE_GROSS_WEIGHT = 72
# Train gross weight (assume full passenger carriers) [t]
m = LOCOMOTIVE_GROSS_WEIGHT + (CARRIAGE_GROSS_WEIGHT + CARRIAGE_SEATS * PASSENGER_GROSS_WEIGHT) * N_CARRIAGES
# Total mass to be accelerated: Train equivalent mass + passenger mass [t]
me = LOCOMOTIVE_GROSS_WEIGHT * (1 + 0.1) + CARRIAGE_GROSS_WEIGHT * N_CARRIAGES * (1 + 0.03) + CARRIAGE_SEATS * PASSENGER_GROSS_WEIGHT * N_CARRIAGES
# Max traction power of locomotive [kW]
P_MAX = 2200
# Max speed allowed on track [m/s]. Max locomotive and carriage speed is 160 km/h. A lower speed than 150 km/h is chosen 
# since the P control will inevitably overshoot a bit when regulating.
V_MAX = 143/3.6
# Tractive effort chosen [kN]
FT_chosen = 80
# Maximum traction force of locomotive available [kN]
FT_MAX_AVAILABLE = 200
# Auxiliary services power [kW]
P_AUXILIARY = 40 + 30 * N_CARRIAGES
# Number of motorized wheelsets
N_WMOT = 4
# Total number of wheelsets
N_WTOT = 4*N_CARRIAGES + N_WMOT
# Contact line voltage [V]
V_LINE = 3000
# Adhesion coefficient, icy track -> 0.25
FAD_MAX = 0.25
# Maximum traction effort (adhesion test) [kN]
ft_max = FAD_MAX * LOCOMOTIVE_GROSS_WEIGHT * 9.81
# Maximum acceleration (adhesion, confort tests) [m/s^2]
amax = min(1, FT_MAX_AVAILABLE/me, FT_chosen/me)
# Tractive effort applied [kN]
fta = me * amax
# Base speed [m/s]
vb = P_MAX / fta
# Breaking deceleration [m/s^2]
abrk = amax
# Breaking time [s]
tbrk = V_MAX / abrk
# Efficiency during traction
etaT = 0.8
# Efficiency during breaking
etaB = 0.8
# Percentage of electric braking. Assume 1 MW of electrical braking
PERC_E_BRK = 0.3 #1000/(0.5*m*(V_MAX)**2/tbrk)
# Stop time [s]. 2 min.
STOP_TIME = int(2*60)
# Time interval difference considered [s]
delta_t = 1

# Save plots?
SAVE_PLOTS = True
# Save data?
SAVE_DATA = True
# Output folder
OUTPUT_FOLDER = "C:\\Users\\Michy\\Desktop\\Electric Systems For Transportation\\Project\\Transportation project\\results"

#-------------------------------------------------------------------------------
# Print information on data

def print_neat(name, x, unit="", digits=2):
    print(name, np.round(x, digits), unit)

print_neat("Maximum traction power:", P_MAX/1000, "MW")
#print_neat("Maximum speed allowed:", V_MAX * 3.6, "km/h")
print_neat("Line voltage:", V_LINE/1000, "kV DC")
print_neat("Number of passanger carriages:", N_CARRIAGES, digits=0)
print_neat("Auxiliary power required:", P_AUXILIARY, "kW")
print_neat("Gross mass convoy:", m, "t", digits=0)
print_neat("Equivalent mass convoy:", me, "t", digits=0)
print_neat("Maximum acceleration allowed:", amax, "m/s^2")
#print_neat("Maximum starting tractive effort (static adhesion test):", ft_max, "kN")
print_neat("Starting tractive effort applied:", fta, "kN")
print_neat("Base speed:", vb * 3.6, "km/h")
print_neat("Braking deceleration:", abrk, "m/s^2")
print_neat("Percentage of electric braking", PERC_E_BRK*100, "%")

#-------------------------------------------------------------------------------

# Distance needed to stop the train travelling at speed v [m]
def braking_distance(v):
    # v is in m/s
    # returns value in m
    return 0.5 * v**2 / abrk

# Adhesion coeff as a function of v in km/h
def fad_v(v):
    return FAD_MAX/(1 + 0.011*v)

# Returns resistance Re = (+-i + rc)*m*g [N]
def Re(current_d):
    # Grade resistance
    if current_d < 14000:
        alpha = np.arctan(2/14000)
    elif current_d < 26000:
        alpha = np.arctan(26/12000)
    elif current_d < 41300:
        alpha = np.arctan(30/16000)
    elif current_d < 65900:
        alpha = np.arctan(116/24600)
    else:
        alpha = 0

    # Curvature resistance
    if current_d > 1450 and current_d < 4220:
        rc = 0.65/(3500 - 55)
    elif current_d > 5900 and current_d < 7920:
        rc = 0.65/(4500 - 55)
    elif current_d > 10200 and current_d < 12900:
        rc = 0.65/(2400 - 55)
    elif current_d > 23200 and current_d < 25190:
        rc = 0.65/(1200 - 55)
    elif current_d > 26120 and current_d < 26870:
        rc = 0.65/(2000 - 55)
    elif current_d > 31150 and current_d < 33340:
        rc = 0.65/(2500 - 55)
    elif current_d > 42000 and current_d < 43000:
        rc = 0.65/(1000 - 55)
    elif current_d > 49000 and current_d < 51000:
        rc = 0.65/(1800 - 55)
    elif current_d > 53000 and current_d < 54000:
        rc = 0.65/(2400 - 55)
    elif current_d > 54400 and current_d < 564000:
        rc = 0.65/(1200 - 55)
    elif current_d > 57000 and current_d < 58500:
        rc = 0.65/(1000 - 55)
    elif current_d > 59500 and current_d < 60750:
        rc = 0.65/(1200 - 55)
    elif current_d > 62500 and current_d < 64700:
        rc = 0.65/(1750 - 55)
    else:
        rc = 0
    
    # Re = (+-i + rc)*m*g
    return m*9.81*(np.sin(alpha) + rc)*1000

# Returns resistance R0 [N]
def R0(v):
    # v is in km/h
    # mg is in kN so m is in t
    return (1.625 + 0.0205 * (v/10)**2) * m * 9.81

# Total resistance R0 + Re [N]. v is in km/h
def total_R(v, current_d):
    return R0(v) + Re(current_d = current_d)

# Returns tractive effort applied [kN]. v in m/s.
def tractive_effort(v, current_d, coasting = False, previous_ft = 0):
    
    # Before coasting: reduce torque gradually -3 % each second if FT > 1 kN
    # If FT is < 1 kN then set it to 0.
    if coasting:
        if previous_ft > 1:
            return previous_ft * 0.90
        else:
            return 0
    # Else if speed is lower then V_MAX return fta if v < vbase else return P_MAX/v
    elif v < V_MAX:
        if v < vb:
            # Constant torque
            return fta
        
        else:
            # Constant power
            return P_MAX/v
    # Else if v==V_MAX then we only need to account for curves and gradient.
    # -> Use a P regulator
    else:
        error = total_R(v * 3.6, current_d)/1000 - previous_ft
        return previous_ft*(1 + 0.003*error)

# Braking force applied by locomotive or carriages
def brk_force(x_FT, carriages=False):
    if x_FT < 0:
        if carriages:
            return x_FT * (1 - PERC_E_BRK) / PERC_E_BRK
        else:
            return x_FT
    return 0

#-------------------------------------------------------------------------------
# Start

# Speed [m/s]
v_ms = [0]
# Speed [km/h]
v_kmh = [0]
# Position on track [m]
d = [0]
# Traction force [kN]
FT_ = [fta]
# Acceleration m/s^2
a = [amax]
# Electric power [kW]
Pel = [P_AUXILIARY]
# Electrical energy [kWh]
E = [P_AUXILIARY * delta_t / 3600]
# Current [A]
I = [Pel[0] / 1000 / V_LINE]
# Index
i = 0

# Acceleration and coasting phase
def acceleration_coasting(length_track, d0, v_ms, v_kmh, d, FT_, a, Pel, E, I, i):
    coasting = False
    while (d[i] - d0) < (length_track - braking_distance(v_ms[i])):
        
        v_ms.append(a[i] * delta_t + v_ms[i])
        v_kmh.append(v_ms[i + 1] * 3.6)
        d.append((v_ms[i + 1] + v_ms[i]) / 2 * delta_t + d[i])
        FT_.append(tractive_effort(v_ms[i + 1], d[i + 1], coasting=coasting, previous_ft=FT_[i]))
        a.append((FT_[i + 1] - total_R(v_kmh[i + 1], d[i + 1])/1000)/me)
        Pel.append(P_AUXILIARY + v_ms[i + 1] * FT_[i + 1] / etaT)
        E.append(Pel[i + 1] * delta_t / 3600 + E[i])
        I.append(Pel[i + 1] * 1000 / V_LINE)
        i += 1

        # Coasting lock-in
        ################################################
        if (d[i] - d0) > (length_track - 7000) and not coasting:
            coasting = True
        ################################################

    return v_ms, v_kmh, d, FT_, a, Pel, E, I, i

# Braking phase
def brake(v_ms, v_kmh, d, FT_, a, Pel, E, I, i):
    while v_ms[i] > 0 :
        v_ms.append(a[i] * delta_t + v_ms[i])
        v_kmh.append(v_ms[i + 1]*3.6)
        d.append((v_ms[i + 1] + v_ms[i])/2 * delta_t + d[i])
        a.append(-abrk)
        # Only part of the braking torque is electric
        FT_.append((me*a[i + 1] + total_R(v_kmh[i + 1], d[i + 1])/1000) * PERC_E_BRK)
        Pel.append(P_AUXILIARY + v_ms[i + 1] * FT_[i + 1] * etaB)
        E.append(Pel[i] * delta_t / 3600 + E[i])
        I.append(Pel[i] * 1000/V_LINE)
        i += 1
    return v_ms, v_kmh, d, FT_, a, Pel, E, I, i

# Stop at a station
def stop_at_station(v_ms, v_kmh, d, FT_, a, Pel, E, I, i):
    
    for k in range(STOP_TIME):
        a.append(0)
        v_ms.append(0)
        v_kmh.append(0)
        d.append(d[i])
        FT_.append(0)
        Pel.append(P_AUXILIARY)
        E.append(Pel[i + 1] * delta_t / 3600 + E[i])
        I.append(Pel[i + 1] * 1000 / V_LINE)
        i += 1
    return v_ms, v_kmh, d, FT_, a, Pel, E, I, i

#---------------------------------------------------------------------------------------
# Start from Pavia. Length of path PAVIA-VOGHERA [m]
PAVIA_VOGHERA = 26000
# Acceleration-coasting
v_ms, v_kmh, d, FT_, a, Pel, E, I, i  = acceleration_coasting(length_track = PAVIA_VOGHERA, d0=0, v_ms=v_ms, v_kmh=v_kmh, d=d, FT_=FT_, a=a, Pel=Pel, E=E, I=I, i=i)
# Now braking...
#print_neat("BRAKE!!!", braking_distance(v_ms[i]), "m")
v_ms, v_kmh, d, FT_, a, Pel, E, I, i = brake(v_ms=v_ms, v_kmh=v_kmh, d=d, FT_=FT_, a=a, Pel=Pel, E=E, I=I, i=i)
# Stop at Voghera 2 minutes
v_ms, v_kmh, d, FT_, a, Pel, E, I, i = stop_at_station(v_ms=v_ms, v_kmh=v_kmh, d=d,FT_=FT_, a = a, Pel=Pel, E=E, I=I, i=i)

#-------------------------------------------------------------------------------------------
# Now from Voghera to Tortona

# Length path from VOGHERA to TORTONA [m]
VOGHERA_TORTONA = 15300
# Starting position [m]
d0 = d[i]
# Start
v_ms, v_kmh, d, FT_, a, Pel, E, I, i  = acceleration_coasting(length_track = VOGHERA_TORTONA, d0=d0, i=i, v_ms=v_ms, v_kmh = v_kmh, d=d, FT_=FT_, a=a, Pel=Pel, E=E, I=I)
# Now braking
v_ms, v_kmh, d, FT_, a, Pel, E, I, i = brake(v_ms=v_ms, v_kmh=v_kmh, d=d, FT_=FT_, a=a, Pel=Pel, E=E, I=I, i=i)
# Stop at Tortona 2 minutes
v_ms, v_kmh, d, FT_, a, Pel, E, I, i = stop_at_station(v_ms=v_ms, v_kmh=v_kmh, d=d,FT_=FT_, a = a, Pel=Pel, E=E, I=I, i=i)

#-------------------------------------------------------------------------------------------
# Now from Tortona to Arquata

# Length path from TORTONA to ARQUATA [m]
TORTONA_ARQUATA = 24600
# Starting position [m]
d0 = d[i]
# Start
v_ms, v_kmh, d, FT_, a, Pel, E, I, i  = acceleration_coasting(length_track = TORTONA_ARQUATA, d0=d0, i=i, v_ms=v_ms, v_kmh = v_kmh, d=d, FT_=FT_, a=a, Pel=Pel, E=E, I=I)
# Now braking
v_ms, v_kmh, d, FT_, a, Pel, E, I, i = brake(v_ms=v_ms, v_kmh=v_kmh, d=d, FT_=FT_, a=a, Pel=Pel, E=E, I=I, i=i)
# Stop at Arquata 2 minutes (this will not be considered for commercial speed!)
# It is used for having nicer graphs.
v_ms, v_kmh, d, FT_, a, Pel, E, I, i = stop_at_station(v_ms=v_ms, v_kmh=v_kmh, d=d,FT_=FT_, a = a, Pel=Pel, E=E, I=I, i=i)

#-------------------------------------------------------------------------------------------
# Print info

print("Commercial speed:", np.round((PAVIA_VOGHERA + VOGHERA_TORTONA + TORTONA_ARQUATA)/(delta_t*(len(v_kmh) - 120))*3.6, 2), "km/h")
print("Total travelled time (including stops):", np.round((i - 120)/60, 2), "min")
print("Expected travel time according to Trenitalia App: 41 min")

#-------------------------------------------------------------------------------------------
# Plot figures

def plot_better(y, label, color, subplot=None, save=False, savename="plot", newfigure=False, title=None):
    # If a new figure is required or plot needs to be saved
    if newfigure or save:
        plt.figure()
    # If no subplot is required
    if subplot is not None:
        plt.subplot(subplot[0], subplot[1], subplot[2])
        if type(y[0]) == list:
            for i, k in enumerate(y):
                plt.plot(k, label=label[i])
        else:
            plt.plot(y, label=label, color=color)
    else:
        if type(y[0]) == list:
            for i, k in enumerate(y):
                plt.plot(k, label=label[i])
        else:
            plt.plot(y, label=label, color=color)
    # Add grid and legend
    plt.grid()
    plt.legend()
    # Add title if present
    if title is not None:
        plt.title(title)
    # Save if required
    if save:
        plt.xlabel("Time [s]")
        plt.savefig(f"{OUTPUT_FOLDER}\\{savename}.png")

# Plot everything
plot_better(a, "a [m/s^2]", "green", subplot=(3,3,1), newfigure=True)
plot_better(v_kmh, "v [km/h]", "blue", subplot=(3,3,2))
plot_better([x/1000 for x in d], "d [km]", "blue", subplot=(3,3,3))
plot_better(Pel, "Pel [kW]", "blue", subplot=(3,3,4))
plot_better([FT_,
             [fad_v(x) * LOCOMOTIVE_GROSS_WEIGHT * 9.81 for x in v_kmh],
             [abs(x) for x in FT_]],
    ["FT [kN]", "fad [kN]", "|FT| [kN]"], "blue", subplot=(3,3,5))
plot_better(E, "E [kWh]", "blue", subplot=(3,3,6))
plot_better(I, "I [A]", "blue", subplot=(3,3,7))
plot_better([[R0(x)/(9.81*me) for x in v_kmh],
              [R0(x)/1000 for x in v_kmh],
              [Re(x)/1000 for x in d],
              [total_R(y, x)/1000 for x, y in zip(d, v_kmh)]],
    ["R0 [N/kN]", "R0 [kN]", "Re [kN]", "R [kN]"], "blue", subplot=(3,3,8))
plot_better([[brk_force(x) for x in FT_],
              [brk_force(x, carriages=True,) for x in FT_],
              [-fad_v(x) * LOCOMOTIVE_GROSS_WEIGHT * 9.81 for x in v_kmh],
              [-fad_v(x) * (m - LOCOMOTIVE_GROSS_WEIGHT) * 9.81 for x in v_kmh]],
["FT locomotive [kN]", "FT carriages [kN]", "fad locomotive [kN]", "fad carriages [kN]"], "blue", subplot=(3,3,9))

# Show plots
plt.show()

#-------------------------------------------------------------------------------------------
# Save plots

if SAVE_PLOTS:
    plot_better(a, "a [m/s^2]", "green", save=True, savename="A")
    plot_better(v_kmh, "v [km/h]", "blue", save=True, savename="V")
    plot_better([x/1000 for x in d], "d [km]", "blue", save=True, savename="d")
    plot_better(Pel, "Pel [kW]", "blue", save=True, savename="Pel")
    plot_better([FT_,
             [fad_v(x) * LOCOMOTIVE_GROSS_WEIGHT * 9.81 for x in v_kmh],
             [abs(x) for x in FT_]],
        ["FT [kN]", "fad [kN]", "|FT| [kN]"], "blue", save=True, savename="FT")
    plot_better(E, "E [kWh]", "blue", save=True, savename="E")
    plot_better(I, "I [A]", "blue", save=True, savename="I")
    plot_better([[R0(x)/1000 for x in v_kmh],
              [Re(x)/1000 for x in d],
              [total_R(y, x)/1000 for x, y in zip(d, v_kmh)]],
        ["R0 [kN]", "Re [kN]", "R [kN]"], "blue", save=True, savename="R")
    plot_better([[brk_force(x) for x in FT_],
              [brk_force(x, carriages=True,) for x in FT_],
              [-fad_v(x) * LOCOMOTIVE_GROSS_WEIGHT * 9.81 for x in v_kmh],
              [-fad_v(x) * (m - LOCOMOTIVE_GROSS_WEIGHT) * 9.81 for x in v_kmh]],
        ["FT locomotive [kN]", "FT carriages [kN]", "fad locomotive [kN]", "fad carriages [kN]"],
        "blue",save=True, savename="ad", title="Force applied during braking phase")
    
    plt.figure()
    plt.plot(v_kmh[0:500], FT_[0:500], label="FT [kN]")
    plt.xlabel("v [km/h]")
    plt.legend()
    plt.grid()
    plt.savefig(f"{OUTPUT_FOLDER}\\FT_v.png")

#-------------------------------------------------------------------------------------------
# Save data

if SAVE_DATA:
    # Output data frame
    df = pd.DataFrame({'t [s]': [x for x in range(i + 1)],
                       'a [m/s^2]': a,
                       'v [km/h]': v_kmh,
                       'd [km]': [x/1000 for x in d],
                       'Pel [kW]': Pel,
                       'FT [kN]': FT_,
                       'E [kWh]': E,
                       'I [A]': I,
                       'R0 [N/kN]': [R0(x)/(9.81*me) for x in v_kmh],
                       'R0 [kN]': [R0(x)/1000 for x in v_kmh],
                       'Re [kN]': [Re(x)/1000 for x in d],
                       'R [kN]': [total_R(y, x)/1000 for x, y in zip(d, v_kmh)]})
    # Output csv
    df.to_csv(f"{OUTPUT_FOLDER}\\track-data-simulated.csv", index = False)


