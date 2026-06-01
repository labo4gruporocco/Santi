import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import time 
import os 
from datetime import datetime  
import glob as glob
from scipy.optimize import curve_fit
from matplotlib.ticker import LogLocator, LogFormatter

# =============================================================================
# ESTILOS DEL PÓSTER (GRAFICO_POSTER) APLICADOS GLOBALES
# =============================================================================
plt.rcParams.update({
    # ---- FIGURA ----
    "figure.figsize": (7, 5),
    "figure.dpi": 300,           # High DPI requerido para póster

    # ---- FUENTES ----
    "font.size": 14,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],

    # ---- EJES ----
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "axes.grid": True,
    "axes.linewidth": 1.5,
    "axes.spines.top": False,    # Sin caja de ploteo (borde superior)
    "axes.spines.right": False,  # Sin caja de ploteo (borde derecho)

    # ---- GRID ----
    "grid.color": "gray",        # Grilla gris
    'grid.alpha': 0.3,           # Grilla suave/clara
    "grid.linestyle": "--",

    # ---- TICKS ----
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.direction": "in",
    "ytick.direction": "in",

    # ---- LÍNEAS y MARCADORES ----
    "lines.linewidth": 3,        # Líneas un poco más gruesas
    "lines.markersize": 8,       # Puntos redondos más grandes

    # ---- LEYENDA ----
    "legend.fontsize": 12,
    "legend.frameon": False,     # Típicamente sin caja en la leyenda para pósters

    # ---- GUARDADO ----
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# =============================================================================
# EXTRACCIÓN Y SEGMENTACIÓN DE DATOS 1
# =============================================================================
file_list_c_1 = sorted(glob.glob("13h_44m_16s/*.pkl"))

frequencies_c_1 = []
v1_c_1 = []
v2_c_1 = []

for file in file_list_c_1:
    # -------- extraer frecuencia del nombre --------
    filename = os.path.basename(file)
    freq = float(filename.split("_")[1].replace("Hz.pkl", ""))
    frequencies_c_1.append(freq)

    # -------- cargar datos --------
    with open(file, "rb") as f:
        data = pickle.load(f)
        # convertir strings a float
        v1_c_1.append(float(data['v1']))
        v2_c_1.append(float(data['v2']))

segments_c_1 = [
    (150140,150170, 0.5 ),
    (248275,248295, 0.7),
    (427100,427550, 1.2),
    (558640,560670,20), 
    (648850,650580,30)
]

# convertir a arrays
frequencies_c_1 = np.array(frequencies_c_1)
v1_c_1 = np.array(v1_c_1)
v2_c_1 = np.array(v2_c_1)

# lista donde guardar resultados
segmented_data_c_1 = []

for start, stop, step in segments_c_1:
    mask = (frequencies_c_1 >= start) & (frequencies_c_1 <= stop)
    
    seg_freq = frequencies_c_1[mask]
    seg_v1 = v1_c_1[mask]
    seg_v2 = v2_c_1[mask]
    
    segmented_data_c_1.append({
        "range": (start, stop),
        "freq": seg_freq,
        "v1": seg_v1,
        "v2": seg_v2
    })

# -------- modelo lorentziano --------
def lorentzian(f, f0, gamma, A, offset):
    return offset + A / (1 + ((f - f0)/gamma)**2)

# =============================================================================
# AJUSTE Y PLOTEO 1
# =============================================================================
f = np.array(segmented_data_c_1[0]["freq"])   
V2 = np.array(segmented_data_c_1[0]["v2"]) 
V1 = np.array(segmented_data_c_1[0]["v1"])     
T = V2/V1 
sigma_V2 = 0.03 * V2 + 0.001
sigma_V1 = 0.03 * V1 + 0.001

sigma_T = T * np.sqrt((sigma_V2 / V2)**2 +(sigma_V1 / V1)**2)

mask = (f > 150150.5) & (f < 150168)   

f = f[mask]
T = T[mask]
sigma_T = sigma_T[mask]

# -------- estimaciones iniciales --------
f0_guess = f[np.argmax(T)]
A_guess = max(T) - min(T)
gamma_guess = (max(f) - min(f)) / 10
offset_guess = min(T)

p0 = [f0_guess, gamma_guess, A_guess, offset_guess]

# -------- ajuste --------
popt, pcov = curve_fit(
    lorentzian, f, T, p0=p0, sigma=sigma_T, absolute_sigma=True, maxfev=10000
)

f0, gamma, A, offset = popt
perr = np.sqrt(np.diag(pcov))
df0, dgamma, dA, doffset = perr

# -------- plot --------
f_fit = np.linspace(min(f), max(f), 1000)
T_fit = lorentzian(f_fit, *popt)

plt.errorbar(
    f/1000, T,
    yerr=sigma_T,
    fmt='o',
    color='#1f77b4',       # Datos en azul
    ecolor='#7f7f7f',      # Barras de error en gris highlight
    elinewidth=1.5,
    capsize=4,
    capthick=1.5,
    alpha=0.9,
    label="Datos"
)
plt.plot(f_fit/1000, T_fit, '-', color='#d62728', label="Fit Lorentziano") # Fit en rojo
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Voltaje")
plt.yscale('log')
ax = plt.gca()
ax.ticklabel_format(style='plain', axis='x', useOffset=False)
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
ax.yaxis.set_major_formatter(LogFormatter(base=10.0))
plt.legend()
plt.show()

f_res = f0
df_res = df0
Tmax = lorentzian(f0, *popt)
dTmax = np.sqrt(dA**2 + doffset**2)
FWHM = 2 * gamma
dFWHM = 2 * dgamma
Q = f0 / FWHM
dQ = Q * np.sqrt((df0/f0)**2 + (dFWHM/FWHM)**2)

print(f"f0 = {f0:.2f} ± {df0:.2f} Hz")
print(f"FWHM = {FWHM:.2f} ± {dFWHM:.2f} Hz")
print(f"Q = {Q:.2f} ± {dQ:.2f}")
print(f"Tmax = {Tmax:.3f} ± {dTmax:.3f} V")

# =============================================================================
# EXTRACCIÓN Y SEGMENTACIÓN DE DATOS 2
# =============================================================================
file_list_c = sorted(glob.glob("13h_55m_38s/*.pkl"))

frequencies_c = []
v1_c = []
v2_c = []

for file in file_list_c:
    filename = os.path.basename(file)
    freq = float(filename.split("_")[1].replace("Hz.pkl", ""))
    frequencies_c.append(freq)

    with open(file, "rb") as f:
        data = pickle.load(f)
        v1_c.append(float(data['v1']))
        v2_c.append(float(data['v2']))

segments_c = [
    (248275,248295, 0.7  ),
    (427100,427550, 1.2 ) ,
]

frequencies_c = np.array(frequencies_c)
v1_c = np.array(v1_c)
v2_c = np.array(v2_c)

segmented_data_c = []

for start, stop, step in segments_c:
    mask = (frequencies_c >= start) & (frequencies_c <= stop)
    
    seg_freq = frequencies_c[mask]
    seg_v1 = v1_c[mask]
    seg_v2 = v2_c[mask]
    
    segmented_data_c.append({
        "range": (start, stop),
        "freq": seg_freq,
        "v1": seg_v1,
        "v2": seg_v2
    })

# =============================================================================
# AJUSTE Y PLOTEO 2
# =============================================================================
f = np.array(segmented_data_c[0]["freq"])   
V2 = np.array(segmented_data_c[0]["v2"]) 
V1 = 2.15     
T = V2/V1 
sigma_V2 = 0.03 * V2 + 0.001
sigma_V1 = 0.03 * V1 + 0.001

sigma_T = T * np.sqrt((sigma_V2 / V2)**2 +(sigma_V1 / V1)**2)

f0_guess = f[np.argmax(T)]
A_guess = max(T) - min(T)
gamma_guess = (max(f) - min(f)) / 10
offset_guess = min(T)

p0 = [f0_guess, gamma_guess, A_guess, offset_guess]

popt, pcov = curve_fit(
    lorentzian, f, T, p0=p0, sigma=sigma_T, absolute_sigma=True, maxfev=10000
)

f0, gamma, A, offset = popt
perr = np.sqrt(np.diag(pcov))
df0, dgamma, dA, doffset = perr

f_fit = np.linspace(min(f), max(f), 1000)
T_fit = lorentzian(f_fit, *popt)

plt.errorbar(
    f, T,
    yerr=sigma_T,
    fmt='o',
    color='#1f77b4',       # Datos en azul
    ecolor='#7f7f7f',      # Barras de error en gris highlight
    elinewidth=1.5,
    capsize=4,
    capthick=1.5,
    alpha=0.9,
    label="Datos"
)
plt.plot(f_fit, T_fit, '-', color='#d62728', label="Ajuste") # Fit en rojo
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Voltaje")
plt.yscale("log")
ax = plt.gca()
ax.ticklabel_format(style='plain', axis='x', useOffset=False)
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
ax.yaxis.set_major_formatter(LogFormatter(base=10.0))

# Ejemplo de highlight con #7f7f7f (por si se necesita usar axvspan para resaltar regiones en el futuro)
# ax.axvspan(min(f), max(f), color='#7f7f7f', alpha=0.1, label="Highlight") 

plt.legend()
plt.show()

f_res = f0
df_res = df0
Tmax = lorentzian(f0, *popt)
dTmax = np.sqrt(dA**2 + doffset**2)
FWHM = 2 * gamma
dFWHM = 2 * dgamma
Q = f0 / FWHM
dQ = Q * np.sqrt((df0/f0)**2 + (dFWHM/FWHM)**2)

print(f"f0 = {f0:.2f} ± {df0:.2f} Hz")
print(f"FWHM = {FWHM:.2f} ± {dFWHM:.2f} Hz")
print(f"Q = {Q:.2f} ± {dQ:.2f}")
print(f"Tmax = {Tmax:.3f} ± {dTmax:.3f} V")

# =============================================================================
# CÁLCULOS LCR
# =============================================================================
R2 = 10000  
R = R2/Tmax - R2
L = Q * R / (2*np.pi*f0)
C = 1 / ((2*np.pi*f0)**2 * L)

print(f"R = {R:.2e} Ohm")
print(f"L = {L:.2e} H")
print(f"C = {C:.2e} F")

# =============================================================================
# AJUSTE Y PLOTEO 3
# =============================================================================
f = np.array(segmented_data_c[1]["freq"])   
V2 = np.array(segmented_data_c[1]["v2"])     
V1 = 2.15     
T = V2/V1 
sigma_V2 = 0.03 * V2 + 0.0001
sigma_V1 = 0.03 * V1 

sigma_T = T * np.sqrt((sigma_V2 / V2)**2 +(sigma_V1 / V1)**2)

mask = (f > 427400) & (f < 427525)   
f = f[mask]
T = T[mask]
sigma_T = sigma_T[mask]

f0_guess = f[np.argmax(T)]
A_guess = max(T)
gamma_guess = (max(f) - min(f)) / 10
offset_guess = min(T)

p0 = [f0_guess, gamma_guess, A_guess, offset_guess]

popt, pcov = curve_fit(
    lorentzian, f, T, p0=p0, sigma=sigma_T, absolute_sigma=True, maxfev=10000
)

f0, gamma, A, offset = popt
perr = np.sqrt(np.diag(pcov))
df0, dgamma, dA, doffset = perr

f_fit = np.linspace(min(f), max(f), 1000)
T_fit = lorentzian(f_fit, *popt)

plt.errorbar(
    f, T,
    yerr=sigma_T,
    fmt='o',
    color='#1f77b4',
    ecolor='#7f7f7f',
    elinewidth=1.5,
    capsize=4,
    capthick=1.5,
    alpha=0.9,
    label="Datos"
)
plt.plot(f_fit, T_fit, '-', color='#d62728', label="Ajuste")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Voltaje")
plt.yscale("log")
ax = plt.gca()
ax.ticklabel_format(style='plain', axis='x', useOffset=False)
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
ax.yaxis.set_major_formatter(LogFormatter(base=10.0))
plt.legend()
plt.show()

# =============================================================================
# EXTRACCIÓN Y SEGMENTACIÓN DE DATOS 3
# =============================================================================
file_list = sorted(glob.glob("13h_28m_49s/*.pkl"))

frequencies = []
v1 = []
v2 = []

for file in file_list:
    filename = os.path.basename(file)
    freq = float(filename.split("_")[1].replace("Hz.pkl", ""))
    frequencies.append(freq)

    with open(file, "rb") as f:
        data = pickle.load(f)
        v1.append(float(data['v1']))
        v2.append(float(data['v2']))
        
segments = [
    (150140,150170, 0.5 ),
    (248275,248295, 0.7),
    (427100,427550, 1.2),
    (558640,560670,20), 
    (648850,650580,30)
]

frequencies = np.array(frequencies)
v1 = np.array(v1)
v2 = np.array(v2)

segmented_data = []

for start, stop, step in segments:
    mask = (frequencies >= start) & (frequencies <= stop)
    
    seg_freq = frequencies[mask]
    seg_v1 = v1[mask]
    seg_v2 = v2[mask]
    
    segmented_data.append({
        "range": (start, stop),
        "freq": seg_freq,
        "v1": seg_v1,
        "v2": seg_v2
    })

# =============================================================================
# AJUSTE Y PLOTEO 4
# =============================================================================
f = np.array(segmented_data[3]["freq"])   
V2 = np.array(segmented_data[3]["v2"])     
V1 = 2.15     
T = V2/V1 
sigma_V2 = 0.03 * V2 + 0.0001
sigma_V1 = 0.03 * V1 

sigma_T = T * np.sqrt((sigma_V2 / V2)**2 +(sigma_V1 / V1)**2)

mask = (f > 560100) & (f < 560560)   

f = f[mask]
T = T[mask]
sigma_T = sigma_T[mask]

f0_guess = f[np.argmax(T)]
A_guess = max(T)
gamma_guess = (max(f) - min(f)) / 10
offset_guess = min(T)

p0 = [f0_guess, gamma_guess, A_guess, offset_guess]

popt, pcov = curve_fit(
    lorentzian, f, T, p0=p0, sigma=sigma_T, absolute_sigma=True, maxfev=10000
)

f0, gamma, A, offset = popt

f_fit = np.linspace(min(f), max(f), 1000)
T_fit = lorentzian(f_fit, *popt)

plt.errorbar(
    f, T,
    yerr=sigma_T,
    fmt='o',
    color='#1f77b4',
    ecolor='#7f7f7f',
    elinewidth=1.5,
    capsize=4,
    capthick=1.5,
    alpha=0.9,
    label="Datos"
)
plt.plot(f_fit, T_fit, '-', color='#d62728', label="Ajuste")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Voltaje")
plt.yscale("log")
ax = plt.gca()
ax.ticklabel_format(style='plain', axis='x', useOffset=False)
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
ax.yaxis.set_major_formatter(LogFormatter(base=10.0))
plt.legend()
plt.show()

# =============================================================================
# AJUSTE Y PLOTEO 5
# =============================================================================
f = np.array(segmented_data[4]["freq"])   
V2 = np.array(segmented_data[4]["v2"])     
V1 = 2.15     
T = V2/V1 
sigma_V2 = 0.03 * V2 + 0.0001
sigma_V1 = 0.03 * V1 

sigma_T = T * np.sqrt((sigma_V2 / V2)**2 +(sigma_V1 / V1)**2)

mask = (f > 650100) & (f < 660600)   

f = f[mask]
T = T[mask]
sigma_T = sigma_T[mask]

f0_guess = f[np.argmax(T)]
A_guess = max(T)
gamma_guess = (max(f) - min(f)) / 10
offset_guess = min(T)

p0 = [f0_guess, gamma_guess, A_guess, offset_guess]

popt, pcov = curve_fit(
    lorentzian, f, T, p0=p0, sigma=sigma_T, absolute_sigma=True, maxfev=10000
)

f0, gamma, A, offset = popt

f_fit = np.linspace(min(f), max(f), 1000)
T_fit = lorentzian(f_fit, *popt)

plt.errorbar(
    f, T,
    yerr=sigma_T,
    fmt='o',
    color='#1f77b4',
    ecolor='#7f7f7f',
    elinewidth=1.5,
    capsize=4,
    capthick=1.5,
    alpha=0.9,
    label="Datos"
)
plt.plot(f_fit, T_fit, '-', color='#d62728', label="Ajuste")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Voltaje")
plt.yscale("log")
ax = plt.gca()
ax.ticklabel_format(style='plain', axis='x', useOffset=False)
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
ax.yaxis.set_major_formatter(LogFormatter(base=10.0))
plt.legend()
plt.show()

# =============================================================================
# GRAFICOS DE RESULTADOS Y TRANSFERENCIAS MÁXIMAS
# =============================================================================
Ts_max = [0.566, 0.281, 0.054, 0.072, 0.059]
Ts_max_errs = [0.011, 0.006, 0.002, 0.001, 0.002]
frecs_modos = [150160.48, 248287.04, 427475.85, 560463.26, 650444.91]
frecs_modos_errs = [0.04, 0.06, 1.26, 2.59, 3.69]

freqs_kHz = np.array(frecs_modos) / 1000  

plt.errorbar(freqs_kHz, Ts_max, yerr=Ts_max_errs, fmt='o', color='#1f77b4', ecolor='#7f7f7f', markersize=8)
plt.plot(freqs_kHz, Ts_max, alpha=0.6, color='#d62728', linewidth=3)
plt.xlabel("Frecuencia del armónico (kHz)")
plt.xticks(freqs_kHz)
plt.ylabel("Transferencia máxima")
plt.title("Transferencia máxima vs frecuencia del armónico")
plt.show()

ns = [ 3, 5, 9, 11, 13]

plt.errorbar(ns, freqs_kHz, yerr=np.array(frecs_modos_errs)/1000, fmt='o', color='#1f77b4', ecolor='#7f7f7f', markersize=8)
plt.ylabel("Frecuencia del armónico (kHz)")
plt.xticks(ns)
plt.yticks(freqs_kHz)
plt.xlabel("Número de modo")
plt.title("f vs n")
plt.show()

# =============================================================================
# VELOCIDAD DE FASE Y AJUSTE LINEAL
# =============================================================================
vs_f = []
for i in range(len(ns)):
    L = 4e-3
    f = frecs_modos[i]
    n = ns[i]
    v_f =  4 * L * f / n
    print("La velocidad de fase en el modo " + str(i) +" es: ", v_f)
    vs_f.append(v_f)

def lineal(x, a, b):
    return a*x + b

ns = np.array(ns)
frecs_modos = np.array(frecs_modos)
frecs_modos_errs = np.array(frecs_modos_errs)

popt, pcov = curve_fit(lineal, ns, frecs_modos, sigma=frecs_modos_errs, absolute_sigma=True, maxfev=10000)
a, b = popt
perr = np.sqrt(np.diag(pcov))
da, db = perr

plt.errorbar(
    ns, frecs_modos,
    yerr=frecs_modos_errs,
    fmt='o',
    color='#1f77b4',
    ecolor='#7f7f7f',
    elinewidth=1.5,
    capsize=4,
    capthick=1.5,
    alpha=0.9,
    label="Datos"
)
plt.plot(ns, lineal(ns, a, b), '-', color='#d62728', linewidth=3, label="Ajuste")
plt.ylabel("Frecuencia (Hz)")
plt.xlabel("Número de modo")
plt.xticks(ns)
plt.legend()
plt.show()

print(f"v_f = {a*4*4*1e-3:.2f} ± {da*4*4*1e-3} m/s")