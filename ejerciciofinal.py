"""
==================
Final Assignment
==================

El largo de los registros es entre 10 y 11 minutos
Fs = 512

FECHA DE ENTREGA: 10/01/2023
SEGUNDA FECHA DE ENTREGA: 10/02/2023


|---- BASELINE --------|
|---- TOSER ------|
|---- RESPIRAR FONDO ------- |
|---- RESPIRAR RAPIDO ----|
|---- CUENTA MENTAL --------|
|---- COLORES VIOLETA ------|
|---- COLORES ROJO --------|
|---- SONREIR -----|
|---- DESEGRADABLE -----|
|---- AGRADABLE --------|
|---- PESTANEOS CODIGO ------ |

* Baseline: esta parte la pueden utilizar para tener ejemplos negativos de cualquier cosa que deseen detectar.  Por
ejemplo si quieren detectar que algo cambia cuando hay "imaginación en colores violeta", extraen features de ese momento y de
este e intentan armar un clasificador.
* Toser: Probablemente queden registrados como picos, provocados por el propio movimiento de la cabeza.
* Respirar fondo vs respirar rápido: quizás puede haber un cambio en alguna frecuencia.
* Cuenta mental: Está reportado que esto genera cambios en las frecuencias altas gamma y beta, de entre 20-30 Hz.
* Colores violeta / rojo:  de acá primero pueden intentar ver si hay cambio en relación a baseline en la frecuencia
de 10 Hz porque para ambos casos cerré los ojos.  Luego pueden intentar ver si un clasificador les puede diferenciar las clases.
* Sonreir: esto quizás genere algunos artefactos, picos en la señal debido al movimiento de la cara.
* Agradable/Desagradable: aca no tengo idea, prueben contra baseline.  No hay nada reportado así.
* Pestañeos:  En esta parte hay pestañeos que pueden intentar extraer.


Los datos, el registro de EEG y el video, están disponibles en el siguiente link:
https://drive.google.com/file/d/1ByQDK4ZPxbqw7T17k--avTcgSCCzs3vi/view?usp=sharing

Objetivo:
El objetivo es dado este registro implementar un análisis de estos datos, exploratorio, superviado
o no supervisado, para intentar identificar que es lo que el sujeto está haciendo en cada bloque.  Pueden
intentar separar dos bloques entre sí, un bloque particular frente al BASELINE (esto es el momento cuando el sujeto
no hace nada particular).  Pueden usar una parte de dos bloques para entrenar y luego intentar predecir las otras partes.
Tienen que producir un PDF informe con gráficos/tablas breve y resumido.

Fecha de entrega: 10 de Enero 2023
Segunda fecha de entrega: 10 de Febrero 2023

"""

# print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import math
import scipy
from scipy import stats
from scipy.signal import butter, lfilter, detrend
from io import StringIO
from scipy.fft import rfft, rfftfreq

# El protocolo experimental que implementamos tiene 2 datasets:
# 1- Dataset de las señales de EEG
# 2- El video de las imágenes.
#
#
# La idea es tomar estos datasets y derivar de forma automática las diferentes secciones.  Esto se puede hacer en base self-supervised, es
# decir tomar los datos de algún dataset, derivar los labels para cada secciones y luego intentar implementar un clasificador multiclase.
#
# Tienen que entregar un PDF, tipo Markdown con código, gráficos y cualquier insight obtenido del dataset.


# MAIN

def main():
    df = pd.read_csv('protocolo/eeg.dat', delimiter=' ', names=[
        'timestamp', 'counter', 'eeg', 'attention', 'meditation', 'blinking'])

    print('Estructura de la informacion:')
    print(df.head())

    # Agregar features
    init_ts = df.timestamp[0]
    df.loc[:, "timer"] = df.timestamp - init_ts # Util para los graficos (eje x)
    df['eeg_detrended'] = detrend(df.eeg)
    df['delta'] = butter_bandpass_filter(df.eeg, 0.5, 2.75)
    df['theta'] = butter_bandpass_filter(df.eeg, 3.5, 6.75)
    df['alpha_low'] = butter_bandpass_filter(df.eeg, 7.5, 9.25)
    df['alpha_high'] = butter_bandpass_filter(df.eeg, 10.0, 11.75)
    df['beta_low'] = butter_bandpass_filter(df.eeg, 13.0, 16.75)
    df['beta_high'] = butter_bandpass_filter(df.eeg, 18.0, 29.75)
    df['gamma_low'] = butter_bandpass_filter(df.eeg, 31.0, 39.75)
    df['gamma_mid'] = butter_bandpass_filter(df.eeg, 41.0, 49.75)

    # Gráficos
    plot_signal(df, "general")
    # eventcounter(df.eeg, df.timer)

    labels = {
        "0:01 - 0:04": "pestaneo_rapido",
        "0:06 - 1:04": "baseline",
        "1:05 - 1:06": "pestaneo_rapido",
        "1:08 - 2:10": "tos",
        "2:11 - 2:13": "pestaneo_rapido",
        "2:14 - 3:10": "respira_hondo",
        "3:11 - 3:13": "pestaneo_rapido",
        "3:14 - 4:09": "respira_rapido",
        "4:10 - 4:12": "pestaneo_rapido",
        "4:13 - 5:09": "cuenta_mental",
        "5:10 - 5:14": "pestaneo_rapido",
        "5:15 - 6:11": "violeta",
        "6:12 - 6:15": "pestaneo_rapido",
        "6:16 - 7:10": "rojo",
        "7:11 - 7:13": "pestaneo_rapido",
        "7:14 - 8:13": "sonreir",
        "8:14 - 8:15": "pestaneo_rapido",
        "8:16 - 8:37": "desagradable",
        "8:38 - 10:10": "agradable",
        "10:12 - 10:13": "pestaneo_rapido",
        "10:14 - 11:00": "pestaneo_codigo",
    }
    for (interval, label) in labels.items():
        start, end = interval.split(" - ")
        filtered_signals = df.loc[(df.timer >= mark_to_ts(start)) & (df.timer <= mark_to_ts(end))]
        process_chunk(filtered_signals, label, start, end)

def mark_to_ts(mark: str):
    mins, segs = mark.split(":")
    return int(mins) * 60 + int(segs)

def process_chunk(signals, label, start_mark, end_mark):
    plot_signal(signals, f"{label} ({start_mark} ~ {end_mark})")

# Bandas de Frecuencia
SAMPLING_RATE = 512
def butter_bandpass(lowcut, highcut, fs=SAMPLING_RATE, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs=SAMPLING_RATE, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_signal(df: pd.DataFrame, plotname: str):
    fig, ax = plt.subplots(nrows=5, ncols=2, sharex='col', **
                         {"figsize": (24, 12), "dpi": 100})
    fig.suptitle(plotname)
    fig.supxlabel('t [seg] (medido desde el inicio del dataset)')

    LINEWIDTH = 0.75

    ax[0, 0].plot(df.timer, df.eeg, color='steelblue', linewidth=LINEWIDTH)
    ax[0, 0].set_ylabel('eeg')
    ax[0, 1].plot(df.timer, df.eeg_detrended, color='steelblue', linewidth=LINEWIDTH)
    ax[0, 1].set_ylabel('eeg_detrended')

    ax[1, 0].plot(df.timer, df.delta, color='green', linewidth=LINEWIDTH)
    ax[1, 0].set_ylabel('delta')
    ax[1, 1].plot(df.timer, df.theta, color='green', linewidth=LINEWIDTH)
    ax[1, 1].set_ylabel('theta')

    ax[2, 0].plot(df.timer, df.alpha_low, color='red', linewidth=LINEWIDTH)
    ax[2, 0].set_ylabel('alpha_low')
    ax[2, 1].plot(df.timer, df.alpha_high, color='red', linewidth=LINEWIDTH)
    ax[2, 1].set_ylabel('alpha_high')

    ax[3, 0].plot(df.timer, df.beta_low, color='violet', linewidth=LINEWIDTH)
    ax[3, 0].set_ylabel('beta_low')
    ax[3, 1].plot(df.timer, df.beta_high, color='violet', linewidth=LINEWIDTH)
    ax[3, 1].set_ylabel('beta_high')

    ax[4, 0].plot(df.timer, df.gamma_low, color='orange', linewidth=LINEWIDTH)
    ax[4, 0].set_ylabel('gamma_low')
    ax[4, 1].plot(df.timer, df.gamma_mid, color='orange', linewidth=LINEWIDTH)
    ax[4, 1].set_ylabel('gamma_mid')

    plt.tight_layout()
    plt.savefig(f"out/fase-{plotname}.png")

    # plt.show()


def eventcounter(eeg, timer):
    # print("Some values from the dataset:\n")
    # print(results[0:10,])
    # print("Matrix dimension: {}".format(results.shape))
    print("EEG Vector Metrics\n")
    print("Length: {}".format(len(eeg)))
    print("Max value: {}".format(eeg.max()))
    print("Min value: {}".format(eeg.min()))
    print("Range: {}".format(eeg.max()-eeg.min()))
    print("Average value: {}".format(eeg.mean()))
    print("Variance: {}".format(eeg.var()))
    print("Std: {}".format(math.sqrt(eeg.var())))
    plt.figure(figsize=(12, 5))
    plt.plot(timer, eeg, color="green")
    plt.ylabel("Amplitude", size=10)
    plt.xlabel("t [seg]", size=10)
    plt.title("Serie temporal de eeg", size=20)
    plt.savefig("out/eeg.png")
    plt.show()

    # Prueba de normalidad
    print('normality = {}'.format(scipy.stats.normaltest(eeg)))
    sns.distplot(eeg)
    plt.title("Normality-1 Analysis on EEG vector")
    plt.savefig("out/norm-1.png")
    plt.show()
    sns.boxplot(eeg, color="red")
    plt.title("Normality-2 Analysis on EEG vector")
    plt.savefig("out/norm2.png")
    plt.show()
    res = stats.probplot(eeg, plot=plt)
    plt.title("Normality-3 Analysis on EEG vector")
    plt.savefig("out/norm3.png")
    plt.show()

    # Find the threshold values to determine what is a blinking and what is not
    umbral_superior = int(eeg.mean()+3*eeg.std())
    print("Upper Threshold: {}".format(umbral_superior))
    umbral_inferior = int(eeg.mean()-3*eeg.std())
    print("Lower Threshold: {}".format(umbral_inferior))
    plt.figure(figsize=(12, 5))
    plt.plot(timer, eeg, color="green")
    plt.plot(timer, np.full(len(eeg), umbral_superior), 'r--')
    plt.plot(timer, np.full(len(eeg), umbral_inferior), 'r--')
    plt.ylabel("Amplitude", size=10)
    plt.xlabel("t [seg]", size=10)
    plt.title("EEG Series with control limits", size=20)
    plt.annotate("Upper Threshold", xy=(500, umbral_superior+10), color="red")
    plt.annotate("Lower Threshold", xy=(500, umbral_inferior+10), color="red")
    plt.savefig("out/blinking-std.png")
    plt.show()

    lowerbound = int(np.percentile(eeg, 1))
    upperbound = int(np.percentile(eeg, 99))

    plt.plot(timer, eeg, color="steelblue")
    plt.plot(timer, np.full(len(eeg), lowerbound), color="goldenrod", ls="--")
    plt.plot(timer, np.full(len(eeg), upperbound), color="goldenrod", ls="--")
    plt.ylabel("Amplitude", size=10)
    plt.xlabel("t [seg]", size=10)
    plt.title("EEG Series with control limits", size=20)
    # dinamizo los valores del eje así se adapta a los datos que proceso
    plt.ylim([min(eeg)*1.1, max(eeg)*1.1])
    plt.annotate("Lower Bound", xy=(500, lowerbound+10), color="goldenrod")
    plt.annotate("Upper Bound", xy=(500, upperbound+10), color="goldenrod")
    plt.savefig('out/blinking-pct.png')
    plt.show()

    # Grafico el filtro de pestañeos/blinking
    # Utilizo una función lambda para marcar los pestañeos

    blinks = list(
        (map(lambda x: 1 if x > upperbound else (-1 if x < lowerbound else 0), eeg)))
    blinks = np.asarray(blinks)

    plt.plot(timer, blinks, color="darksalmon")
    plt.title("Blinking Filter", size=20)
    plt.ylabel("Class", size=10)
    plt.xlabel("t [seg]", size=10)
    plt.savefig('out/blinkingfilter.png')
    plt.show()

    # Encuentro picos positivos. Filtro los valores donde blink==1, y luego analizo que haya habido un salto realmente (para no contar dos veces puntos consecutivos).
    # Con un map y una funcion lambda obtengo una lista con booleanos para los valores donde hay picos realmente.
    # Luego los filtro con una función filter y otra lambda
    peak = np.where(blinks == 1)[0]

    peakdiff = np.diff(np.append(0, peak))

    boolpeak = list(map(lambda x: x > 100, peakdiff))

    peakslocation = list(filter(lambda x: x, boolpeak*peak))

    # Repito para los valles, mismo algoritmo pero busco blinks == -1
    valley = np.where(blinks == -1)[0]

    valleydiff = np.diff(np.append(0, valley))

    boolvalley = list(map(lambda x: x > 100, valleydiff))

    valleylocation = list(filter(lambda x: x, boolvalley*valley))

    # Hago un append de los valles y los picos, y los ordeno. Luego los cuento para imprimir tanto la cantidad de pestañeos, como la localización de los mismos

    blinklocations = np.sort(np.append(peakslocation, valleylocation))

    blinkcount = np.count_nonzero(blinklocations)

    print(f'Count of Blinks: {blinkcount}')
    print('Location of Blinks')
    print(blinklocations)


if __name__ == "__main__":
    main()
