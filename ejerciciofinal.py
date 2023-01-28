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
import requests
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
    signals = pd.read_csv('protocolo/eeg.dat', delimiter=' ', names=[
        'timestamp', 'counter', 'eeg', 'attention', 'meditation', 'blinking'])

    print('Estructura de la informacion:')
    print(signals.head())

    data = signals.values
    eeg = data[:, 2]

    plot_signal(signals)


def plot_signal(df: pd.DataFrame):
    init_ts = df.timestamp[0]
    df.loc[:, "timer"] = df.timestamp - init_ts

    ticks = [i for i in range(0, int(df.timer.max())+60, 60)]

    _, ax = plt.subplots(nrows=4, ncols=2, sharex='col', **
                         {"figsize": (24, 12), "dpi": 100})

    LINEWIDTH = 0.75

    # En la primera columna grafico señales en el tiempo

    ax[0, 0].plot(df.timer, df.eeg, color='steelblue', linewidth=LINEWIDTH)
    ax[0, 0].set_title('EEG - Señales en el tiempo')
    ax[0, 0].set_ylabel('eeg(t)')

    ax[1, 0].plot(df.timer, df.meditation, color='green', linewidth=LINEWIDTH)
    ax[1, 0].set_ylabel('meditation(t)')

    ax[2, 0].plot(df.timer, df.attention, color='orange', linewidth=LINEWIDTH)
    ax[2, 0].set_ylabel('attention(t)')

    ax[3, 0].plot(df.timer, df.blinking, color='red', linewidth=LINEWIDTH)
    ax[3, 0].set_ylabel('blinking(t)')

    ax[3, 0].set_xticks(ticks)
    ax[3, 0].set_xlabel("t [s]")

    # En la 2da columna grafico en el espectro de frecuencias
    N = len(df)
    SAMPLE_RATE = 512  # dato del enunciado

    xf = rfftfreq(N, 1 / SAMPLE_RATE)
    trunc = len(xf) // 10 # Frecuencias por encima de esto casi ni se ve la amplitud

    ax[0, 1].plot(xf[:trunc], np.abs(rfft(df.eeg))[:trunc],
                  color='steelblue', linewidth=LINEWIDTH)
    ax[0, 1].set_ylabel('eeg(f)')
    ax[0, 1].set_title('EEG - Espectro')

    ax[1, 1].plot(xf[:trunc], np.abs(rfft(df.meditation))[:trunc],
                  color='green', linewidth=LINEWIDTH)
    ax[1, 1].set_ylabel('meditation(f)')

    ax[2, 1].plot(xf[:trunc], np.abs(rfft(df.attention))[:trunc],
                  color='orange', linewidth=LINEWIDTH)
    ax[2, 1].set_ylabel('attention(f)')

    ax[3, 1].plot(xf[:trunc], np.abs(rfft(df.blinking))[:trunc],
                  color='red', linewidth=LINEWIDTH)
    ax[3, 1].set_ylabel('blinking(f)')
    ax[3, 1].set_xlabel(
        "f [Hz]\n(truncado porque a mayores frecuencias las señales son mínimas)")

    plt.tight_layout()
    plt.savefig("out/timepo-espectro.png")

    plt.show()


if __name__ == "__main__":
    main()
