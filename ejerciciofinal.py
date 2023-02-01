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
from collections import Counter
from scipy.signal import butter, lfilter, detrend
from io import StringIO
from scipy.fft import rfft, rfftfreq
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score

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

PREPROCESS_DATASET = False
PREPROCESSED_DATASET_PATH = 'out/eeg_enriched.csv'

PESTANEO_RAPIDO = "pestaneo_rapido"
SEED = 17

LABELS = {
    "0:01 - 0:04": PESTANEO_RAPIDO,
    "0:06 - 1:04": "baseline",
    "1:05 - 1:06": PESTANEO_RAPIDO,
    "1:08 - 2:10": "tos",
    "2:11 - 2:13": PESTANEO_RAPIDO,
    "2:14 - 3:10": "respira_hondo",
    "3:11 - 3:13": PESTANEO_RAPIDO,
    "3:14 - 4:09": "respira_rapido",
    "4:10 - 4:12": PESTANEO_RAPIDO,
    "4:13 - 5:09": "cuenta_mental",
    "5:10 - 5:14": PESTANEO_RAPIDO,
    "5:15 - 6:11": "violeta",
    "6:12 - 6:15": PESTANEO_RAPIDO,
    "6:16 - 7:10": "rojo",
    "7:11 - 7:13": PESTANEO_RAPIDO,
    "7:14 - 8:13": "sonreir",
    "8:14 - 8:15": PESTANEO_RAPIDO,
    "8:16 - 8:37": "desagradable",
    "8:38 - 10:10": "agradable",
    "10:12 - 10:13": PESTANEO_RAPIDO,
    "10:14 - 11:00": "pestaneo_codigo",
}


def main():
    if PREPROCESS_DATASET:
        preprocess()
        return
    else:
        df = pd.read_csv(PREPROCESSED_DATASET_PATH)

    print('Estructura de la informacion enriquecida:')
    print(df.head())

    # plots(df)
    modelos(df)


def preprocess():
    df = pd.read_csv('protocolo/eeg.dat', delimiter=' ', names=[
        'timestamp', 'counter', 'eeg', 'attention', 'meditation', 'blinking'])

    print('Estructura de la informacion:')
    print(df.head())

    # Agregar features
    init_ts = df.timestamp[0]
    df.loc[:, "timer"] = df.timestamp - init_ts  # Para el eje x
    df['eeg_detrended'] = detrend(df.eeg)
    add_band(df, 'eeg_detrended', None, None)
    add_band(df, 'attention', None, None)
    add_band(df, 'meditation', None, None)
    add_band(df, 'theta', 3.5, 6.75)
    add_band(df, 'alpha_low', 7.5, 9.25)
    add_band(df, 'alpha_high', 10.0, 11.75)
    add_band(df, 'beta_low', 13.0, 16.75)
    add_band(df, 'beta_high', 18.0, 29.75)
    add_band(df, 'gamma_low', 31.0, 39.75)
    add_band(df, 'gamma_mid', 41.0, 49.75)

    # Agregar clase
    for (interval, label) in LABELS.items():
        start, end = interval.split(" - ")
        predicate = (
            df.timer >= mark_to_ts(start)) & (df.timer <= mark_to_ts(end))
        df.loc[predicate, "label"] = label

    # Persistir
    df.to_csv(PREPROCESSED_DATASET_PATH, index=False)
    print('Estructura de la informacion enriquecida:')
    print(df.head())


def plots(df: pd.DataFrame):
    plot_signal(df, "general")

    for (interval, label) in LABELS.items():
        start, end = interval.split(" - ")
        filtered_signals = df.loc[(df.timer >= mark_to_ts(start)) &
                                  (df.timer <= mark_to_ts(end))]
        process_chunk(filtered_signals, label, start, end)


def mark_to_ts(mark: str):
    mins, segs = mark.split(":")
    return int(mins) * 60 + int(segs)


def process_chunk(signals, label, start_mark, end_mark):
    plot_signal(signals, f"{label} ({start_mark} ~ {end_mark})")


def crest_factor(x):
    return np.max(np.abs(x)) / np.sqrt(np.mean(np.square(x)))


def peak_to_peak(a):
    return abs(np.max(a)) + abs(np.min(a))


def shannon_entropy(a):
    return scipy.stats.entropy(list(Counter(a).values()), base=2)


def hjorth(a):
    first_deriv = np.diff(a)
    second_deriv = np.diff(a, 2)

    var_zero = np.mean(a ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity

    return activity, morbidity, complexity


# Bandas de Frecuencia
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y


def add_band(df, name, lowcut, highcut, fs=512, order=5):
    print(f"======================= ADDING BAND {name} ======================")
    if lowcut and highcut:
        df[name] = butter_bandpass_filter(
            df.eeg_detrended, lowcut, highcut, fs, order)

    grouped = df.groupby(['timestamp'])[name]

    df[f'{name}_mean'] = grouped.transform('mean')
    df[f'{name}_std'] = grouped.transform('std')
    df[f'{name}_ptp'] = grouped.transform(peak_to_peak)
    df[f'{name}_entropy'] = grouped.transform(shannon_entropy)
    df[f'{name}_crest'] = grouped.transform(crest_factor)
    df[f'{name}_hjorth_activity'] = grouped.transform(lambda a: hjorth(a)[0])
    df[f'{name}_hjorth_morbidity'] = grouped.transform(lambda a: hjorth(a)[1])
    df[f'{name}_hjorth_complexity'] = grouped.transform(lambda a: hjorth(a)[2])


def plot_signal(df: pd.DataFrame, plotname: str):
    fig, ax = plt.subplots(nrows=4, ncols=2, sharex='col', **
                           {"figsize": (24, 12), "dpi": 100})
    fig.suptitle(plotname)
    fig.supxlabel('t [seg] (medido desde el inicio del dataset)')

    LINEWIDTH = 0.75

    ax[0, 0].plot(df.timer, df.eeg_detrended,
                  color='steelblue', linewidth=LINEWIDTH)
    ax[0, 0].set_ylabel('eeg_detrended')
    ax[1, 0].plot(df.timer, df.theta, color='green', linewidth=LINEWIDTH)
    ax[1, 0].set_ylabel('theta')

    ax[0, 1].plot(df.timer, df.alpha_low, color='red', linewidth=LINEWIDTH)
    ax[0, 1].set_ylabel('alpha_low')
    ax[1, 1].plot(df.timer, df.alpha_high, color='red', linewidth=LINEWIDTH)
    ax[1, 1].set_ylabel('alpha_high')

    ax[2, 0].plot(df.timer, df.beta_low, color='violet', linewidth=LINEWIDTH)
    ax[2, 0].set_ylabel('beta_low')
    ax[3, 0].plot(df.timer, df.beta_high, color='violet', linewidth=LINEWIDTH)
    ax[3, 0].set_ylabel('beta_high')

    ax[2, 1].plot(df.timer, df.gamma_low, color='orange', linewidth=LINEWIDTH)
    ax[2, 1].set_ylabel('gamma_low')
    ax[3, 1].plot(df.timer, df.gamma_mid, color='orange', linewidth=LINEWIDTH)
    ax[3, 1].set_ylabel('gamma_mid')

    plt.tight_layout()
    plt.savefig(f"out/fase-{plotname}.png")

    plt.show()


def modelos(df: pd.DataFrame):
    # Seteo clase binaria
    df.loc[df.label == PESTANEO_RAPIDO, 'target'] = 1.0
    df.loc[df.label != PESTANEO_RAPIDO, 'target'] = 0.0
    print("DF con target", df.head())

    # Selecciono features
    features = [f"{band}_{metric}"
                for band in ['eeg_detrended', 'theta', 'alpha_low', 'alpha_high', 'beta_low', 'beta_high', 'gamma_low', 'gamma_mid']
                for metric in ['mean', 'std', 'ptp', 'entropy', 'crest', 'hjorth_activity', 'hjorth_morbidity', 'hjorth_complexity']
                ]
    df = df[features + ['label', 'target', 'timer']]

    # Dropeo registros sin label o con otros problemas
    print("DF shape", df.shape)
    df = df.dropna()
    print("DF shape sin NaNs", df.shape)

    # Escalo datos
    print("DF summary", df.describe())
    scaler = MinMaxScaler()
    df.loc[:, features] = scaler.fit_transform(df[features])
    print("DF escalado summary", df.describe())

    # Split train/test
    CUT = "5:14"
    train_full = df[(df.timer <= mark_to_ts(CUT)) & (df.label != "tos")]
    train_counts = train_full.target.value_counts()
    print("Registros por clase (train), pre-sampling:", train_counts)
    train_full = train_full.groupby('target', group_keys=False).apply(
        lambda x: x.sample(train_counts.min(), random_state=SEED))
    train_class = train_full['target']
    train = train_full[features]
    test_full = df[(df.timer > mark_to_ts(CUT)) &
                   (df.timer < mark_to_ts("10:14"))]
    test_counts = test_full.target.value_counts()
    print("Registros por clase (test), pre-sampling:", test_counts)
    test_full = test_full.groupby('target', group_keys=False).apply(
        lambda x: x.sample(test_counts.min(), random_state=SEED))
    test_class = test_full['target']
    test = test_full[features]

    print("train shape (balanceado)", train.shape)
    print("test shape (balanceado)", test.shape)

    # Inicializo modelos
    models = {
        "log_rec": LogisticRegression(random_state=SEED),
        "svm_linear": SVC(kernel='linear', random_state=SEED),
        "svm_poly": SVC(kernel='poly', random_state=SEED),
        "svm_rbf": SVC(kernel='rbf', random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }

    plt.figure()
    plt.title("ROC")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    for name, model in models.items():
        start = time()

        def prefix():
            return f"{time() - start:.2f}s {name}:"
        print()
        print(prefix(), "About to fit")
        model.fit(train, train_class)
        print(prefix(), "About to predict")
        predictions = model.predict(test)
        print(prefix(), "About to measure")
        fpr, tpr, thresholds = roc_curve(test_class, predictions)
        area = auc(fpr, tpr)
        print(prefix(),
              f"Accurracy score = {accuracy_score(test_class, predictions)}")
        print(prefix(), f"AUC = {area}")
        print(prefix(),
              f"Confusion matrix = \n{confusion_matrix(test_class, predictions)}")
        plt.plot(fpr, tpr, label=f"{name} (AUC={area:.3f})")

    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
