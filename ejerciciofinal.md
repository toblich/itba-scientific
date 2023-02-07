# Ejercicio Final - Tobías Lichtig

## Contexto

Para este trabajo final, se cuenta con un dataset con la señal de un EEG (electroencefalograma) con una frecuencia de muestreo de 512Hz y una duración total de aproximadamente 11 minutos. Durante este periodo, el sujeto en cuestión realiza diversas tareas físicas y mentales. Mediante un video que acompaña a la señal, se puede determinar en qué momento el sujeto está realizando qué tarea.

## Objetivo

El objetivo final es entrenar un clasificador que reconozca cuándo el sujeto está pestañeando rápidamente.

## Desarrollo

Con tal fin, primero se identifican los tiempos entre los que el individuo ejecuta cada tarea. Luego, se generan diversas features a partir del dataset original que permiten estudiar la señal desde diferentes lentes, y se exploran las mismas mediante visualizaciones. Finalmente, se entrenan distintos modelos y se los compara.

### Identificar etapas

Observando el video y sabiendo de antemano la secuencia de etapas, se determina el siguiente intervalo para cada una. Cabe destacar que los tiempos están medidos como offsets desde el comienzo de la señal del dataset, con lo que `00:00` es el comienzo del dataset, habiendo transcurrido 0 minutos y 0 segundos desde el inicio del mismo. El siguiente mapa resume esta clasificación manual:

```python
PESTANEO_RAPIDO = "pestaneo_rapido"
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
```

### Generar features

Considerando el objetivo del trabajo, se pre-procesa la señal para extraer ciertas bandas de frecuencias que facilitan el análisis del EEG. Las señales generadas son:
- EEG sin tendencia
- Ondas theta
- Ondas alpha bajas
- Ondas alpha altas
- Ondas beta bajas
- Ondas beta altas
- Ondas gamma bajas
- Ondas gamma medias
- Al probar generar las ondas delta, los resultados son extraños (con valores infinitos y `NaN`), con lo que se las descarta.

Además, a cada una de las señales generadas listadas y a las métricas iniciales de atención y meditación presentes junto al EEG, se las procesa para calcular por cada segundo las siguientes métricas:
- Media
- Desvío estándar
- Peak-to-peak
- Entropía de Shannon
- Factor de cresta
- Parámetros de Hjorth (activity, morbidity, complexity)

También se anota cada medición con la clase definida al comienzo. Este preprocesamiento genera como salida, a partir de un dataset de 9.1MB en texto plano, un dataset extendido de 475MB en formato CSV.

```python
PREPROCESSED_DATASET_PATH = 'out/eeg_enriched.csv'


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


def mark_to_ts(mark: str):
    mins, segs = mark.split(":")
    return int(mins) * 60 + int(segs)


def crest_factor(x):
    return np.max(np.abs(x)) / np.sqrt(np.mean(np.square(x)))


def peak_to_peak(a):
    return abs(np.max(a)) + abs(np.min(a))


def shannon_entropy(a):
    return entropy(list(Counter(a).values()), base=2)


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

```

### Visualizar las señales

Tanto para el dataset completo como para cada etapa dentro del mismo, se visualizan las señales generadas (sin todas las métricas por segundo, sino las señales base). Cabe destacar que en este paso ya se descartan las señales delta (como fue nombrado anteriormente) y la señal de pestañeo provista por el dispositivo, ya que la misma era constantemente nula. También se excluyen de los gráficos las señales de atención y meditación, ya que no aparentan tener mayor relevancia para este caso y se desea evitar tener demasiadas cosas en un mismo gráfico. A continuación se muestran algunas de las visualizaciones generadas.

```python
def plots(df: pd.DataFrame):
    plot_signal(df, "general")

    for (interval, label) in LABELS.items():
        start, end = interval.split(" - ")
        filtered_signals = df.loc[(df.timer >= mark_to_ts(start)) &
                                  (df.timer <= mark_to_ts(end))]
        process_chunk(filtered_signals, label, start, end)

def process_chunk(signals, label, start_mark, end_mark):
    plot_signal(signals, f"{label} ({start_mark} ~ {end_mark})")

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
```

### Dataset completo (de comienzo a fin)

![Dataset entero](https://github.com/toblich/itba-scientific/raw/main/out/fase-general.png "Duración completa del dataset")

### Primer pestañeo rápido

![Primer pestañeo rápido](https://github.com/toblich/itba-scientific/raw/main/out/fase-pestaneo_rapido%20(0:01%20~%200:04).png "Primer pestañeo rápido")

### Baseline

![Baseline](https://github.com/toblich/itba-scientific/raw/main/out/fase-baseline%20(0:06%20~%201:04).png "Baseline")

### Segundo pestañeo rápido

![Segundo pestañeo rápido](https://github.com/toblich/itba-scientific/raw/main/out/fase-pestaneo_rapido%20(1:05%20~%201:06).png "Segundo pestañeo rápido")

### Tos

![Tos](https://github.com/toblich/itba-scientific/raw/main/out/fase-tos%20(1:08%20~%202:10).png "Tos")

### Tercer pestañeo rápido

![Tercer pestañeo rápido](https://github.com/toblich/itba-scientific/raw/main/out/fase-pestaneo_rapido%20(2:11%20~%202:13).png "Tercer pestañeo rápido")

### Respirar hondo

![Respirar hondo](https://github.com/toblich/itba-scientific/raw/main/out/fase-respira_hondo%20(2:14%20~%203:10).png "Respirar hondo")

### Último pestañeo rápido

![Último pestañeo rápido](https://github.com/toblich/itba-scientific/raw/main/out/fase-pestaneo_rapido%20(10:12%20~%2010:13).png "Último pestañeo rápido")

### Pestañeo código

![Pestañeo código](https://github.com/toblich/itba-scientific/raw/main/out/fase-pestaneo_codigo%20(10:14%20~%2011:00).png "Pestañeo código")

### Entrenar diferentes modelos y compararlos

Dado que el objetivo es distinguir el pestañeo rápido de las demás acciones, primero se crea una columna en el dataset con la clase binaria: `1` para el pestañeo rápido, `0` para las demás tareas. Luego, se eliminan los datos que no entran en ninguna clase (algunos pocos segundos entre etapas a veces) y se eliminaron algunas features que parecen poco relevantes o fallidas, en particular las ondas delta y lo relacionado a la atención y meditación. El siguiente paso es escalar los datos mediante un `MinMaxScaler`[^scaler], dado que algunos de los modelos usados son sensibles a la diferencia de escala entre variables.

[^scaler]: Esto surgió de los experimentos previos, en donde la ejecución de algunos modelos generaba warnings al respecto y sugería usar `MinMaxScaler` o `StandardScaler`. Habiendo probado ambos, se obtuvieron mejores resultados con el primero.

Luego, se separa el dataset en conjuntos de entrenamiento y prueba. Para ello, se tomaron aproximadamente los primeros 5 minutos del dataset para entrenamiento, y lo restante para testeo, con la salvedad de que se descartaron las etapas de tos y pestañeo código por involucrar también bastante pestañeo, lo que definitivamente resulta confuso para el modelo[^train-tos]. Dado que en cada conjunto las clases están muy desbalanceadas, ya que hay unos pocos segundos de pestañeo rápido por cada minuto de otra tarea, se hace un muestreo aleatorio de la clase mayoritaria de forma tal de tener misma cantidad de registros de pestañeo rápido (clase `1`) y de las otras (clase `0`).

[^train-tos]: No se incluye en el cuerpo del informe, pero se hicieron experimentos incluyendo la tos en el entrenamiento y los resultados siempre fueron peores.

A continuación, se entrena cada modelo sobre el conjunto de training, se lo prueba contra el conjunto de testing, y se toman algunas métricas de error. En particular, se midió la clasificación por su precisión, su matriz de confusión, y su área bajo la curva ROC.

La siguiente tabla y el gráfico posterior resumen los modelos probados con sus métricas de error

| Modelo                   | Precisión | Matriz de confusión       |
| ------------------------ | --------- | ------------------------- |
| Regresión Logística      | 0.8292    | [[3436  190] [1049 2577]] |
| Random Forest            | 0.6629    | [[3616   10] [2435 1191]] |
| SVM Lineal               | 0.8333    | [[3434  192] [1017 2609]] |
| SVM Polinómico (grado 3) | 0.7996    | [[3475  151] [1302 2324]] |
| SVM Radial (rbf)         | 0.8315    | [[3476  150] [1072 2554]] |
| Red neuronal[^nn]        | 0.8127    | [[3461  165] [1193 2433]] |

[^nn]: Luego de algunas pruebas manuales con redes de entre una y dos capas ocultas, se terminó eligiendo por una red de 1 capa oculta con 40 nodos en la misma, ya que agregando muchos más nodos se notaba una caída considerable en la performance, debido claramente al overfitting por el tamaño del conjunto de entrenamiento y la gran variabilidad de las redes con muchas neuronas.

![Gráficos ROC](https://github.com/toblich/itba-scientific/raw/main/out/roc.png "Gráficos ROC")

## Conclusiones

A partir de los resultados obtenidos, el modelo que mejor performa en este caso parece ser el SVM lineal. Sin embargo, el SVM radial también tiene resultados muy similares, y la regresión logística sorprendentemente también arroja resultados decentes. Cabe destacar igual que ningún modelo superó el 83.3% de precisión, lo que implica que, subjetivamente, ninguno es demasiado confiable de todos modos. De todos modos, es sorprendente que el Random Forest arroja los resultados de menor precisión total, pero observando la matriz de confusión se nota que genera muchos falsos negativos pero a su vez es el modelo que menos falsos positivos genera.

-------------

## Anexo

- Para obtener el dataset original, descargárselo de [este link](https://drive.google.com/file/d/1ByQDK4ZPxbqw7T17k--avTcgSCCzs3vi/view?usp=sharing).
- Para ver el código completo de este ejercicio, ver [este archivo](https://github.com/toblich/itba-scientific/blob/main/ejerciciofinal.py). Notar que la variable `PREPROCESS_DATASET` marca si se ejecuta solo el pre-procesamiento del dataset, o si se toma del disco el dataset ya pre-procesado y se hacen los gráficos y modelos.
- Para ver las visualizaciones de todas las etapas, ver las imágenes en [el directorio `out/`](https://github.com/toblich/itba-scientific/tree/main/out)
