# Identificador de Canciones
# Alejandra Velasco y Miguel Perez
# Buscamos que este programa determine a partir de un audio que cancion contiene sin importar las conversaciones de fondo, ruido o baja calidad del mismo.

import sys
import subprocess


def load_libraries():
    # Descargar paquetes necesarios:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    except Exception as e:
        print(e)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice"])
    except Exception as e:
        print(e)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    except Exception as e:
        print(e)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    except Exception as e:
        print(e)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "typing"])
    except Exception as e:
        print(e)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    except Exception as e:
        print(e)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub"])
    except Exception as e:
        print(e)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wave"])
    except Exception as e:
        print(e)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "contextlib"])
    except Exception as e:
        print(e)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "glob"])
    except Exception as e:
        print(e)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "os"])
    except Exception as e:
        print(e)


load_libraries()

# Librerias que utilizaremos
import os
import numpy as np
import scipy
from scipy import fft, signal
from scipy.io.wavfile import read
from scipy.fft import fft, fftfreq
import glob
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
import wave
import contextlib
from pydub import AudioSegment
from pydub.playback import play


# Cargamos nuestra base de datos (canciones en buena calidad) y nuestras muestras de audio

"""Ya que tenemos nuestras canciones, debemos poder identificarlas de una manera concisa, 
esto sera por medio de los picos resultantes de aplicarles la transformada rapida de furier. 
Estos representaran la identidad de cada cancion"""

# key = hash (numero grande con binned frequencies y delta), value = lista de tuples con tiempo de pico ancla y song_id
# database es la union de todos los hashes


def graphs(hz, song_data, name):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(name)

    # graph 1

    time_to_plot = np.arange(hz * 1, hz * 1.3, dtype=int)
    axs[0, 0].plot(time_to_plot, song_data[time_to_plot])
    axs[0, 0].set_title("Sound Signal Over Time")
    axs[0, 0].set_xlabel("Time Index")
    axs[0, 0].set_ylabel("Magnitude")

    # graph 2
    N = len(song_data)
    fft = scipy.fft.fft(song_data)
    transform_y = 2.0 / N * np.abs(fft[0 : N // 2])
    transform_x = scipy.fft.fftfreq(N, 1 / hz)[: N // 2]
    axs[1, 0].plot(transform_x, transform_y)
    axs[1, 0].set_title("Fourier Transform")
    axs[1, 0].set_xlabel("Frequency (Hz)")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].set_xlim(0, 2000)

    # graph 3
    peaks, props = signal.find_peaks(transform_y, prominence=0, distance=1000)
    n_peaks = 15
    largest_peaks_indices = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
    largest_peaks = peaks[largest_peaks_indices]
    axs[0, 1].plot(transform_x, transform_y, label="Spectrum")
    axs[0, 1].scatter(
        transform_x[largest_peaks],
        transform_y[largest_peaks],
        color="r",
        zorder=10,
        label="Constrained Peaks",
    )
    axs[0, 1].set_title("Picos audio completo")
    axs[0, 1].set_xlabel("Frequency (Hz)")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].set_xlim(0, 1000)

    # graph 4
    window_length_seconds = 1
    window_length_samples = int(window_length_seconds * hz)
    window_length_samples += window_length_samples % 2
    frequencies, times, stft = signal.stft(
        song_data,
        hz,
        nperseg=window_length_samples,
        nfft=window_length_samples,
        return_onesided=True,
    )

    constellation_map = []

    for time_idx, window in enumerate(stft.T):
        spectrum = abs(window)
        peaks, props = signal.find_peaks(spectrum, prominence=0, distance=200)
        n_peaks = 5
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])
    axs[1, 1].scatter(*zip(*constellation_map))
    axs[1, 1].set_title("Picos por ventana (1s) acumulados")
    axs[1, 1].set_xlabel("Tiempo en seg")
    axs[1, 1].set_ylabel("Frequency (hz)")
    axs[1, 1].set_ylim(0, 1500)

    fig.tight_layout()
    fig.show()


# hz = samplerate = samples per second = muestras por segundo
def encontrar_picos(hz, song_data):
    intervalos = 0.5  # intervalos de la cancion en segundos
    muestras = int(hz * intervalos)  # cuantas muestras por intervalo
    picos_por_muestra = 15  # tomaremos los 15 picos mas prominentes por intervalo

    # Agregaremos ceros al array song_data para que pueda ser dividido en las muestras sin residuo y abarque toda la cancion
    song = np.pad(song_data, (0, muestras - song_data.size % muestras))

    # stft = Short Time Fourier Transform
    hz_muestra, segmentos_tiempo, stft = signal.stft(
        song, hz, nperseg=muestras, nfft=muestras, return_onesided=True
    )

    mapa_picos = []

    for idx_tiempo, interval in enumerate(stft.T):
        # print(idx_tiempo)
        spectrum = abs(interval)  # Valores reales, no complejos
        picos, propiedades = signal.find_peaks(spectrum, prominence=0, distance=200)

        picos_por_muestra = min(
            picos_por_muestra, len(picos)
        )  # solo los picos mas prominentes, maximo 20
        picos_maximos = np.argpartition(propiedades["prominences"], -picos_por_muestra)[
            -picos_por_muestra:
        ]

        for pico in picos[picos_maximos]:
            frecuencia = hz_muestra[pico]
            mapa_picos.append([idx_tiempo, frecuencia])

    return mapa_picos


def crear_diccionarios(picos, song_id):
    hashes = {}
    hz_max = 44100  # Frecuencia maxima en archivos wav es 44.1 kHz cuando usas 16 bits
    bits = 16
    for idx, (time, hz) in enumerate(picos):
        for time2, hz2 in picos[idx : idx + 100]:
            # compara un pico ancla con los siguientes 100 picos para obtener un array de frec1, frec2, tiempo entre ellas (10 veces mas memoria, 10000 mas rapido)
            delta = time2 - time
            if delta <= 1 or delta > 10:
                # si el pico esta muy cerca o muy lejos no lo contaremos
                continue

            binned_hz = hz / hz_max * (2**bits)
            binned_hz_2 = hz2 / hz_max * (2**bits)

            hash = int(binned_hz) | (int(binned_hz_2) << 10) | (int(delta) << 20)
            hashes[hash] = (time, song_id)

    return hashes  # hashes es un diccionario con key = hash (numero grande con binned frequencies y delta) y  value = tuple con tiempo de pico ancla y song_id


song_name_index = {}


def load_database():
    print("Loading Database, please wait...")
    dirname = os.path.dirname(__file__)
    files = os.path.join(dirname, "songs")

    canciones = glob.glob(files + "/*.wav")  # Agarra solo los archivos .wav
    database: Dict[int, List[Tuple[int, int]]] = {}
    # Go through each song, using where they are alphabetically as an id
    for index, filename in enumerate(tqdm(sorted(canciones))):
        song_name_index[index] = filename
        # Encontrar picos y hashes
        hz, song_data = read(filename)
        picos = encontrar_picos(hz, song_data)
        hashes = crear_diccionarios(picos, index)
        # For each hash, append it to the list for this hash
        for hash, tiempo_songid in hashes.items():
            if hash not in database:
                database[hash] = []
            database[hash].append(tiempo_songid)
    return database


def score_hashes_against_database(hashes, database):
    hits_por_cancion = {}
    for hash, (sample_time, _) in hashes.items():
        if hash in database:
            # checa si las binned frecuencies con su delta estan en el database
            matches = database[hash]
            # todas las ocurrencias, key = binned frequencies y delta, value = tuples con tiempo de pico ancla y song_id
            for source_time, song_index in matches:
                if song_index not in hits_por_cancion:
                    hits_por_cancion[song_index] = []
                hits_por_cancion[song_index].append((hash, sample_time, source_time))

    scores = {}
    for song_index, matches in hits_por_cancion.items():
        song_scores_by_offset = {}
        for hash, sample_time, source_time in matches:
            delta = source_time - sample_time
            if delta not in song_scores_by_offset:
                song_scores_by_offset[delta] = 0
            song_scores_by_offset[delta] += 1
            # print("Source time: ", source_time)

        max = (0, 0)
        for offset, score in song_scores_by_offset.items():
            if score > max[1]:
                max = (offset, score)

        scores[song_index] = max

    # Ordenar los scores usando dictionary comprehension, acomoda basado en los segundos valores del tuple en value de cada key
    scores = list(sorted(scores.items(), key=lambda x: x[1][1], reverse=True))

    return scores


slice_indexes = {}
last_file = 0


def get_song_slice(hashes, fname, rec_time):
    slice_indexes = {}
    accuracy = 5
    with contextlib.closing(wave.open(fname, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    windows = duration // rec_time
    windows = int(windows)
    windows = windows * accuracy - (accuracy - 1)
    slices = []
    global last_file
    last_file = windows
    for window in range(windows):
        t1 = (window * 1000) / accuracy * rec_time
        t2 = t1 + 1000 * rec_time

        newAudio = AudioSegment.from_wav(fname)
        newAudio = newAudio[t1:t2]
        slice_name = "song_slice" + str(window) + ".wav"
        newAudio.export(slice_name, format="wav")
        slices.append(slice_name)

    database_song: Dict[int, List[Tuple[int, int]]] = {}
    for index, filename in enumerate(tqdm(sorted(slices))):
        slice_indexes[index] = filename
        # Encontrar picos y hashes
        hz, song_data = read(filename)
        picos = encontrar_picos(hz, song_data)
        hashes2 = crear_diccionarios(picos, index)
        # For each hash, append it to the list for this hash
        for hash, tiempo_songid in hashes2.items():
            if hash not in database_song:
                database_song[hash] = []
            database_song[hash].append(tiempo_songid)

    score = score_hashes_against_database(hashes, database_song)[:1]
    slice_idx = score[0][0]
    slice_name = slice_indexes[slice_idx]
    pos = 10
    while slice_name[pos] != ".":
        pos += 1
    slice_position = int(slice_name[10:pos]) / accuracy * rec_time
    minutes = slice_position // 60
    seconds = slice_position - 60 * minutes
    print("Best song slice match at ", minutes, " min ", int(seconds), " sec")

    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, slice_name)
    newAudio = AudioSegment.from_wav(file)
    newAudio = newAudio[slice_position : slice_position + rec_time * 1000]
    newAudio.export("snippet.wav", format="wav")


database = load_database()


def top_3(file_name, time):
    # Load a short recording with some background noise
    hz_recording, song_data_recording = read(file_name)
    # Create the constellation and hashes
    picos = encontrar_picos(hz_recording, song_data_recording)
    hashes = crear_diccionarios(picos, None)

    scores = score_hashes_against_database(hashes, database)[:1]
    # Las mejores n matches (n=1)

    for song_id, score in scores:
        name = song_name_index[song_id].split(".")
        name = name[0]
        i = 0
        while name[i : i + 5] != "songs":
            i += 1
        name = name[i + 6 :]
        print("\nBest Match:")
        print(f"{name}: {score[1]} matches con desfase de {score[0]}\n")

    graphs(hz_recording, song_data_recording, "Recording")
    try:
        print("Finished recording \n")
        print("Finding best song slice match, please wait...")
        get_song_slice(hashes, song_name_index[song_id], time)
    except Exception as e:
        print(e)

    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, "snippet.wav")
    hz_song_slice, song_slice_data = read(file)
    try:
        graphs(hz_song_slice, song_slice_data, name)
    except Exception as e:
        print(e)


def record(duration):
    # Sampling frequency
    frequency = 44100

    print("\n\nRecording, please wait...")
    # to record audio from
    # sound-device into a Numpy
    recording = sd.rec(int(duration * frequency), samplerate=frequency, channels=1)

    # Wait for the audio to complete
    sd.wait()

    # using scipy to save the recording in .wav format
    # This will convert the NumPy array to an audio file with the given sampling frequency
    write("recording0.wav", frequency, recording)
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, "recording0.wav")
    top_3(file, duration)


def delete_extra_files():
    files = ["newsound.wav", "recording0.wav", "snippet.wav"]
    for i in files:
        dirname = os.path.dirname(__file__)
        file = os.path.join(dirname, i)
        if os.path.isfile(file):
            os.remove(file)
        else:
            pass

    for i in range(last_file):
        file_path = "song_slice" + str(i) + ".wav"

        dirname = os.path.dirname(__file__)
        file = os.path.join(dirname, file_path)
        if os.path.isfile(file):
            os.remove(file)
        else:
            pass


choice = int(input("Desea realizar una grabación? 1.Si 2.No --> "))
if choice == 1:
    while choice == 1:
        time = int(input("Duración en seg? "))
        record(time)
        delete_extra_files()
        choice = int(input("\nDesea realizar otra grabación? 1.Si 2.No --> "))


"""
Bibliography:

Li, A., & Wang, C. (n.d.). An Industrial-Strength Audio Search Algorithm. https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf
"""

k = input("press close to exit")
