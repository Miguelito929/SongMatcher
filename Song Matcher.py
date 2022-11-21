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
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tkinter"])
    except Exception as e:
        print(e)
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "matplotlib.backends.backend_tkagg",
            ]
        )
    except Exception as e:
        print(e)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "FigureCanvasTkAgg"]
        )
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
import tkinter as tk
from tkinter import Tk, Label, Button
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()
root.title("Song Matcher")
root.resizable(1, 1)

root.config(cursor="arrow")
root.config(bg="lightblue")
root.config(bd=15)
root.config(relief="ridge")

frame2 = Frame(root, width=0, height=0)
frame2.pack(fill="both", expand=1, side=BOTTOM)
frame2.config(cursor="heart")
frame2.config(bg="lightblue")
frame2.config(bd=25)
frame2.config(relief="sunken")

frame1 = Frame(root, width=0, height=0)
frame1.pack(fill="both", expand=1, side=BOTTOM)
frame1.config(cursor="heart")
frame1.config(bg="lightblue")
frame1.config(bd=25)
frame1.config(relief="sunken")


def graphs(hz, song_data, fr, name, pos):
    for widget in fr.winfo_children():
        widget.destroy()
    if name != None:
        title = name + " at " + pos
        label2 = Label(frame2, text=title)
        label2.pack()
    else:
        label1 = Label(frame1, text="Recording")
        label1.pack()

    # graph 1
    figure1 = plt.Figure(figsize=(3, 2), dpi=100)
    axs1 = figure1.add_subplot(111)
    time_to_plot = np.arange(hz * 1, hz * 1.3, dtype=int)
    axs1.plot(time_to_plot, song_data[time_to_plot])
    scatter1 = FigureCanvasTkAgg(figure1, fr)
    scatter1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    axs1.set_title("Sound Signal ")
    axs1.set_xlabel("Time")
    axs1.set_ylabel("Magnitude")

    # graph 2
    figure2 = plt.Figure(figsize=(3, 2), dpi=100)
    axs2 = figure2.add_subplot(111)
    N = len(song_data)
    fft = scipy.fft.fft(song_data)
    transform_y = 2.0 / N * np.abs(fft[0 : N // 2])
    transform_x = scipy.fft.fftfreq(N, 1 / hz)[: N // 2]
    axs2.plot(transform_x, transform_y)
    scatter2 = FigureCanvasTkAgg(figure2, fr)
    scatter2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    axs2.set_title("Fourier Transform ")
    axs2.set_xlabel("Frequency (Hz)")
    axs2.set_ylabel("Amplitude")
    axs2.set_xlim(0, 2000)

    # graph 3
    figure3 = plt.Figure(figsize=(3, 2), dpi=100)
    axs3 = figure3.add_subplot(111)
    peaks, props = signal.find_peaks(transform_y, prominence=0, distance=1000)
    n_peaks = 15
    largest_peaks_indices = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
    largest_peaks = peaks[largest_peaks_indices]
    axs3.plot(transform_x, transform_y, label="Spectrum")
    axs3.scatter(
        transform_x[largest_peaks],
        transform_y[largest_peaks],
        color="r",
        zorder=10,
        label="Constrained Peaks",
    )
    scatter3 = FigureCanvasTkAgg(figure3, fr)
    scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    axs3.set_title("Picos audio ")
    axs3.set_xlabel("Frequency (Hz)")
    axs3.set_ylabel("Amplitude")
    axs3.set_xlim(0, 1000)

    # graph 4
    figure4 = plt.Figure(figsize=(3, 2), dpi=100)
    axs4 = figure4.add_subplot(111)

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

    axs4.scatter(*zip(*constellation_map))
    scatter4 = FigureCanvasTkAgg(figure4, fr)
    scatter4.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    axs4.set_title("Picos por seg acumulados")
    axs4.set_xlabel("Tiempo en seg")
    axs4.set_ylabel("Frequency (hz)")
    axs4.set_ylim(0, 1500)


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


# key = hash (numero grande con binned frequencies y delta), value = lista de tuples con tiempo de pico ancla y song_id
# database es la union de todos los hashes
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
    new_time = str(int(minutes)) + " min " + str(int(seconds)) + " sec"
    return new_time


database = load_database()


def best_match(file_name, time):
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

    graphs(hz_recording, song_data_recording, frame1, None, None)

    print("Finished recording \n")
    print("Finding best song slice match, please wait...")
    pos = get_song_slice(hashes, song_name_index[song_id], time)

    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, "snippet.wav")
    hz_song_slice, song_slice_data = read(file)
    graphs(hz_song_slice, song_slice_data, frame2, name, pos)
    delete_extra_files()
    return [name, pos]


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
    name_pos = best_match(file, duration)
    return name_pos


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


# Interfaz

fields = ("Duración", "Canción", "Posición", "Canciones disponibles")


def Grabar(entries):
    # period rate:
    duracion = int(entries["Duración"].get())
    name_pos = record(duracion)
    entries["Canción"].delete(0, tk.END)
    entries["Canción"].insert(0, name_pos[0])
    entries["Posición"].delete(0, tk.END)
    entries["Posición"].insert(0, name_pos[1])


song_tuples = list(sorted(song_name_index.items(), key=lambda x: x[1][1]))
song_list = []
for i in song_tuples:
    song_list.append(i[1])
song_str = ""
for i in song_list:
    n = i.split(".")
    n = n[0]
    j = 0
    while n[j : j + 5] != "songs":
        j += 1
    n = n[j + 6 :]
    if i != song_list[-1]:
        song_str = song_str + n + ", "
    else:
        song_str = song_str + n


def makeform(root, fields):
    entries = {}
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=20, text=field + ": ", anchor="w")
        ent = tk.Entry(row, width=180)
        if field == "Duración":
            ent.insert(0, "8")
        elif field == "Canciones disponibles":
            ent.insert(0, song_str)
        else:
            ent.insert(0, "(dejar en blanco)")
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries[field] = ent
    return entries


if __name__ == "__main__":
    ents = makeform(root, fields)
    b1 = tk.Button(root, text="Comenzar grabación", command=(lambda e=ents: Grabar(e)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(root, text="Quit", command=root.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    root.mainloop()
