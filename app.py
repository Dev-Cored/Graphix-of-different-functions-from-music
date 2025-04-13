
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import io
import base64
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""
Разделяет аудиофайл на сегменты указанной длины и возвращает средние значения амплитуды
"""
def segment_audio_and_get_means(audio_path, segment_length_seconds):

    y, sr = librosa.load(audio_path, sr=None)
    segment_length_samples = int(segment_length_seconds * sr)
    num_segments = len(y) // segment_length_samples
    segment_means = []

    for i in range(num_segments):
        start = i * segment_length_samples
        end = start + segment_length_samples
        segment = y[start:end]

        mean_amplitude = np.mean(np.abs(segment))
        segment_means.append(float(mean_amplitude))

    if len(y) % segment_length_samples != 0:
        last_segment = y[num_segments * segment_length_samples:]
        mean_amplitude = np.mean(np.abs(last_segment))
        segment_means.append(float(mean_amplitude))

    return segment_means, y, sr

"""
Создает три графика на основе данных амплитуды и возвращает их в виде base64 изображения
"""
def create_three_graphs(amplitude_data):

    if not isinstance(amplitude_data, np.ndarray):
        amplitude_data = np.array(amplitude_data, dtype=np.float32)

    segment_indices = np.arange(len(amplitude_data))

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # 1. Логарифмический график
    log_data = np.log(amplitude_data + 1e-10)
    axs[0].plot(segment_indices, log_data, color='blue')
    axs[0].set_title('Логарифмический график амплитуды')
    axs[0].set_xlabel('Номер сегмента')
    axs[0].set_ylabel('log(Амплитуда)')
    axs[0].grid(True)

    # 2. Показательный график (экспоненциальный)
    exp_data = np.exp(amplitude_data) - 1
    axs[1].plot(segment_indices, exp_data, color='red')
    axs[1].set_title('Показательный график амплитуды')
    axs[1].set_xlabel('Номер сегмента')
    axs[1].set_ylabel('exp(Амплитуда) - 1')
    axs[1].grid(True)

    # 3. График функции корня
    sqrt_data = np.sqrt(amplitude_data)
    axs[2].plot(segment_indices, sqrt_data, color='green')
    axs[2].set_title('График функции корня амплитуды')
    axs[2].set_xlabel('Номер сегмента')
    axs[2].set_ylabel('sqrt(Амплитуда)')
    axs[2].grid(True)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)

    data = base64.b64encode(buf.getbuffer()).decode("utf-8")
    plt.close(fig)

    return f"data:image/png;base64,{data}"

"""
Извлекает интересные характеристики из аудиофайла
"""
def generate_audio_features(y, sr):

    duration = float(librosa.get_duration(y=y, sr=sr))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    tempo_result = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(tempo_result[0]) if not isinstance(tempo_result[0], np.ndarray) else float(tempo_result[0][0])

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_cent = float(np.mean(cent))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mean_contrast = float(np.mean(contrast))

    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_indices = np.where(pitches > 0)
        if len(pitch_indices[0]) > 0:
            mean_pitch = float(np.mean(pitches[pitch_indices]))
        else:
            mean_pitch = 0.0
    except:
        mean_pitch = 0.0

    energy = float(np.sum(y ** 2) / len(y))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfccs = [float(x) for x in np.mean(mfccs, axis=1)]

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.sum(y_harmonic ** 2) / len(y_harmonic))
    percussive_energy = float(np.sum(y_percussive ** 2) / len(y_percussive))

    harm_perc_ratio = harmonic_energy / percussive_energy if percussive_energy > 0 else 0.0

    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))

    features = {
        'duration': f"{duration:.2f} сек.",
        'tempo': f"{tempo:.1f} BPM",
        'brightness': f"{mean_cent:.2f} Гц",
        'energy': f"{energy:.6f}",
        'harmonic_energy': f"{harmonic_energy:.6f}",
        'percussive_energy': f"{percussive_energy:.6f}",
        'harmonic_percussive_ratio': f"{harm_perc_ratio:.2f}",
        'loudness': f"{mean_rms:.6f}"
    }

    return features


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Файл не найден!')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('Файл не выбран!')
            return redirect(request.url)

        segment_length = float(request.form.get('segment_length', 1.0))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                segment_means, y, sr = segment_audio_and_get_means(filepath, segment_length)

                graphs_img = create_three_graphs(segment_means)

                audio_features = generate_audio_features(y, sr)

                min_amplitude = float(np.min(segment_means))
                max_amplitude = float(np.max(segment_means))
                avg_amplitude = float(np.mean(segment_means))

                return render_template('result.html',
                                       filename=filename,
                                       graphs_img=graphs_img,
                                       audio_features=audio_features,
                                       num_segments=len(segment_means),
                                       min_amplitude=min_amplitude,
                                       max_amplitude=max_amplitude,
                                       avg_amplitude=avg_amplitude)

            except Exception as e:
                import traceback
                flash(f'Ошибка при обработке файла: {str(e)}')
                flash(traceback.format_exc())
                return redirect(request.url)
        else:
            flash('Разрешены только файлы формата mp3, wav и ogg!')
            return redirect(request.url)

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)