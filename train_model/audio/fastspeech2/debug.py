import librosa

audio, sr = librosa.load('./corpus/audio_000001.wav', sr=None)

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=22050,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    fmin=0,
    fmax=8000
)

print("------------mel--------------------")
print(mel.shape)
print(mel)
print("------------mel_db--------------------")
mel_db = librosa.power_to_db(mel, ref=1.0)
print(mel_db.shape)
print(mel_db)
print("-------------reverse to mel-------------------")
mel_reverse = librosa.db_to_power(mel_db, ref=1.0)
print(mel_reverse.shape)
print(mel_reverse)
print("--------------------------------")


