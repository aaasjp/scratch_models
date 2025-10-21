
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hifi_gan'))
import librosa
import numpy as np
import torch
from hifi_gan.models import Generator
from hifi_gan.meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from hifi_gan.env import AttrDict
import json
from scipy.io.wavfile import write

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def get_mel_spectrogram(audio, sample_rate, n_fft, hop_length, win_length, n_mels, fmin, fmax):
    """提取梅尔频谱"""
    # 确保音频是float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # 提取梅尔频谱
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    print(f"mel shape: {mel.shape}")
    print(f"mel: {mel}")
    print("--------------------------------")
    '''
    # 转换到对数域
    mel_db = librosa.power_to_db(mel, ref=np.max)
    print(f"mel_db shape: {mel_db.shape}")
    print(f"mel_db: {mel_db}")
    print("--------------------------------")
    return mel_db.T  # [T, n_mels]
    '''
    return mel #[n_mels, T]

def convert_mel_format(mel):# [T, n_mels]

    mel_magnitude = np.sqrt(mel + 1e-9)  # 对应torch中的sqrt操作
    
    # 使用torch风格的对数压缩
    C = 1
    clip_val = 1e-5
    mel_compressed = np.log(np.clip(mel_magnitude, a_min=clip_val, a_max=None) * C)
    
    return mel_compressed #[n_mels, T]


def convert_mel_to_wav(mel, device='cpu',output_dir='test_generated_files'):

    config_file = './hifi_gan/pretrained_models/LJ_V1/config.json'
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint('./hifi_gan/pretrained_models/LJ_V1/generator_v1', device)
    generator.load_state_dict(state_dict_g['generator'])


    generator.eval()
    generator.remove_weight_norm()

    x=convert_mel_format(mel)
    x=torch.FloatTensor(x).to(device)
    x=x.unsqueeze(0)
    print(f"x shape: {x.shape}")
    print(f"x: {x}")
    print("--------------------------------")

    with torch.no_grad():
        """将梅尔频谱转换为音频"""
        # x.shape: [1, n_mels, T]
        y_g_hat = generator(x)

        audio = y_g_hat.squeeze()
        print(f"audio shape: {audio.shape}")
        print(f"audio: {audio}")
        print("--------------------------------")
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        print(f"audio shape: {audio.shape}")
        print(f"audio: {audio}")
        print("--------------------------------")
        output_file = os.path.join(output_dir, 'test_generated.wav')
        write(output_file, h.sampling_rate, audio)
        print(output_file)

if __name__ == "__main__":
    audio, sample_rate = librosa.load("test_files/audio_000001.wav", sr=None)
    mel = get_mel_spectrogram(audio, sample_rate, 1024, 256, 1024, 80, 0, 8000)
    print(f"mel shape: {mel.shape}")
    print(f"mel: {mel}")
    print("--------------------------------")
    convert_mel_to_wav(mel=mel, device='cpu',output_dir='test_generated_files')