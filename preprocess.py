import os
import pickle
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from librosa.util import normalize
from numpy.random import RandomState

rootDir = './wavs'
targetDir = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


def mel_gan_handler(x, fft_length = 1024, hop_length = 256, sr = 22050):
    wav = normalize(x)
    p = (fft_length - hop_length) // 2
    wav = np.squeeze(np.pad(wav, (p, p), "reflect"))
    fft = librosa.stft(
                       wav, 
                       n_fft = fft_length, 
                       hop_length = hop_length,
                       window = 'hann',
                       center = False
                     )
    # 這裡的 abs 是 sqrt(實部**2 + 虛部**2)
    mag = abs(fft)
    mel_basis = mel(sr=sr, n_fft=1024, fmin = 0.0 , fmax=None, n_mels=80)
    mel_output = np.dot(mel_basis,mag)
    log_mel_spec = np.log10(np.maximum(1e-5,mel_output)).astype(np.float32)
    return log_mel_spec
    
# resample 到 22050 
new_rate = 22050
for subdir in sorted(subdirList):
    print(subdir)
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    for fileName in sorted(fileList):
        x, fs = sf.read(os.path.join(dirName,subdir,fileName))
        # change sr -> 22050 Since mel_gan use 22050
        # print(fs, new_rate)
        x = librosa.resample(y=x, orig_sr=fs, target_sr=new_rate)
        S = mel_gan_handler(x=x)   
        np.save(os.path.join(targetDir, subdir, fileName[:-5]),
                 S.astype(np.float32), allow_pickle=False)
    print(f"Done --- {subdir}")