import torch
import torch.nn as nn
import numpy as np
import librosa

"""
Author: qiuqiangkong
Date: Apr 1, 2023, 2:17 PM GMT+8
Commit Hash: 1685d8f6d55f40f3fd9112f874306a2d2b193803
License: MIT
Project URL: https://github.com/qiuqiangkong/torchlibrosa

This class is used to calculate the logmel spectrogram using pytorch. 
The mel filter bank is the pytorch implementation of as librosa.filters.mel.
"""

class LogmelFilterBank(nn.Module):
    def __init__(self, sr=22050, n_fft=2048, n_mels=64, fmin=0.0, fmax=None, 
        is_log=True, ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        r"""Calculate logmel spectrogram using pytorch. The mel filter bank is 
        the pytorch implementation of as librosa.filters.mel 
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        if fmax == None:
            fmax = sr//2

        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        r"""Calculate (log) mel spectrogram from spectrogram.

        Args:
            input: (*, n_fft), spectrogram
        
        Returns: 
            output: (*, mel_bins), (log) mel spectrogram
        """

        # Mel spectrogram
        print(input.shape)
        print(self.melW.shape)
        input = input.unsqueeze(1)
        input = input.transpose(2, 3)
        mel_spectrogram = torch.matmul(input, self.melW)
        # (*, mel_bins)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output


    def power_to_db(self, input):
        r"""Power to db, this function is the pytorch implementation of 
        librosa.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise librosa.util.exceptions.ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db, max=np.inf)

        return log_spec

logmelFilterBank = LogmelFilterBank()
