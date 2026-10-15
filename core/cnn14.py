"""PANNs CNN14 (Kong et al. 2020, AudioSet), inference only.

Architecture and weights from https://github.com/qiuqiangkong/audioset_tagging_cnn (MIT licence).
The log-mel front end uses torch.stft instead of torchlibrosa; the mel filterbank is loaded from the
checkpoint, so features match the original model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    sample_rate, n_fft, hop, n_mels = 32000, 1024, 320, 64

    def __init__(self, classes_num=527):
        super().__init__()
        self.register_buffer("window", torch.hann_window(self.n_fft, periodic=True))
        self.register_buffer("melW", torch.zeros(self.n_fft // 2 + 1, self.n_mels))
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num)

    @classmethod
    def from_checkpoint(cls, path) -> "Cnn14":
        state = torch.load(str(path), map_location="cpu", weights_only=True)["model"]
        model = cls()
        model.melW.copy_(state.pop("logmel_extractor.melW"))
        state = {k: v for k, v in state.items() if not k.startswith("spectrogram_extractor.")}
        model.load_state_dict(state, strict=False)
        missing = set(model.state_dict()) - set(state) - {"window", "melW"}
        assert not missing, f"CNN14 checkpoint missing {sorted(missing)}"
        return model.eval()

    def logmel(self, waveform):  # (B, samples) -> (B, 1, frames, mels)
        spec = torch.stft(waveform, self.n_fft, self.hop, window=self.window, center=True,
                          pad_mode="reflect", return_complex=True).abs() ** 2
        mel = spec.transpose(1, 2) @ self.melW
        return (10.0 * torch.log10(mel.clamp(min=1e-10))).unsqueeze(1)

    def forward(self, waveform):
        x = self.logmel(waveform)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for block in (self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5):
            x = block(x)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = x.mean(dim=3)
        x = x.max(dim=2).values + x.mean(dim=2)
        embedding = F.relu_(self.fc1(x))
        clipwise = torch.sigmoid(self.fc_audioset(embedding))
        return clipwise, embedding
