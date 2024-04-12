import os
import torch
import torchaudio
from torch.utils.data import dataset, DataLoader
import torch.nn.functional as F
from layer import *

class MusicNetSpectrogramDataset(Dataset):
    def __init__(self, root_dir, files):
        """
        Args:
            root_dir (string): MusicNet 데이터셋의 디렉토리 경로.
            files (list): 사용할 오디오 파일 이름의 리스트.
        """
        self.root_dir = root_dir
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        waveform, sample_rate = torchaudio.load(file_path)

        # STFT 변환을 수행
        # n_fft, hop_length, 및 win_length는 필요에 따라 조정 가능
        spec = torchaudio.transforms.Spectrogram(n_fft=2048, win_length=2048, hop_length=512)(waveform)

        # 스펙트로그램의 로그 크기를 계산 (데이터의 스케일을 조정하기 위해)
        # 로그 변환 전에 1e-6을 더하여 수치적 안정성 보장
        log_spec = torch.log(spec + 1e-6)

        return log_spec, sample_rate

root_dir = '/Users/daeun/Desktop/pz_code/path/musicnet/train_data'
files = [
    '2202.wav', '2203.wav', '2204.wav',  # Flute
    '2241.wav', '2242.wav', '2243.wav', '2244.wav', '2288.wav', '2289.wav'  # Violin
]

dataset = MusicNetSpectrogramDataset(root_dir=root_dir, files=files)
train_data = DataLoader(dataset, batch_size=1, shuffle=True)

data_list = []
for spec, sample_rate in train_data:
    data_list.append(spec)
    # 여기에 모델 학습 로직을 추가합니다.
    # print(spec.shape, sample_rate)


# 가장 긴 시간 차원의 길이를 찾음
max_length = max(x.size(3) for x in data_list)
# print(max_length)

# block_size 정의
block_size = 1  # 예시 block_size

# max_length를 block_size의 배수로 설정
if max_length % block_size != 0:
    max_length = ((max_length // block_size) + 1) * block_size

# 데이터를 동일한 시간 차원 길이로 패딩
padded_data_list = [F.pad(input=x, pad=(0, max_length - x.size(3), 0, 0), mode='constant', value=0) for x in data_list]

