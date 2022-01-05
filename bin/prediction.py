# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from typing_extensions import Required
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch import Tensor
import os
from tools import revise
import pandas as pd
from tqdm import tqdm

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)

import datetime as dt

now = dt.datetime.now()
formattedDate = now.strftime("%y%m%d_%H시%M분")

def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'wav') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)

# require (x)- > required (o)
parser = argparse.ArgumentParser(description='KoSpeech')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--audio_path', type=str, required=True)
parser.add_argument('--submission', type=bool, required=True)
parser.add_argument('--device', type=str, required=False, default='cpu')
opt = parser.parse_args()

# test wav 폴더 전체에 대해 inference 하는 경우
path_test_wav = opt.audio_path
vocab = KsponSpeechVocabulary('data/vocab/cssiri_character_vocabs.csv')
model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(opt.device)
sentences = []
test_folder = os.listdir(path_test_wav)


# 체크용 print

print(f"{(file_num:=len(test_folder))} files prediction starts...")
print(f"Save results to excel(in 'submission' folder): {bool(opt.submission)}")

for i, test_wav in tqdm(enumerate(test_folder)):
    
    feature = parse_audio(os.path.join(path_test_wav, test_wav), del_silence=True)
    input_length = torch.LongTensor([len(feature)])

    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    # greedy_search -> recognize로 변경 + 변수 중에서 opt.device 삭제
    if isinstance(model, ListenAttendSpell):
        model.encoder.device = opt.device
        model.decoder.device = opt.device

        y_hats = model.recognize(feature.unsqueeze(0), input_length)
    elif isinstance(model, DeepSpeech2):
        model.device = opt.device
        y_hats = model.recognize(feature.unsqueeze(0), input_length)
    elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
        y_hats = model.recognize(feature.unsqueeze(0), input_length)

    sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
    sentences.append(revise(sentence))
    with open(f"./outputs/prediction_{formattedDate}.txt", "a") as f:
        f.write(f"{i})" + f" [{test_wav}]" + '\t' + f"{revise(sentence)}")
        f.write("\n\n")

print(f"{file_num}files prediction completed...")

# submission에 넣음

# 엑셀 저장 에러시 불러와서 해결

# with open('D:/code/cssiri/outputs/prediction_211213_03시19분.txt', 'r') as f:
#     sentences = [fs.split('\t')[-1].strip('\n') for _ in range(20000) if (fs:=f.readline()) != '\n']

if opt.submission:
    submission = pd.read_excel('./submission/submission_origin.xlsx')
    submission['ReadingLableText'] = sentences
    with pd.ExcelWriter('./submission/submission_cssiri.xlsx') as writer: 
        submission.to_excel(writer,sheet_name="cssiri")