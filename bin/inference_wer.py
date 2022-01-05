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
from tools import wer, cer, revise
from tqdm import tqdm
import shutil

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)



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
parser.add_argument('--transcript_path', type=str, required=True)
parser.add_argument('--dst_path', type=str, required=False, default = './outputs')
parser.add_argument('--device', type=str, required=False, default='cpu')
opt = parser.parse_args()

# test wav 폴더 전체에 대해 inference 하는 경우
path_test_wav = opt.audio_path
path_transcript = opt.transcript_path
vocab = KsponSpeechVocabulary('data/vocab/cssiri_character_vocabs.csv')
model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(opt.device)
dst_path = os.path.join(opt.dst_path, 'wer_results_{}.txt'.format('_'.join(opt.model_path.split('/')[-3:-1])))
sentences = []
test_folder = os.listdir(path_test_wav)
# 순서대로 문장 길이, 맞은 갯수, 삭제 단어 갯수, 대체 단어 갯수, 삭제 단어 갯수, 대체 단어 갯수
Length_wer, NumCor, NumSub, NumDel, NumIns = 0,0,0,0,0
# 순서대로 편집거리, 문장 길이
Dist, Length_cer = 0, 0
with open(path_transcript,'r') as f:
    transcript = list(map(lambda x: x.split('\t')[1] , f.readlines()))

# 체크용 print

print(f"{(file_num:=len(test_folder))} files test starts...")
print(f"CER/WER caculation: {bool(opt.transcript_path)}")
if opt.transcript_path:
    print(f"-> transcript: {opt.transcript_path}")
print(f"Output destination: {opt.dst_path}")

if os.path.isfile(dst_path):
    shutil.rmtree(dst_path)

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
    Length_wer += len(transcript[i].split())
    numCor, numSub, numDel, numIns, WER = wer(transcript[i], revise(sentence))
    dist, length_cer, CER = cer(transcript[i], revise(sentence))
    NumCor += numCor; NumSub += numSub; NumDel += numDel; NumIns += numIns
    Dist += dist; Length_cer += length_cer
    with open(dst_path, "a") as f:
        f.write(f"{i})" + f" [{test_wav}]" + '\n')
        f.write("[transcript] " + f"{transcript[i]}" + '\n')
        f.write("[prediction] " + f"{revise(sentence)}" + '\n')
        f.write("Ncor: " + f'{str(numCor)}' + '\t')
        f.write("Nsub: " + f'{str(numSub)}' + '\t')
        f.write("Ndel: " + f'{str(numDel)}' + '\t')
        f.write("Nins: " + f'{str(numIns)}' + '\n')
        f.write("Lev distance: " + f'{str(dist)}' + '\t')
        f.write("length: " + f'{str(length_cer)}' + '\n')
        f.write("WER: " + f'{str(WER)}'+ '\t')
        f.write("CER: " + f'{str(CER)}'+ '\t')
        f.write("\n\n")
with open(dst_path, "a") as f:
        f.write("[최종결과]" + '\n')
        f.write("Ncor: " + f'{str(NumCor)}' + '\t')
        f.write("Nsub: " + f'{str(NumSub)}' + '\t')
        f.write("Ndel: " + f'{str(NumDel)}' + '\t')
        f.write("Nins: " + f'{str(NumIns)}' + '\n')
        f.write("Lev distance: " + f'{str(Dist)}' + '\t')
        f.write("length: " + f'{str(Length_cer)}' + '\n')
        f.write("WER: " + f'{(NumSub + NumDel + NumIns) / (float) (Length_wer)}'+ '\t')
        f.write("CER: " + f'{(Dist) / (float) (Length_cer)}'+ '\t')
        f.write("\n\n")

print(f"{file_num}files test completed...")