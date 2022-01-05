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

import os
import pandas as pd

# id/char 사전 만들기
def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    freq_list = ch_labels["freq"]

    for (id_, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


# 문장을 id로
def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        # 사전에 없는 경우 넘어가라 -> 그냥 묵음처리나 마찬가지.
        except KeyError:
            continue

    return target[:-1]


# 한글 전사에서 빈번하게 사용된 상위 2000개 단어 id 사전 만들기
def generate_character_labels(transcripts, labels_dest):
    print('create_char_labels started..')

    label_list = list()
    label_freq = list()

    for transcript in transcripts:
        for ch in transcript:
            if ch not in label_list:
                label_list.append(ch)
                label_freq.append(1)
            else:
                label_freq[label_list.index(ch)] += 1

    # sort together Using zip
    label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
    label = {'id': [0, 1, 2], 'char': ['<pad>', '<sos>', '<eos>'], 'freq': [0, 0, 0]}

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label['id'].append(idx + 3)
        label['char'].append(ch)
        label['freq'].append(freq)

    label['id'] = label['id'][:986]
    label['char'] = label['char'][:986]
    label['freq'] = label['freq'][:986]

    label_df = pd.DataFrame(label)
    label_df.to_csv(os.path.join(labels_dest, "cssiri_character_vocabs.csv"), encoding="utf-8", index=False)


def generate_character_script(audio_paths, transcripts, labels_dest):
    print('create_script started..')
    char2id, id2char = load_label(os.path.join(labels_dest, "cssiri_character_vocabs.csv"))

    with open(os.path.join("transcripts.txt"), "w") as f:
        for audio_path, transcript in zip(audio_paths, transcripts):
            char_id_transcript = sentence_to_target(transcript, char2id)
            audio_path = audio_path.replace('txt', 'wav')
            f.write(f'{audio_path}\t{transcript}\t{char_id_transcript}\n')
