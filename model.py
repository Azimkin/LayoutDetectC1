from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn

WORD_MAX_LEN = 64

def _create_alphabet():
    import string
    latin = string.ascii_lowercase  # a-z (26)
    cyrillic = ''.join(chr(c) for c in range(0x430, 0x450)) + 'ё'  # а-я + ё (33)
    digits = string.digits  # 0-9 (10)
    symbols = string.punctuation + ' '  # 33

    return latin + cyrillic + digits + symbols

_ALPHABET = _create_alphabet()
_CHAR_2_IDX = {c: i for i, c in enumerate(_ALPHABET)}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WordClass(Enum):
    SYMBOL = 0
    RUS = 1
    ENG = 2
    RUS_ON_ENG = 3
    ENG_ON_RUS = 4


class LayoutDetectC1(nn.Module):
    def __init__(self, num_classes, num_chars):
        super().__init__()
        self.conv1 = nn.Conv1d(num_chars, 256, kernel_size=7)

        self.pool1 = nn.MaxPool1d(3)

        self.conv2 = nn.Conv1d(256, 256, kernel_size=7)

        self.pool2 = nn.MaxPool1d(3)

        self.conv3 = nn.Conv1d(256, 256, kernel_size=3)

        self.fc1 = nn.Linear(256 * 2, 1024)

        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x

    @staticmethod
    def encode(text):
        text = text.lower()[:WORD_MAX_LEN]
        x = torch.zeros(len(_ALPHABET), WORD_MAX_LEN)

        for i, ch in enumerate(text):
            if ch in _CHAR_2_IDX:
                x[_CHAR_2_IDX[ch], i] = 1.0

        return x

    def classify_word(self, word: str) -> WordClass:
        x = LayoutDetectC1.encode(word)
        x = x.unsqueeze(0).to(_DEVICE)

        with torch.no_grad():  # отключаем градиенты
            outputs = self(x)
            pred = outputs.argmax(dim=1)

        return WordClass(pred.item())

    def classify_phrase(self, phrase: str) -> Tuple[WordClass, list[WordClass]]:
        """
        Классифицирует фразу символ за символом.
        :param phrase: Фраза для классификации
        :return: Класс (медиана со всех слов)
        """
        words = phrase.lower().split(" ")
        results: list[WordClass] = []
        for word in words:
            results.append(self.classify_word(word))

        from statistics import median
        return WordClass(int(median(r.value for r in results))), results

    @staticmethod
    def empty():
        return LayoutDetectC1(num_classes=5, num_chars=len(_ALPHABET)).to(_DEVICE)

    @staticmethod
    def load(model_path: str) -> "LayoutDetectC1":
        model = LayoutDetectC1.empty()
        model.load_state_dict(torch.load(model_path, map_location=_DEVICE))
        model.to(_DEVICE)
        model.eval()
        return model
