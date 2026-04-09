# char-cnn-layout-detect

Небольшой проект на PyTorch для определения раскладки текста по фразе или слову.

Модель `LayoutDetectC1` классифицирует вход как один из классов:

- `SYMBOL` — символы/прочее
- `RUS` — русский текст
- `ENG` — английский текст
- `RUS_ON_ENG` — русский текст на английской раскладке
- `ENG_ON_RUS` — английский текст на русской раскладке

## Требования

- Python 3.10+
- `torch`

## Запуск

1. Убедитесь, что файл модели лежит по пути `out/LayoutDetectC1.pth`.
2. Запустите:

```bash
python main.py
```

После запуска программа попросит ввести фразу и выведет предсказанный класс и список классов по словам.

## Flask API

Запуск:

```bash
python api.py
```

Примеры запросов:

```bash
curl -X POST http://127.0.0.1:5000/word \
  -H "Content-Type: application/json" \
  -d '{"word":"ghbdtn"}'

curl -X POST http://127.0.0.1:5000/phrase \
  -H "Content-Type: application/json" \
  -d '{"phrase":"ghbdtn world"}'
```

## Пример

```text
Enter phrase: ghbdtn world
Class: WordClass.ENG_ON_RUS
Values: [<WordClass.ENG_ON_RUS: 4>, <WordClass.ENG: 2>]
```

## Как это работает

Текст приводится к нижнему регистру, обрезается до 64 символов и кодируется как one-hot по символам. Затем свёрточная сеть делает предсказание для каждого слова, а итоговый класс фразы берётся как медиана по словам.

## Структура проекта

- `main.py` — точка входа
- `model.py` — модель и логика классификации