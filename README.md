# Brain MRI Tumor Detection

Проект по учебной практике, посвящённый детектированию опухолей головного мозга на МРТ-снимках и сравнению пяти моделей компьютерного зрения на одном размеченном датасете.

## Что есть в проекте

- локальная структура проекта на Python;
- Colab-ноутбуки для обучения моделей;
- единые конфиги, утилиты и точка входа;
- сохранённые артефакты экспериментов.

## Модели

- `YOLOv8`
- `Faster R-CNN`
- `SSD`
- `DETR`
- `EfficientDet`

## Датасет

Используется датасет:

- [MRI for Brain Tumor with Bounding Boxes](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes/data)

Локально данные ожидаются в `data/raw/brain-tumor-mri`.

## Структура

```text
brain-mri-project/
├── main.py                              – основная точка входа
├── requirements.txt                     – список используемых зависимостей
├── README.md                            – краткое описание проекта
├── configs/
│   └── default.yaml                     – базовая конфигурация проекта
├── data/
│   ├── raw/                             – исходные данные
│   └── processed/                       – подготовленные данные
├── notebooks/
│   ├── colab_yolov8_brain_mri.ipynb     – обучение YOLOv8
│   ├── colab_faster_rcnn_brain_mri.ipynb – обучение Faster R-CNN
│   ├── colab_ssd_brain_mri.ipynb        – обучение SSD
│   ├── colab_detr_brain_mri.ipynb       – обучение DETR
│   └── colab_efficientdet_brain_mri.ipynb – обучение EfficientDet
├── scripts/                             – вспомогательные служебные файлы
├── src/
│   ├── dataset/
│   │   ├── prepare.py                   – подготовка датасета
│   │   ├── statistics.py                – расчёт статистики по данным
│   │   ├── torch_detection.py           – загрузка данных для PyTorch-моделей
│   │   └── yolo_detection.py            – подготовка данных для YOLO
│   ├── evaluation/
│   │   └── metrics.py                   – вычисление метрик качества
│   ├── models/
│   │   ├── yolo.py                      – работа с YOLOv8
│   │   ├── faster_rcnn.py               – работа с Faster R-CNN
│   │   ├── ssd.py                       – работа с SSD
│   │   ├── detr.py                      – работа с DETR
│   │   ├── efficientdet.py              – работа с EfficientDet
│   │   └── registry.py                  – единая регистрация моделей
│   ├── training/
│   │   ├── dataloaders.py               – подготовка загрузчиков данных
│   │   └── train.py                     – логика обучения
│   └── utils/
│       ├── config.py                    – работа с конфигурациями
│       ├── io.py                        – вспомогательные функции ввода-вывода
│       ├── paths.py                     – управление путями проекта
│       └── seed.py                      – фиксация случайности
├── artifacts/                           – сохранённые веса моделей
└── results/
    ├── logs/                            – результаты и служебные сводки
    ├── plots/                           – графики обучения
    └── predictions/                     – примеры предсказаний моделей
```

## Ноутбуки

В папке `notebooks/` лежат отдельные Colab-ноутбуки для каждой модели:

- `colab_yolov8_brain_mri.ipynb`
- `colab_faster_rcnn_brain_mri.ipynb`
- `colab_ssd_brain_mri.ipynb`
- `colab_detr_brain_mri.ipynb`
- `colab_efficientdet_brain_mri.ipynb`

## Запуск локальных режимов

```bash
python main.py --mode plan --model yolo
python main.py --mode analyze --model yolo
python main.py --mode prepare --model yolo
```
