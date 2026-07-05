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
  artifacts/
  configs/
  data/
  notebooks/
  results/
  scripts/
  src/
  main.py
  requirements.txt
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
