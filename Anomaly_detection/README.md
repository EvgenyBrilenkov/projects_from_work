# Anomaly Detection 

Anomaly Detection  - это задача детекции аномалий на фотографиях.  
В данном проекте оно реализовано через обучение без учителя, используя алгоритмы библиотеки anomalib.  
На вход при обучении подаются нормальные (эталонные) изображения, алгоритмы запоминают их признаки.  
Во время теста подаются изображения с аномалиями. Алгоритмы, основываясь на признаках эталонных изображений, должны выявить расхождения нормальных участков и аномальных.

Использованные алгоритмы:

- PaDiM
- PatchCore
- CFlow

## Архитектура репозитория  

Anomaly_detection  
|  
|___training_code (директория с обучением моделей и подбором параметров)  
|  
|___inference_code (директория с кодом для инференса/теста моделей)  
|  
|___images (изображения для README)  
|  
|___env.yml (файл с окружением)  
|  
|___models_comparing.xlsx (таблица с метриками моделей)  

## Устройство кода

Важно!
PaDiM и PatchCore учатся в 1 эпоху.
CFlow - в указанное количество эпох.

Для ноутбуков с обучением каждая ячейка - отдельное обучение модели на новых параметрах.
При этом для многих моделей были опробованы разные сверточные сети для извлечения признаков, веса которых лежат в директории /weights.

Использованные feature-extractors/backbones:

- ResNet-18
- ResNet-50
- Wide ResNet-50-2
- EfficientNet-B7
- CSPDarknet53

В каждой ячейке ноутбука:
```
# Загрузка данных
train_dataset = Folder(
    name="custom_folder", 
    root="/wrk/data/processed", # Путь до директории с данными
    normal_dir="normal_dir",  # Директория с эталонными данными
    abnormal_dir="abnormal_dir", # Директория с аномальными данными
    mask_dir="mask_dir", # Директория с масками (для метрик по локализации аномалий)
    train_batch_size=10,
    eval_batch_size=10,
    num_workers=2,
    seed=42    
)

# Загрузка feature-extractor
weights = torch.load("/wrk/weights/resnet18.pth", map_location='cuda') # Загрузка весов нужного backbone

# Выбранная модель и ее параметры
model = Patchcore(
    backbone=custom_backbone,
    layers=["layer2", "layer3"],
    pre_trained=False,
    coreset_sampling_ratio=0.01,
    num_neighbors=3
)

# В коде инференса:
predictions = engine.predict(model=model,
                             datamodule=test_dataset,
                             ckpt_path="/wrk/notebooks/results/Patchcore/custom_folder/fit_50_2layers_0.1_5/weights/lightning/model.ckpt") # Путь до обученной модели
```

## Наилучшие модели (эксперименты еще продолжаются)

На данный момент самые лучшие модели:  

PaDiM:  
- backbone: Wide ResNet-50-2
- n_features: 500
- layers: ["layer1", "layer2", "layer3"]

PatchCore:  
- backbone: ResNet-50
- layers: ["layer2", "layer3"]
- coreset_sampling_ratio: 0.1
- num_neighbors: 5

| Model     | image_AUROC | image_F1Score | pixel_AUROC | pixel_F1Score |
|-----------|-------------|---------------|-------------|---------------|
| PaDiM     | 0.711       | 0.922         | 0.952       | 0.791         |
| PatchCore | 0.886       | 0.932         | 0.933       | 0.744         |


## Примеры работы моделей

![sample_1](/Anomaly_detection/images/sample_1.jpg)   

![sample_2](/Anomaly_detection/images/sample_2.jpg)
