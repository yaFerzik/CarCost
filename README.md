<div align="center">
<h1>
  CarCost
</h1>
</div>

### О проекте:
<div>
  Небольшой проект для изучения классического ml. Предсказание цен автомобилей
  
</div>

---

### Технологии:
<div>
  Python, pandas, numpy, scikit-learn, FastAPI, HTML, CSS, JavaScript
</div>

---

### Ссылка на датасет:

  https://www.kaggle.com/datasets/andreinovikov/used-cars-dataset/data


---

### Цель проекта:

1. Создание модели для предсказания цен автомобилей
2. Создание модели для классификации автомобилей
3. Создание api для обертки модели
4. Создание Клиента для работы с api

---

### Модель:

  DecisionTreeRegressor


---

### Метрики:
1. MAE
2. RMSE
3. R²

---

### Результаты:
1. Проведен разведывательный анализ данных
2. Введены изменения, способствующие лучшей корреляции
3. Реализован собственный K-fold target encoder
4. Подобрана лучшая по метрикам модель
5. Реализована инфраструктура для дообучения модели
6. Создан api, с возможность предсказания цены для отправляемого необработанного объекта
7. Реализован простой и минималистичный (временный) веб-клиент

---

### В процессе

1. Реализация дополнительных эндпоинтов
2. Возможное добавление базы данных и инфраструктуры учета некоторых автомобилей
3. Реализация настольного клиента
---

### Структура проекта:

```text
Car_Cost_project/
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── artifacts/
│   └── static/
├── ml/
│   ├── preprocessing.py
│   └── train_and_save_model.py
├── notebooks/
├── data/
│   └── cars.csv
└── README.md
```

---

### Запуск проекта:

Создание виртуального окружения:

```powershell
python -m venv .venv
```

Активация в Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Установка зависимостей:

```powershell
pip install -r api\requirements.txt
```

Обучение и сохранение модели:

```powershell
python -m ml.train_and_save_model
```

Запуск API:

```powershell
python -m uvicorn api.main:app --reload
```

После запуска веб-клиент доступен по адресу:

```text
http://127.0.0.1:8000/
```

Документация Swagger:

```text
http://127.0.0.1:8000/docs
```

Проверка состояния API:

```text
http://127.0.0.1:8000/health
```

