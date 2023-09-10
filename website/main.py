from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import json
import psycopg2
from psycopg2 import sql
from psycopg2 import extras
from datetime import datetime
from tqdm import tqdm
import pickle
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, project_root)
from src.models.predict_model import inference_model

from src.models.train_model import ModelBuilder, FeatureSelector

app = Flask(__name__)

# Загрузка конфигурации из файла
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

db_params = config["db_params"]

# Подключение к базе данных
try:
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    print("Подключение к базе данных прошло успешно.")
except psycopg2.OperationalError as e:
    print("Ошибка подключения к базе данных:", e)

# Директория для загрузки файлов
UPLOAD_FOLDER = './flagged/data_file/'
model_path = '../models/model.pickle'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['model_path'] = model_path

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'json', 'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_and_write_in_table(file_path, table_name):
    # Подключение к базе данных
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # Чтение первой строки из CSV-файла, которая содержит заголовки колонок
    with open(file_path, 'r') as file:
        header = file.readline().strip()
    column_names = header.split(',')
    cleaned_column_names = [name.strip() for name in column_names]
    cleaned_column_names.remove('')

    # Создание таблицы (если она не существует) с названиями колонок из файла
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        {", ".join([f"{name} VARCHAR(255)" for name in cleaned_column_names if name])}
    )
    """

    try:
        cursor.execute(create_table_sql)
    except Exception as e:
        print("Table already exist. Start write in table")
        print(str(e))
    conn.commit()

    df = pd.read_csv(file_path)

    # Создание SQL-запроса для вставки данных в PostgreSQL
    columns_str = ", ".join(cleaned_column_names)
    placeholders = ", ".join(["%s"] * len(cleaned_column_names))
    insert_sql = sql.SQL(f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})")

    # Вставка данных в таблицу
    for index, row in df.iterrows():
        values = [row[column] for column in cleaned_column_names]
        cursor.execute(insert_sql, values)
    conn.commit()

    # Закрытие соединения
    conn.close()


@app.route('/')
def index():
    return render_template('index.html')


# Обработчик маршрута для страницы загрузки файла
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                render_template('No file part')

        file = request.files['file']

        if file.filename == '':
            raise Exception('No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(filepath)

            # Проверяем, что файл был передан в запросе
            if 'file' not in request.files:
                pass

            file = request.files['file']

            if file.filename == '':
                pass

            # Сохраняем содержимое файла на сервере
            file.save(filepath)

            print("Считываю модель")

            with open(app.config['model_path'], 'rb') as file:
                models = pickle.load(file)

            print("Загружаю датасет")
            feature_dataset = pd.read_csv(filepath, delimiter=';')
            print(feature_dataset.shape)
            try:
                print("Формирую предсказания модели")
                inference_df = inference_model(input_dataset=feature_dataset, models=models)
                today = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
                inference_df['load_dt'] = today
                today = today.replace(' ', '_').replace('-', '_').replace(':',
                                                             '_')
                inference_file_path = f'{app.config["UPLOAD_FOLDER"]}inference_{today}.csv'
                inference_df.to_csv(inference_file_path)
                create_and_write_in_table(inference_file_path, table_name='result_table')
                print("Результат успешно записан в таблицу!")
                return redirect(url_for('index'))

            except Exception as e:
                print(str(e))
                return redirect(url_for('index'))

        else:
            raise Exception('Invalid file format')

    except Exception as e:
        error_message = "Произошла ошибка при загрузке данных. Проверьте загружаемый файл."
        return render_template('index.html', error_message=error_message)


if __name__ == '__main__':
    app.secret_key = 'supersecretkey'  # Замените на свой секретный ключ
    app.run(debug=True)
