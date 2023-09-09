from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import json
import psycopg2
from psycopg2 import sql
from psycopg2 import extras
import os

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
    # Далее можно выполнять SQL-запросы
    # Не забудьте закрыть соединение и курсор после использования
except psycopg2.OperationalError as e:
    print("Ошибка подключения к базе данных:", e)

# Директория для загрузки файлов
UPLOAD_FOLDER = './flagged/data_file/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

    # Создание таблицы (если она не существует) с названиями колонок из файла
    table_name = table_name
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        {", ".join([f"{name} VARCHAR(255)" for name in column_names])}
    )
    """
    try:
        cursor.execute(create_table_sql)
    except:
        print("Table already exist. Start write in table")
    conn.commit()

    # Загрузка данных из CSV-файла с помощью Pandas
    df = pd.read_csv(file_path)

    # Вставка данных в таблицу с использованием пакетной вставки
    data = [tuple(row[name] for name in column_names) for _, row in df.iterrows()]
    insert_sql = sql.SQL(f"""
    INSERT INTO {table_name} ({", ".join(column_names)}) VALUES ({", ".join(["%s"] * len(column_names))})
    """)
    extras.execute_batch(cursor, insert_sql, data)
    conn.commit()

    # Закрытие соединения
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')


# Обработчик маршрута для страницы загрузки файла
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Добавьте код для записи данных в базу данных
            create_and_write_in_table(filepath, table_name='raw_dataset')
            flash('File uploaded successfully')
            return redirect(url_for('index'))

        else:
            flash('Invalid file format')
            return redirect(request.url)

    # Если это GET-запрос, просто отобразите форму загрузки файла
    return render_template('index.html')

# ... (остальной код)

if __name__ == '__main__':
    app.secret_key = 'supersecretkey'  # Замените на свой секретный ключ
    app.run(debug=True)



if __name__ == '__main__':
    app.secret_key = 'supersecretkey'  # Замените на свой секретный ключ
    app.run(debug=True)
