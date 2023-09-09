from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import json
import psycopg2
from psycopg2 import sql
import os

app = Flask(__name__)

# Загрузка конфигурации из файла
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

db_params = config["db_params"]

conn = psycopg2.connect(**db_params)

# Директория для загрузки файлов
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'json', 'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_table():
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # SQL-запрос для создания таблицы, если она еще не существует
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS files (
        id SERIAL PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        file_data BYTEA NOT NULL
    )
    """

    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
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

        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Чтение данных из файла и вставка их в базу данных
        with open(filepath, 'rb') as f:
            file_data = f.read()

        insert_sql = sql.SQL("""
        INSERT INTO files (filename, file_data) VALUES (%s, %s)
        """)

        cursor.execute(insert_sql, [filename, psycopg2.Binary(file_data)])
        conn.commit()
        conn.close()

        flash('File uploaded successfully')
        return redirect(url_for('index'))

    else:
        flash('Invalid file format')
        return redirect(request.url)


if __name__ == '__main__':
    create_table()
    app.secret_key = 'supersecretkey'  # Замените на свой секретный ключ
    app.run(debug=True)