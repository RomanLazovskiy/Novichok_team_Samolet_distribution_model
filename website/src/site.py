from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Обработчик для главной страницы
@app.route('/')
def index():
    return render_template('index.html')

# ... ваш код ...

if __name__ == '__main__':
    app.run(debug=True)
