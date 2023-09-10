document.addEventListener("DOMContentLoaded", function () {
    const progressBar = document.getElementById("upload-progress");
    const progressText = document.getElementById("progress-text");
    const form = document.querySelector('form');

    form.addEventListener('submit', function (e) {
        e.preventDefault();

        const formData = new FormData();
        formData.append('file', document.querySelector('#file-input').files[0]);

        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', function (e) {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                progressBar.value = percentComplete;
                progressText.textContent = `Прогресс: ${percentComplete.toFixed(2)}%`;
            }
        });

        xhr.onload = function () {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                if (data.error) {
                    console.error(data.error);
                } else {
                    console.log(data.success);
                    // Здесь можно добавить код для отображения результата на странице
                }
            } else {
                console.error('Ошибка:', xhr.status, xhr.statusText);
            }
        };

        xhr.onerror = function () {
            console.error('Произошла ошибка при отправке запроса.');
        };

        xhr.open('POST', '/upload', true);
        xhr.send(formData);
    });
});
