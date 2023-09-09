document.addEventListener("DOMContentLoaded", function() {
    document.querySelector('form').addEventListener('submit', function (e) {
        e.preventDefault();

        const formData = new FormData();
        formData.append('file', document.querySelector('#file-input').files[0]);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error(data.error);
            } else {
                console.log(data.success);
                // Здесь можно добавить код для отображения результата на странице
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});
