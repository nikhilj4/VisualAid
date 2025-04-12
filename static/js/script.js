// static/js/script.js
function uploadImage() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');

    if (!file) {
        alert('Please select an image!');
        return;
    }

    loading.style.display = 'block';
    results.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || `Server error: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        loading.style.display = 'none';
        if (data.error) {
            alert(data.error);
            return;
        }

        results.style.display = 'grid';
        document.getElementById('uploadedImage').src = data.image_url;
        document.getElementById('caption').textContent = data.caption;
        document.getElementById('ocrText').textContent = data.ocr_text;
        document.getElementById('explanation').textContent = data.explanation;
        document.getElementById('audioSource').src = data.audio_url;
        document.getElementById('audioPlayer').load();
        document.getElementById('audioPlayer').focus();
    })
    .catch(error => {
        loading.style.display = 'none';
        console.error('Error:', error);
        alert(`Failed to process image: ${error.message}. Please try a smaller image or check the server.`);
    });
}