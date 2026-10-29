const input = document.querySelector('#file-input');
const zone = document.querySelector('#drop-zone');
const browse = document.querySelector('#browse-button');
const preview = document.querySelector('#preview-wrap');
const previewImg = document.querySelector('#image-preview');
const prompt = document.querySelector('#upload-prompt');
const classify = document.querySelector('#classify-button');
const errorBox = document.querySelector('#error-message');
let selectedFile = null;
let previewUrl = null;

async function loadHealth() {
  const status = document.querySelector('#api-status');
  try {
    const response = await fetch('/api/health');
    const health = await response.json();
    status.textContent = health.model_loaded ? `Model online · ${health.device}` : 'Checkpoint needed';
    document.querySelector('.live-indicator').classList.toggle('ready', health.model_loaded);
    document.querySelector('#inference-device').textContent = health.model_loaded ? `PyTorch · ${health.device}` : 'PyTorch · checkpoint needed';
    for (const id of ['#hero-accuracy', '#metric-accuracy']) document.querySelector(id).innerHTML = `${health.accuracy.toFixed(2)}<span>%</span>`;
  } catch {
    status.textContent = 'API unavailable';
    document.querySelector('#inference-device').textContent = 'Start the API to enable inference';
  }
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
  document.querySelector('#result-badge').className = 'result-badge error';
  document.querySelector('#result-badge').innerHTML = '<i></i> Needs attention';
}

function resetResult() {
  errorBox.hidden = true;
  document.querySelector('#result-badge').className = 'result-badge';
  document.querySelector('#result-badge').innerHTML = '<i></i> Awaiting image';
  document.querySelector('#result-output').hidden = true;
  document.querySelector('#result-content').hidden = false;
}

function setFile(file) {
  if (!file) return;
  if (!['image/png', 'image/jpeg', 'image/webp', 'image/bmp'].includes(file.type)) return showError('Choose a PNG, JPG, WEBP, or BMP image.');
  if (file.size > 8 * 1024 * 1024) return showError('This image is over 8 MB. Choose a smaller file.');
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImg.src = previewUrl;
  document.querySelector('#preview-name').textContent = file.name;
  preview.hidden = false;
  prompt.hidden = true;
  classify.disabled = false;
  errorBox.hidden = true;
  resetResult();
}

browse.addEventListener('click', event => { event.stopPropagation(); input.click(); });
zone.addEventListener('click', event => { if (!event.target.closest('button')) input.click(); });
zone.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); input.click(); } });
input.addEventListener('change', () => { setFile(input.files[0]); input.value = ''; });
for (const eventName of ['dragenter', 'dragover']) zone.addEventListener(eventName, event => { event.preventDefault(); zone.classList.add('dragover'); });
for (const eventName of ['dragleave', 'drop']) zone.addEventListener(eventName, event => { event.preventDefault(); zone.classList.remove('dragover'); });
zone.addEventListener('drop', event => setFile(event.dataTransfer.files[0]));
document.querySelector('#remove-image').addEventListener('click', event => {
  event.stopPropagation();
  selectedFile = null;
  preview.hidden = true;
  prompt.hidden = false;
  classify.disabled = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  resetResult();
});

classify.addEventListener('click', async () => {
  if (!selectedFile) return;
  const original = classify.innerHTML;
  classify.disabled = true;
  classify.innerHTML = '<span class="spinner"></span> Reading the look…';
  errorBox.hidden = true;
  const start = performance.now();
  const body = new FormData();
  body.append('file', selectedFile);
  try {
    const response = await fetch('/api/predict', { method: 'POST', body });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Could not classify this image.');
    document.querySelector('#result-content').hidden = true;
    document.querySelector('#result-output').hidden = false;
    document.querySelector('#prediction-image').src = previewUrl;
    document.querySelector('#prediction-name').textContent = result.prediction;
    document.querySelector('#prediction-confidence').textContent = `${result.confidence.toFixed(1)}% confidence`;
    document.querySelector('#confidence-bars').innerHTML = result.top_predictions.map(item => `<div class="confidence-row"><span>${item.label}</span><span class="confidence-track"><i style="width:${Math.max(2, item.confidence)}%"></i></span><span class="confidence-value">${item.confidence.toFixed(1)}%</span></div>`).join('');
    document.querySelector('#result-badge').className = 'result-badge ready';
    document.querySelector('#result-badge').innerHTML = '<i></i> Prediction ready';
    document.querySelector('#inference-device').textContent = `PyTorch · ${result.device}`;
    document.querySelector('#latency-label').textContent = `${Math.round(performance.now() - start)} ms`;
  } catch (error) {
    showError(error.message);
  } finally {
    classify.disabled = false;
    classify.innerHTML = original;
  }
});

loadHealth();
