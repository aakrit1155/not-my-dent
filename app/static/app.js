const form = document.querySelector('#claim-form');
const demoButton = document.querySelector('#demo-button');
const runButton = document.querySelector('#run-button');
const processing = document.querySelector('#processing');
const results = document.querySelector('#results');
const error = document.querySelector('#form-error');
const modeTag = document.querySelector('#mode-tag');
const photo = document.querySelector('#proof-image');
const modal = document.querySelector('#proof-modal');
let demoMode = false;
let currentProofUrl = null;

function setError(message) { error.textContent = message; error.hidden = !message; }
function displayFile(card) {
  const input = card.querySelector('input');
  const file = input.files[0];
  card.classList.toggle('selected', Boolean(file));
  card.querySelector('.file-name').textContent = file ? `${file.name} · ${(file.size / 1048576).toFixed(1)} MiB` : card.dataset.field === 'claim_image' ? 'JPG, PNG, WEBP · max 15 MiB' : 'MP4, MOV, WEBM, MKV · max 220 MiB';
  if (file) { demoMode = false; modeTag.textContent = '○  Uploaded files mode'; }
}
document.querySelectorAll('.upload-card').forEach(card => {
  const input = card.querySelector('input');
  input.addEventListener('change', () => displayFile(card));
  card.addEventListener('dragover', event => { event.preventDefault(); card.classList.add('dragover'); });
  card.addEventListener('dragleave', () => card.classList.remove('dragover'));
  card.addEventListener('drop', event => {
    event.preventDefault(); card.classList.remove('dragover');
    if (event.dataTransfer.files.length) { input.files = event.dataTransfer.files; displayFile(card); }
  });
});
demoButton.addEventListener('click', () => {
  demoMode = true;
  document.querySelectorAll('.upload-card').forEach(card => {
    card.querySelector('input').value = '';
    card.classList.add('selected');
  });
  const labels = ['Sample walkaround · before pickup', 'Sample walkaround · after drop-off', 'Sample vendor damage photo'];
  document.querySelectorAll('.upload-card .file-name').forEach((node, i) => { node.textContent = labels[i]; });
  document.querySelector('#claim-text').value = 'Deep dent and scratch on the front passenger bumper';
  modeTag.textContent = '✦  Sample inputs ready · generated on the server';
  setError('');
});

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function parseJson(response) {
  let data;
  try { data = await response.json(); } catch { throw new Error(`Server returned HTTP ${response.status}`); }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (HTTP ${response.status})`);
  return data;
}
async function pollJob(url) {
  for (let attempt = 0; attempt < 180; attempt++) {
    const job = await parseJson(await fetch(url, { cache: 'no-store' }));
    if (job.state === 'complete') return job.report;
    if (job.state === 'failed') throw new Error(job.error || 'Report generation failed');
    await sleep(1000);
  }
  throw new Error('Processing took too long. Please try again.');
}
function renderReport(report) {
  document.querySelector('#verdict').textContent = report.verdict;
  document.querySelector('#explanation').textContent = report.explanation;
  document.querySelector('#confidence').textContent = `${(report.confidence * 100).toFixed(1)}%`;
  document.querySelector('#claim-echo').textContent = report.claim_text;
  document.querySelector('#disclaimer').textContent = report.disclaimer;
  const panel = document.querySelector('.verdict-panel');
  panel.classList.toggle('is-valid', report.claim_valid);
  document.querySelector('#verdict-badge').textContent = report.claim_valid ? 'NEW DAMAGE · DEMO' : 'DISPUTE SUPPORTED · DEMO';
  currentProofUrl = report.proof_image_url;
  photo.src = currentProofUrl;
  results.hidden = false;
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
form.addEventListener('submit', async event => {
  event.preventDefault(); setError(''); results.hidden = true;
  const claim = document.querySelector('#claim-text').value.trim();
  if (!claim) return setError('Enter the vendor’s claim description.');
  if (!demoMode) {
    const fields = ['video_before', 'video_after', 'claim_image'];
    if (fields.some(field => !form.elements[field].files.length)) return setError('Choose all three files, or load the sample inputs.');
    if (fields.slice(0, 2).some(field => form.elements[field].files[0].size > 220 * 1048576)) return setError('Each video must be at most 220 MiB.');
    if (form.elements.claim_image.files[0].size > 15 * 1048576) return setError('The vendor photo must be at most 15 MiB.');
  }
  const payload = new FormData();
  payload.set('claim_text', claim);
  payload.set('demo', String(demoMode));
  if (!demoMode) ['video_before', 'video_after', 'claim_image'].forEach(field => payload.set(field, form.elements[field].files[0]));
  runButton.disabled = true; processing.hidden = false;
  document.querySelector('#progress-copy').textContent = demoMode ? 'Generating the sample comparison…' : 'Uploading files, then generating the mock comparison…';
  try {
    const created = await parseJson(await fetch('/api/verify-claim', { method: 'POST', body: payload }));
    document.querySelector('#progress-copy').textContent = 'Preparing your simulated report…';
    renderReport(await pollJob(created.status_url));
  } catch (err) { setError(err.message || 'Something went wrong.'); }
  finally { runButton.disabled = false; processing.hidden = true; }
});
document.querySelector('#proof-open').addEventListener('click', () => {
  if (!currentProofUrl) return;
  document.querySelector('#proof-large').src = currentProofUrl;
  modal.showModal();
});
document.querySelector('#proof-close').addEventListener('click', () => modal.close());
modal.addEventListener('click', event => { if (event.target === modal) modal.close(); });
