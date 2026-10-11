const form = document.querySelector('#claim-form');
const runButton = document.querySelector('#run-button');
const processing = document.querySelector('#processing');
const progress = document.querySelector('#progress-copy');
const results = document.querySelector('#results');
const error = document.querySelector('#form-error');
const modeTag = document.querySelector('#mode-tag');
const modal = document.querySelector('#proof-modal');
const FIELDS = ['video_before', 'video_after', 'claim_image'];
const VIDEO_HINT = 'MP4, MOV, WEBM, MKV, AVI · max 220 MiB';

// Wording mirrors the notebook's final assessment.
const VERDICTS = {
  'PRE-EXISTING DAMAGE': { badge: 'DISPUTE SUPPORTED', tone: 'pre-existing', text: 'The matching claimed damage was visible before the rental and after return. The evidence supports disputing the charge.' },
  'POSSIBLE NEW DAMAGE': { badge: 'CLAIM SUPPORTED', tone: 'new', text: 'The claimed damage matches the AFTER evidence but was not identifiable in the adequately inspected BEFORE location. This is consistent with possible new damage.' },
  'INCONCLUSIVE': { badge: 'INCONCLUSIVE', tone: 'unclear', text: 'The available evidence does not establish a reliable conclusion for the complete claim.' },
};

function setError(message) { error.textContent = message; error.hidden = !message; }
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}
function displayFile(card) {
  const file = card.querySelector('input').files[0];
  card.classList.toggle('selected', Boolean(file));
  card.querySelector('.file-name').textContent = file ? `${file.name} · ${(file.size / 1048576).toFixed(1)} MiB` : card.dataset.field === 'claim_image' ? 'JPG, PNG, WEBP · max 15 MiB' : VIDEO_HINT;
  const count = FIELDS.filter(field => form.elements[field].files.length).length;
  modeTag.textContent = count === 3 ? '✓  All files ready' : `○  ${count} of 3 files added`;
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

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function parseJson(response) {
  let data;
  try { data = await response.json(); } catch { throw new Error(`Server returned HTTP ${response.status}`); }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (HTTP ${response.status})`);
  return data;
}
async function pollJob(url) {
  const started = Date.now();
  while (Date.now() - started < 30 * 60 * 1000) {
    const job = await parseJson(await fetch(url, { cache: 'no-store' }));
    if (job.state === 'complete') return job.report;
    if (job.state === 'failed') throw new Error(job.error || 'Analysis failed');
    const minutes = Math.floor((Date.now() - started) / 60000);
    progress.textContent = job.models === 'loading'
      ? 'Models are still loading on the server — your job will start right after…'
      : `${job.state === 'queued' ? 'Waiting in queue' : 'Comparing frames with the claim'} · ${minutes} min elapsed. This usually takes a few minutes.`;
    await sleep(2000);
  }
  throw new Error('Processing took too long. Please try again.');
}

function openImage(src) { document.querySelector('#proof-large').src = src; modal.showModal(); }
function imageButton(src, alt, label) {
  const button = el('button', 'proof-button');
  button.type = 'button';
  const img = el('img'); img.src = src; img.alt = alt; img.loading = 'lazy';
  button.append(img, el('span', '', label || '⤢  Enlarge'));
  button.addEventListener('click', () => openImage(src));
  return button;
}
function score(result) { return result ? `${result.type} ${result.confidence.toFixed(3)}` : 'not detected'; }

function evidenceCard(title, side, showStatus) {
  const card = el('div', 'evidence-card');
  const head = el('div', 'evidence-head');
  head.append(el('span', 'kicker', title));
  if (side.timestamp_seconds !== undefined) head.append(el('span', 'evidence-time', side.timestamp_seconds === null ? '—' : `${side.timestamp_seconds.toFixed(2)} s`));
  card.append(head);
  if (side.images) {
    const frame = imageButton(side.images.crop, `${title} zoomed region`, '⤢  Zoom');
    card.append(frame);
    const full = el('button', 'text-link', 'View full frame');
    full.type = 'button';
    full.addEventListener('click', () => openImage(side.images.frame));
    card.append(full);
  } else {
    card.append(el('div', 'evidence-empty', 'Matching location not found'));
  }
  const facts = el('dl', 'evidence-facts');
  const add = (k, v) => { facts.append(el('dt', '', k), el('dd', '', v)); };
  if (showStatus) add('Status', side.status);
  if (side.confidence) { add('Detector (full)', score(side.confidence.full_frame)); add('Detector (zoom)', score(side.confidence.zoom)); }
  card.append(facts, el('p', 'evidence-desc', side.description));
  return card;
}

function renderReport(report) {
  const overall = VERDICTS[report.overall_verdict] || VERDICTS.INCONCLUSIVE;
  document.querySelector('#verdict').textContent = report.overall_verdict;
  document.querySelector('#explanation').textContent = report.message || overall.text;
  document.querySelector('#verdict-badge').textContent = overall.badge;
  document.querySelector('.verdict-panel').dataset.tone = overall.tone;
  document.querySelector('#area-count').textContent = String(report.findings.length);
  document.querySelector('#claim-echo').textContent = report.claim;

  const container = document.querySelector('#findings');
  container.replaceChildren();
  report.findings.forEach((finding, i) => {
    const verdict = VERDICTS[finding.verdict] || VERDICTS.INCONCLUSIVE;
    const article = el('article', 'finding');
    const head = el('div', 'proof-head');
    const title = el('div');
    title.append(el('h3', '', `Claimed area ${i + 1}`), el('p', '', finding.comparison));
    const badge = el('span', 'finding-badge', finding.verdict);
    badge.dataset.tone = verdict.tone;
    head.append(title, badge);
    const grid = el('div', 'evidence-grid');
    grid.append(
      evidenceCard('COMPANY CLAIM', finding.company, false),
      evidenceCard('BEFORE RENTAL', finding.before, true),
      evidenceCard('AFTER RETURN', finding.after, true),
    );
    article.append(head, grid, imageButton(finding.figure_path, `Evidence comparison for claimed area ${i + 1}`, '⤢  Enlarge comparison'));
    container.append(article);
  });
  results.hidden = false;
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

form.addEventListener('submit', async event => {
  event.preventDefault(); setError(''); results.hidden = true;
  const claim = document.querySelector('#claim-text').value.trim();
  if (!claim) return setError('Enter the vendor’s claim description.');
  if (FIELDS.some(field => !form.elements[field].files.length)) return setError('Choose both videos and the vendor photo.');
  if (FIELDS.slice(0, 2).some(field => form.elements[field].files[0].size > 220 * 1048576)) return setError('Each video must be at most 220 MiB.');
  if (form.elements.claim_image.files[0].size > 15 * 1048576) return setError('The vendor photo must be at most 15 MiB.');
  const payload = new FormData();
  payload.set('claim_text', claim);
  FIELDS.forEach(field => payload.set(field, form.elements[field].files[0]));
  runButton.disabled = true; processing.hidden = false;
  progress.textContent = 'Uploading files…';
  try {
    const created = await parseJson(await fetch('/api/verify-claim', { method: 'POST', body: payload }));
    progress.textContent = 'Upload complete. Starting analysis…';
    renderReport(await pollJob(created.status_url));
  } catch (err) { setError(err.message || 'Something went wrong.'); }
  finally { runButton.disabled = false; processing.hidden = true; }
});
document.querySelector('#proof-close').addEventListener('click', () => modal.close());
modal.addEventListener('click', event => { if (event.target === modal) modal.close(); });
