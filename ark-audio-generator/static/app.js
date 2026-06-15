/* ═══════════════════════════════════════════════════════════════
   AI Sing-Along Generator  –  Frontend logic
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ──────────────────────────────────────────────────────────────────────────────
// Sample melodies – covering global & ethnic diversity
// ──────────────────────────────────────────────────────────────────────────────

const SAMPLES = [
  {
    flag: '🇮🇳', region: 'India',
    title: 'Bollywood Festive',
    genre: 'pop', mood: 'happy', instruments: 'sitar,tabla,harmonium',
    desc: 'A lively Bollywood-style tune with tabla beats, sitar ornamentation and a joyful melody perfect for a festive sing-along.',
    melody: 'A cheerful Bollywood melody with sitar-style ornaments, rising chorus, and a joyful festive feel rooted in North Indian classical music',
  },
  {
    flag: '🌍', region: 'West Africa',
    title: 'Highlife Groove',
    genre: 'folk', mood: 'energetic', instruments: 'kora,talking drums,guitar',
    desc: 'West African highlife rhythm with kora, talking drums and an irresistible call-and-response groove.',
    melody: 'A lively West African highlife melody with kora-style arpeggios, talking drum rhythm, and a bright call-and-response vocal hook',
  },
  {
    flag: '🇯🇵', region: 'Japan',
    title: 'Pentatonic Serenade',
    genre: 'ambient', mood: 'calm', instruments: 'shakuhachi,koto,taiko',
    desc: 'A serene Japanese melody on pentatonic scale with shakuhachi flute, koto plucks and gentle taiko accents.',
    melody: 'A peaceful Japanese melody in the pentatonic scale with shakuhachi-style breathy flute, koto arpeggios, and soft taiko drum accents',
  },
  {
    flag: '🇧🇷', region: 'Brazil',
    title: 'Bossa Nova Breeze',
    genre: 'bossa-nova', mood: 'romantic', instruments: 'guitar,bass,percussion',
    desc: 'A gentle bossa nova groove with nylon guitar, soft syncopated bass and a warm, romantic mood.',
    melody: 'A warm bossa nova melody with nylon guitar chord comping, syncopated bass, light shaker groove and a tender romantic feel',
  },
  {
    flag: '🇮🇪', region: 'Celtic / Irish',
    title: 'Reel & Jig',
    genre: 'folk', mood: 'energetic', instruments: 'fiddle,tin whistle,bodhrán',
    desc: 'A rousing Irish jig with fiddle and tin whistle over a steady bodhrán beat — get on your feet!',
    melody: 'A lively Irish jig melody with fiddle runs, tin whistle descant, steady bodhrán beat and an infectious dance-floor energy',
  },
  {
    flag: '🇸🇦', region: 'Middle East',
    title: 'Arabian Night',
    genre: 'ambient', mood: 'mysterious', instruments: 'oud,darbuka,strings',
    desc: 'A flowing Arabic melody over oud and darbuka rhythms with rich ornaments and evocative minor mode.',
    melody: 'A flowing Arabic melody with oud slides and ornaments, darbuka rhythm, rich string pads and a mysterious minor-scale character',
  },
  {
    flag: '🇰🇷', region: 'Korea',
    title: 'K-Pop Anthem',
    genre: 'pop', mood: 'uplifting', instruments: 'synthesizer,drums,bass',
    desc: 'An infectious K-Pop chorus with bright synths, punchy beat and an uplifting singalong hook.',
    melody: 'A punchy K-Pop chorus melody with bright synth pads, powerful snare-heavy beat, catchy hook and an uplifting triumphant feel',
  },
  {
    flag: '🇺🇸', region: 'Gospel / Soul',
    title: 'Sunday Soul',
    genre: 'r-and-b', mood: 'uplifting', instruments: 'organ,choir,drums',
    desc: 'A soulful gospel anthem with Hammond organ, full choir harmonies and a soaring, uplifting melody.',
    melody: 'A soulful gospel anthem with Hammond organ chords, rich choir harmonies, steady gospel beat and a powerful uplifting melody',
  },
  {
    flag: '🇪🇸', region: 'Flamenco / Spain',
    title: 'Flamenco Pasión',
    genre: 'folk', mood: 'aggressive', instruments: 'flamenco guitar,cajon,palmas',
    desc: 'Passionate flamenco with intense guitar rasgueado, cajón beats and clapping palmas.',
    melody: 'An intense flamenco melody with fast guitar rasgueado, cajón accents, clapping palmas and passionate Phrygian scale character',
  },
  {
    flag: '🇲🇽', region: 'Mexico / Latin',
    title: 'Mariachi Fiesta',
    genre: 'folk', mood: 'happy', instruments: 'trumpet,violin,guitarrón',
    desc: 'A festive mariachi sound with bright trumpets, violin melody and driving guitarrón bass.',
    melody: 'A bright mariachi melody with lead trumpet, violin harmonies, guitarrón bass and a joyful celebratory feel in major key',
  },
  {
    flag: '🇷🇺', region: 'Slavic Folk',
    title: 'Balaika Dance',
    genre: 'folk', mood: 'energetic', instruments: 'balalaika,accordion,percussion',
    desc: 'A spinning Slavic folk dance with balalaika tremolo, accordion chords and stomping percussion.',
    melody: 'A lively Slavic folk dance melody with balalaika tremolo, accordion bass-chord pattern and energetic stomping rhythm',
  },
  {
    flag: '🌊', region: 'Pacific / Hawaiian',
    title: 'Aloha Spirit',
    genre: 'folk', mood: 'calm', instruments: 'ukulele,guitar,bass',
    desc: 'A breezy Hawaiian melody with ukulele strumming, gentle steel guitar and a warm Aloha feeling.',
    melody: 'A breezy Hawaiian melody with ukulele strumming, gentle steel guitar slides, soft bass and a warm relaxed island atmosphere',
  },
];

// ──────────────────────────────────────────────────────────────────────────────
// Generation steps (used for the progress tracker)
// ──────────────────────────────────────────────────────────────────────────────

const STEPS = [
  { label: 'Request received',   minProgress: 0  },
  { label: 'Loading AI model',   minProgress: 10 },
  { label: 'Generating music',   minProgress: 30 },
  { label: 'Applying effects',   minProgress: 80 },
  { label: 'Exporting MP3',      minProgress: 92 },
];

// ──────────────────────────────────────────────────────────────────────────────
// State
// ──────────────────────────────────────────────────────────────────────────────

let currentJobId   = null;
let pollTimer      = null;
let activeCardIdx  = null;

// ──────────────────────────────────────────────────────────────────────────────
// DOM helpers
// ──────────────────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

function show(id)  { $(id).classList.remove('hidden'); }
function hide(id)  { $(id).classList.add('hidden');    }
function section(which) {
  ['progressSection', 'resultSection', 'errorSection'].forEach(s => hide(s));
  if (which) show(which);
}

// ──────────────────────────────────────────────────────────────────────────────
// Build sample cards
// ──────────────────────────────────────────────────────────────────────────────

function buildSamples() {
  const grid = $('samplesGrid');
  SAMPLES.forEach((s, i) => {
    const card = document.createElement('div');
    card.className = 'sample-card';
    card.innerHTML = `
      <span class="sample-flag">${s.flag}</span>
      <span class="sample-region">${s.region}</span>
      <div class="sample-title">${s.title}</div>
      <div class="sample-desc">${s.desc}</div>
    `;
    card.addEventListener('click', () => fillSample(s, i, card));
    grid.appendChild(card);
  });
}

function fillSample(s, idx, card) {
  // Deactivate previous
  document.querySelectorAll('.sample-card.active')
    .forEach(c => c.classList.remove('active'));
  card.classList.add('active');
  activeCardIdx = idx;

  $('melodyInput').value = s.melody;
  updateCharCount(s.melody.length);
  if (s.genre) $('genre').value = s.genre;
  if (s.mood)  $('mood').value  = s.mood;
  if (s.instruments) $('instruments').value = s.instruments;

  $('melodyInput').focus();
  $('melodyInput').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ──────────────────────────────────────────────────────────────────────────────
// Live range + char-count updates
// ──────────────────────────────────────────────────────────────────────────────

function updateCharCount(n) {
  $('charCount').textContent = n;
}

function initControls() {
  $('melodyInput').addEventListener('input', e => updateCharCount(e.target.value.length));

  $('duration').addEventListener('input', e => {
    $('durationVal').textContent = e.target.value + 's';
  });
  $('guidance').addEventListener('input', e => {
    $('guidanceVal').textContent = parseFloat(e.target.value).toFixed(1);
  });
  $('temperature').addEventListener('input', e => {
    $('tempVal').textContent = parseFloat(e.target.value).toFixed(2);
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Form submission
// ──────────────────────────────────────────────────────────────────────────────

$('genForm').addEventListener('submit', async e => {
  e.preventDefault();
  const melody = $('melodyInput').value.trim();
  if (!melody) {
    $('melodyInput').focus();
    return;
  }

  const payload = {
    melody,
    genre:           $('genre').value     || null,
    mood:            $('mood').value      || null,
    instruments:     $('instruments').value.trim() || null,
    frequency_range: $('frequency').value || null,
    duration:        parseFloat($('duration').value),
    crescendo:       $('crescendo').value,
    guidance_scale:  parseFloat($('guidance').value),
    temperature:     parseFloat($('temperature').value),
  };

  // Disable form
  const btn = $('generateBtn');
  btn.disabled = true;
  $('btnText').textContent = 'Generating…';

  section('progressSection');
  initProgressSteps();
  updateProgress({ status: 'pending', message: 'Queued…', progress: 0 });
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });

  try {
    const res  = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const { job_id } = await res.json();
    currentJobId = job_id;
    startPolling(job_id, payload);
  } catch (err) {
    showError(err.message);
    resetButton();
  }
});

// ──────────────────────────────────────────────────────────────────────────────
// Progress steps
// ──────────────────────────────────────────────────────────────────────────────

function initProgressSteps() {
  const list = $('stepsList');
  list.innerHTML = '';
  STEPS.forEach((s, i) => {
    const div = document.createElement('div');
    div.className = 'step';
    div.id = `step-${i}`;
    div.innerHTML = `<div class="step-dot"></div><span>${s.label}</span>`;
    list.appendChild(div);
  });
}

function updateProgressSteps(progress) {
  STEPS.forEach((s, i) => {
    const el = $(`step-${i}`);
    const nextMin = STEPS[i + 1]?.minProgress ?? 101;
    if (progress >= nextMin) {
      el.className = 'step done';
      el.querySelector('.step-dot').textContent = '✓';
    } else if (progress >= s.minProgress) {
      el.className = 'step active';
      el.querySelector('.step-dot').textContent = '';
    } else {
      el.className = 'step';
      el.querySelector('.step-dot').textContent = '';
    }
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Polling
// ──────────────────────────────────────────────────────────────────────────────

function startPolling(jobId, payload) {
  clearInterval(pollTimer);
  pollTimer = setInterval(() => pollStatus(jobId, payload), 3000);
}

async function pollStatus(jobId, payload) {
  try {
    const res  = await fetch(`/api/status/${jobId}`);
    if (!res.ok) return;
    const data = await res.json();

    updateProgress(data);

    if (data.status === 'done') {
      clearInterval(pollTimer);
      showResult(jobId, data, payload);
    } else if (data.status === 'error') {
      clearInterval(pollTimer);
      showError(data.message);
      resetButton();
    }
  } catch (_) {
    // Network blip – keep polling
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// UI state handlers
// ──────────────────────────────────────────────────────────────────────────────

function updateProgress(data) {
  const pct = data.progress ?? 0;
  $('progressBar').style.width = pct + '%';
  $('statusMessage').textContent = data.message || '';
  updateProgressSteps(pct);
}

function showResult(jobId, data, payload) {
  section('resultSection');

  // Set audio source
  const player = $('audioPlayer');
  player.src = `/api/download/${jobId}`;

  // Set download link
  const dlBtn = $('downloadBtn');
  dlBtn.href = `/api/download/${jobId}`;

  // Meta badges
  const meta = $('trackMeta');
  meta.innerHTML = '';
  const badges = [
    data.genre   ? `🎸 ${data.genre}` : null,
    data.mood    ? `✨ ${data.mood}`   : null,
    data.duration ? `⏱ ${data.duration}s` : null,
    payload?.instruments ? `🎹 ${payload.instruments}` : null,
  ].filter(Boolean);
  badges.forEach(b => {
    const span = document.createElement('span');
    span.className = 'meta-badge';
    span.textContent = b;
    meta.appendChild(span);
  });

  resetButton();
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function showError(msg) {
  section('errorSection');
  $('errorMessage').textContent = msg || 'An unexpected error occurred. Please try again.';
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function resetButton() {
  const btn = $('generateBtn');
  btn.disabled = false;
  $('btnText').textContent = 'Generate Track';
}

// ──────────────────────────────────────────────────────────────────────────────
// "Make another" and "Retry" buttons
// ──────────────────────────────────────────────────────────────────────────────

$('newTrackBtn').addEventListener('click', () => {
  // Clean up job on server
  if (currentJobId) {
    fetch(`/api/job/${currentJobId}`, { method: 'DELETE' }).catch(() => {});
    currentJobId = null;
  }
  section(null);
  $('audioPlayer').src = '';
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

$('retryBtn').addEventListener('click', () => {
  section(null);
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ──────────────────────────────────────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────────────────────────────────────

buildSamples();
initControls();
