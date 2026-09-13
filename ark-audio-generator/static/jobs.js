/* ═══════════════════════════════════════════════════════════════
   Job Tracker  –  separate screen for watching generation jobs
   ═══════════════════════════════════════════════════════════════ */

'use strict';

const $ = id => document.getElementById(id);

// Optional deep-link: /jobs.html?job=<id> highlights & scrolls to one job.
const HIGHLIGHT_ID = new URLSearchParams(location.search).get('job');

let pollTimer = null;

const MODE_ICON = { vocal: '🎤', text: '✍️' };

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function fmtTime(epochSeconds) {
  if (!epochSeconds) return '';
  const d = new Date(epochSeconds * 1000);
  return d.toLocaleString();
}

// Human-friendly "time remaining", e.g. 45s / 2m 15s / 3m.
function fmtEta(seconds) {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '';
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return rem ? `${m}m ${rem}s` : `${m}m`;
}

// Extra live detail for an in-flight job: ETA and token progress.
function progressDetail(job) {
  if (job.status !== 'processing') return '';
  const parts = [];
  if (Number.isFinite(job.eta_seconds) && job.eta_seconds > 0) {
    parts.push(`~${fmtEta(job.eta_seconds)} left`);
  }
  if (job.tokens_total) {
    parts.push(`${job.tokens_done ?? 0}/${job.tokens_total} tokens`);
  }
  return parts.length ? ` · ${parts.join(' · ')}` : '';
}

function badges(job) {
  const out = [];
  if (job.genre)    out.push(`🎸 ${esc(job.genre)}`);
  if (job.mood)     out.push(`✨ ${esc(job.mood)}`);
  if (job.key)      out.push(`🎼 ${esc(job.key)}`);
  if (job.tempo)    out.push(`🥁 ${Math.round(job.tempo)} BPM`);
  if (job.duration) out.push(`⏱ ${job.duration}s`);
  return out.map(b => `<span class="meta-badge">${b}</span>`).join('');
}

function actions(job) {
  if (job.status === 'done') {
    const links = [
      `<a class="primary" href="/api/download/${job.id}?variant=mix" download>⬇️ Download</a>`,
    ];
    if (job.has_accompaniment) {
      links.push(`<a href="/api/download/${job.id}?variant=accompaniment" download>🎹 Music only</a>`);
    }
    links.push(`<button data-del="${job.id}">🗑 Remove</button>`);
    return links.join('');
  }
  if (job.status === 'error') {
    return `<button data-del="${job.id}">🗑 Remove</button>`;
  }
  return '';   // queued / processing: nothing actionable yet
}

function statusText(job) {
  if (job.status === 'queued' && job.queue_position) {
    return `In queue · position ${job.queue_position}`;
  }
  return job.message || job.status;
}

function renderJob(job) {
  const pct = job.progress ?? 0;
  const icon = MODE_ICON[job.mode] || '🎵';
  const highlight = job.id === HIGHLIGHT_ID ? ' highlight' : '';
  const barClass = job.status === 'done' ? ' done'
    : job.status === 'error' ? ' error' : '';

  return `
    <div class="job-card${highlight}" id="job-${job.id}">
      <div class="job-top">
        <div class="job-title">
          <span class="job-mode-icon">${icon}</span>
          <span>${job.mode === 'vocal' ? 'Vocal → Music' : 'Melody → Music'}</span>
        </div>
        <span class="status-pill ${job.status}">${esc(job.status)}</span>
      </div>
      <div class="job-id">${esc(job.id)}</div>
      <div class="job-bar-wrap"><div class="job-bar${barClass}" style="width:${pct}%"></div></div>
      <div class="job-msg">${esc(statusText(job))}${
        job.status === 'error' ? '' : ` · ${pct}%`
      }${progressDetail(job)}</div>
      <div class="job-meta">${badges(job)}</div>
      <div class="job-actions">${actions(job)}</div>
      <div class="job-time">Created ${fmtTime(job.created_at)}</div>
    </div>`;
}

async function loadJobs() {
  try {
    const res = await fetch('/api/jobs');
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const { jobs } = await res.json();
    renderList(jobs);
  } catch (err) {
    $('refreshNote').textContent = `⚠️ ${err.message} — retrying…`;
  }
}

function renderList(jobs) {
  const list = $('jobList');
  if (!jobs || jobs.length === 0) {
    list.innerHTML = `
      <div class="empty-state">
        <span class="big">🎵</span>
        No jobs yet. <a href="/">Generate a track</a> and it will show up here.
      </div>`;
    return;
  }
  list.innerHTML = jobs.map(renderJob).join('');

  // Wire up "Remove" buttons.
  list.querySelectorAll('button[data-del]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.getAttribute('data-del');
      btn.disabled = true;
      await fetch(`/api/job/${id}`, { method: 'DELETE' }).catch(() => {});
      loadJobs();
    });
  });

  // Scroll a deep-linked job into view once.
  if (HIGHLIGHT_ID && !renderList._scrolled) {
    const el = $(`job-${HIGHLIGHT_ID}`);
    if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'center' }); renderList._scrolled = true; }
  }

  $('refreshNote').textContent = 'Auto-refreshing every 3s…';
}

function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(loadJobs, 3000);
}

$('refreshBtn').addEventListener('click', loadJobs);

// Pause polling when the tab is hidden; resume (and refresh) when visible.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    clearInterval(pollTimer);
  } else {
    loadJobs();
    startPolling();
  }
});

loadJobs();
startPolling();
