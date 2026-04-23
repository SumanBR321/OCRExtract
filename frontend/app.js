/**
 * app.js — OCRExtract Dashboard Logic
 *
 * Responsibilities:
 *   - Poll GET /status every 2 seconds
 *   - Update all UI elements reactively
 *   - Trigger POST /start on button click
 *   - Render live log lines with timestamps & colour coding
 *   - Manage elapsed timer
 */

'use strict';

// ── Config ──────────────────────────────────────────────────────────────────
const API_BASE      = 'http://127.0.0.1:8000';
const POLL_INTERVAL = 2000;   // ms

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $start        = document.getElementById('btn-start');
const $download     = document.getElementById('btn-download');
const $clearLogs    = document.getElementById('btn-clear-logs');
const $statusDot    = document.getElementById('status-dot');
const $statusLabel  = document.getElementById('status-label');
const $progressBar  = document.getElementById('progress-bar');
const $progressPct  = document.getElementById('progress-pct');
const $progressTrack= document.getElementById('progress-track');
const $currentFile  = document.getElementById('current-file');
const $logsContainer= document.getElementById('logs-container');
const $logsDot      = document.getElementById('logs-dot');
const $statTotal    = document.getElementById('stat-total');
const $statProcessed= document.getElementById('stat-processed');
const $statRows     = document.getElementById('stat-rows');
const $statErrors   = document.getElementById('stat-errors');
const $statElapsed  = document.getElementById('stat-elapsed');

// ── State ─────────────────────────────────────────────────────────────────────
let pollTimer     = null;
let elapsedTimer  = null;
let elapsedSecs   = 0;
let lastLogCount  = 0;
let isRunning     = false;
let isComplete    = false;

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmt(n) {
  if (n === null || n === undefined) return '—';
  return Number(n).toLocaleString();
}

function fmtElapsed(secs) {
  const m = String(Math.floor(secs / 60)).padStart(2, '0');
  const s = String(secs % 60).padStart(2, '0');
  return `${m}:${s}`;
}

function nowTime() {
  const d = new Date();
  return d.toTimeString().slice(0, 8);
}

/** Classify a log message for colour coding */
function classifyLog(msg) {
  if (/❌|error|fail|fatal/i.test(msg))  return 'error';
  if (/✅|complete|success|saved/i.test(msg)) return 'success';
  if (/⚠️|warn|flag/i.test(msg))           return 'warn';
  return 'info';
}

function appendLog(msg, timestamp) {
  const line = document.createElement('div');
  line.className = `log-line ${classifyLog(msg)}`;

  const time = document.createElement('span');
  time.className = 'log-time';
  time.textContent = timestamp || nowTime();

  const text = document.createElement('span');
  text.className = 'log-msg';
  text.textContent = msg;

  line.appendChild(time);
  line.appendChild(text);
  $logsContainer.appendChild(line);

  // Auto-scroll to bottom
  $logsContainer.scrollTop = $logsContainer.scrollHeight;
}

// ── Status updater ────────────────────────────────────────────────────────────
function applyStatus(data) {
  const total     = data.total_files     ?? 0;
  const processed = data.processed_files ?? 0;
  const rows      = data.extracted_rows  ?? 0;
  const errors    = data.errors          ?? 0;
  const pct       = total > 0 ? Math.round((processed / total) * 100) : 0;

  // Stats
  $statTotal.textContent     = fmt(total);
  $statProcessed.textContent = fmt(processed);
  $statRows.textContent      = fmt(rows);
  $statErrors.textContent    = fmt(errors);

  // Progress bar
  $progressBar.style.width = `${pct}%`;
  $progressPct.textContent = `${pct}%`;
  $progressTrack.setAttribute('aria-valuenow', pct);

  if (pct >= 100) $progressBar.classList.add('complete');
  else            $progressBar.classList.remove('complete');

  // Current file
  const cf = data.current_file;
  if (cf) {
    $currentFile.textContent = cf;
    $currentFile.classList.add('active');
  } else {
    $currentFile.textContent = data.is_complete ? 'Done ✅' : '—';
    $currentFile.classList.remove('active');
  }

  // Status dot & label
  $statusDot.className = 'status-dot';
  if (data.is_running) {
    $statusDot.classList.add('running');
    $statusLabel.textContent = 'Running';
  } else if (data.is_complete) {
    $statusDot.classList.add('complete');
    $statusLabel.textContent = 'Complete';
  } else if (errors > 0 && !data.is_running) {
    $statusDot.classList.add('error');
    $statusLabel.textContent = 'Errors';
  } else {
    $statusLabel.textContent = 'Idle';
  }

  // New log lines (only append deltas)
  const logs = data.logs ?? [];
  if (logs.length > lastLogCount) {
    logs.slice(lastLogCount).forEach(msg => appendLog(msg));
    lastLogCount = logs.length;
  }

  // Running state transitions
  const wasRunning = isRunning;
  isRunning  = data.is_running;
  isComplete = data.is_complete;

  $start.disabled = isRunning;

  if (isComplete && !wasRunning && elapsedTimer) {
    // Stop elapsed timer when done
    clearInterval(elapsedTimer);
    elapsedTimer = null;
    $logsDot.style.animation = 'none';
    $logsDot.style.background = 'var(--accent)';
  }

  // Show download button when complete
  if (isComplete) {
    $download.classList.add('visible');
  }
}

// ── Polling ───────────────────────────────────────────────────────────────────
async function poll() {
  try {
    const res = await fetch(`${API_BASE}/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    applyStatus(data);

    // Stop polling when pipeline finishes
    if (data.is_complete && !data.is_running) {
      stopPolling();
      appendLog('⏹ Polling stopped — pipeline complete.', nowTime());
    }
  } catch (err) {
    console.warn('Poll error:', err.message);
    appendLog(`⚠️ Cannot reach backend: ${err.message}`, nowTime());
  }
}

function startPolling() {
  if (pollTimer) return;
  poll();                                      // immediate first call
  pollTimer = setInterval(poll, POLL_INTERVAL);
}

function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = null;
}

// ── Elapsed timer ─────────────────────────────────────────────────────────────
function startElapsedTimer() {
  elapsedSecs = 0;
  $statElapsed.textContent = fmtElapsed(0);
  clearInterval(elapsedTimer);
  elapsedTimer = setInterval(() => {
    elapsedSecs++;
    $statElapsed.textContent = fmtElapsed(elapsedSecs);
  }, 1000);
}

// ── Start pipeline ────────────────────────────────────────────────────────────
async function handleStart() {
  if (isRunning) return;

  // Reset UI
  $download.classList.remove('visible');
  $progressBar.style.width = '0%';
  $progressPct.textContent = '0%';
  $statTotal.textContent     = '—';
  $statProcessed.textContent = '—';
  $statRows.textContent      = '—';
  $statErrors.textContent    = '—';
  $currentFile.textContent   = '—';
  $currentFile.classList.remove('active');
  lastLogCount = 0;

  appendLog('▶ Sending start request to backend…', nowTime());

  try {
    const res = await fetch(`${API_BASE}/start`, { method: 'POST' });
    const data = await res.json();

    if (!res.ok) {
      appendLog(`❌ Server rejected start: ${data.detail || res.status}`, nowTime());
      return;
    }

    appendLog(`✅ ${data.message}`, nowTime());
    $start.disabled = true;
    startElapsedTimer();
    startPolling();

  } catch (err) {
    appendLog(`❌ Failed to reach backend: ${err.message}`, nowTime());
    appendLog('ℹ️ Make sure the FastAPI server is running on port 8000.', nowTime());
  }
}

// ── Download Excel ────────────────────────────────────────────────────────────
function handleDownload() {
  window.open(`${API_BASE}/download`, '_blank');
}

// ── Clear logs ────────────────────────────────────────────────────────────────
function handleClearLogs() {
  $logsContainer.innerHTML = '';
  lastLogCount = 0;
}

// ── Event listeners ───────────────────────────────────────────────────────────
$start.addEventListener('click',     handleStart);
$download.addEventListener('click',  handleDownload);
$clearLogs.addEventListener('click', handleClearLogs);

// ── Init ───────────────────────────────────────────────────────────────────────
(function init() {
  appendLog('ℹ️ OCRExtract dashboard ready. Click "Start Processing" to begin.', nowTime());

  // Do a single status check on load so the UI reflects any in-progress run
  fetch(`${API_BASE}/status`)
    .then(r => r.json())
    .then(data => {
      applyStatus(data);
      if (data.is_running) {
        appendLog('ℹ️ A pipeline is already running — attaching to it.', nowTime());
        startElapsedTimer();
        startPolling();
      }
    })
    .catch(() => {
      appendLog('⚠️ Backend not reachable yet. Start the server then refresh.', nowTime());
    });
})();
