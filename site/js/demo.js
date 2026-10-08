// Live lab controller: runs PPO training in a pool of Web Workers, aggregates
// metrics across seeds, and drives the charts, summary table and viewer.
// Also renders the signal-propagation explorer.

import { SCHEMES, SCHEME_IDS } from './core/init.js';
import { DEFAULT_CONFIG } from './core/ppo.js';
import { propagate } from './core/signal.js';
import { deserialize, forward } from './core/nn.js';
import { CartPole, CARTPOLE } from './core/cartpole.js';
import { Rng } from './core/rng.js';
import { LineChart, formatNumber } from './charts.js';
import { aggregate, summarize } from './aggregate.js';

const $ = (id) => document.getElementById(id);
const color = (scheme) => `var(--series-${SCHEME_IDS.indexOf(scheme) + 1})`;
const DEFAULT_ON = ['orthoPPO', 'pytorch', 'orthoPlain', 'large'];
const pct = (v) => (v === null || v === undefined || Number.isNaN(v) ? '–' : `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`);

const METRICS = [
  { key: 'return', title: 'Episode return', subtitle: 'Mean over episodes that finished in each rollout. 500 is the maximum.', yMin: 0, yMax: 500 },
  { key: 'entropy', title: 'Policy entropy', subtitle: 'ln 2 ≈ 0.69 means a uniform policy. A low value at step 0 means the init already committed to actions.', yMin: 0, yMax: 0.7 },
  { key: 'gradNorm', title: 'Actor gradient norm', subtitle: 'Mean pre-clip L2 norm per minibatch. Log scale.', logY: true },
  { key: 'weightNorm', title: 'Actor weight norm', subtitle: 'L2 norm of all actor parameters.' },
  { key: 'dormant', title: 'Dormant neurons', subtitle: 'Share of hidden units with normalized activation ≤ 0.1 (Sokar et al. 2023).', yMin: 0, yFormat: pct },
  { key: 'saturated', title: 'Saturated tanh units', subtitle: 'Share of activations with |h| > 0.99, where tanh gradients vanish. Empty for ReLU.', yMin: 0, yFormat: pct },
  { key: 'srank', title: 'Feature rank (srank)', subtitle: 'Effective rank of the last hidden layer, δ = 0.01 (Kumar et al. 2021).', yMin: 0 },
  { key: 'valueLoss', title: 'Critic loss', subtitle: 'Mean squared TD(λ) error. Log scale.', logY: true },
];

// ---------- controls ----------
const checks = $('scheme-checks');
checks.innerHTML = SCHEME_IDS.map(
  (id) => `<label class="chip" title="${SCHEMES[id].note}"><input type="checkbox" value="${id}" ${DEFAULT_ON.includes(id) ? 'checked' : ''}>
    <i style="background:${color(id)}"></i>${SCHEMES[id].label}</label>`,
).join('');

function readConfig() {
  const depth = +$('depth').value;
  const width = +$('width').value;
  return {
    schemes: [...checks.querySelectorAll('input:checked')].map((c) => c.value),
    seeds: +$('seeds').value,
    base: {
      ...DEFAULT_CONFIG,
      activation: $('activation').value,
      hidden: Array(depth).fill(width),
      lr: +$('lr').value,
      totalSteps: +$('steps').value,
      maxGradNorm: $('clip').checked ? 0.5 : null,
    },
  };
}

// ---------- charts ----------
const charts = METRICS.map((m) => ({ m, chart: new LineChart($('charts'), { ...m, xLabel: 'Step' }) }));

const state = { runs: new Map(), schemes: [], workers: new Set(), queue: [], totalUpdates: 0, doneUpdates: 0, running: false, base: null };

function renderLegend(host, schemes, labelFn = (id) => SCHEMES[id].label) {
  host.innerHTML = schemes.map((id) => `<span><i style="background:${color(id)}"></i>${labelFn(id)}</span>`).join('');
}

function runsFor(scheme) {
  return [...state.runs.values()].filter((r) => r.scheme === scheme);
}

let renderPending = false;
function scheduleRender() {
  if (renderPending) return;
  renderPending = true;
  setTimeout(() => {
    renderPending = false;
    renderLab();
  }, 200);
}

function renderLab() {
  for (const { m, chart } of charts) {
    chart.opts.xMax = state.base ? state.base.totalSteps : undefined;
    chart.setData(
      state.schemes.map((scheme) => ({
        id: scheme,
        label: SCHEMES[scheme].label,
        color: color(scheme),
        points: aggregate(runsFor(scheme).map((r) => r.history), m.key),
      })),
    );
  }
  renderSummary();
  const pctDone = state.totalUpdates ? state.doneUpdates / state.totalUpdates : 0;
  $('progress-bar').style.width = `${(pctDone * 100).toFixed(1)}%`;
}

function renderSummary() {
  const rows = state.schemes.map((scheme) => ({ scheme, s: summarize(runsFor(scheme).map((r) => r.history)) }));
  const head = `<thead><tr><th>Scheme</th><th>Runs</th><th class="num">Final return</th><th class="num">Steps to 475</th>
    <th class="num">Entropy at start</th><th class="num">Dormant (final)</th><th class="num">Saturated (final)</th><th class="num">srank (final)</th><th class="num">Weight norm (start → final)</th></tr></thead>`;
  const body = rows
    .map(({ scheme, s }) => {
      const runs = runsFor(scheme);
      const diverged = runs.filter((r) => r.diverged).length;
      return `<tr><td><span class="swatch" style="background:${color(scheme)}"></span>${SCHEMES[scheme].label}</td>
        <td>${runs.filter((r) => r.done).length}/${runs.length}${diverged ? ` <span style="color:var(--critical)">(${diverged} diverged)</span>` : ''}</td>
        <td class="num">${formatNumber(s.finalReturn)}${s.finalReturnRange ? ` <span class="muted">(${s.finalReturnRange.map(formatNumber).join('–')})</span>` : ''}</td>
        <td class="num">${s.solveStep === null ? '–' : `${formatNumber(s.solveStep)}${s.solvedRuns < s.runs ? ` <span class="muted">(${s.solvedRuns}/${s.runs})</span>` : ''}`}</td>
        <td class="num">${formatNumber(s.entropy0)}</td>
        <td class="num">${pct(s.dormant)}</td>
        <td class="num">${pct(s.saturated)}</td>
        <td class="num">${formatNumber(s.srank)}</td>
        <td class="num">${formatNumber(s.weightNorm0)} → ${formatNumber(s.weightNorm)}</td></tr>`;
    })
    .join('');
  $('summary').innerHTML = rows.length ? head + `<tbody>${body}</tbody>` : '<tbody><tr><td class="muted">Results appear here as runs progress.</td></tr></tbody>';
}

// ---------- worker pool ----------
function poolSize() {
  const cores = navigator.hardwareConcurrency || 4;
  return Math.max(1, Math.min(8, cores - 1));
}

function launchNext() {
  if (!state.running) return;
  const job = state.queue.shift();
  if (!job) {
    if (!state.workers.size) finish();
    return;
  }
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  state.workers.add(worker);
  const run = state.runs.get(job.runId);
  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'metrics') {
      run.history.push(msg.m);
      state.doneUpdates += 1;
      scheduleRender();
    } else if (msg.type === 'snapshot') {
      run.snapshot = { step: msg.step, net: msg.net };
      if (run.seed === 1) viewer.offer(run);
    } else if (msg.type === 'done' || msg.type === 'error') {
      run.done = true;
      run.diverged = !!msg.diverged;
      if (msg.type === 'error') run.error = msg.message;
      if (run.diverged || msg.type === 'error') state.totalUpdates -= Math.max(0, run.expected - run.history.length);
      worker.terminate();
      state.workers.delete(worker);
      scheduleRender();
      launchNext();
    }
  };
  worker.onerror = (err) => {
    run.done = true;
    run.error = err.message || 'worker failed';
    worker.terminate();
    state.workers.delete(worker);
    $('status').textContent = `A run failed: ${run.error}`;
    launchNext();
  };
  worker.postMessage({ type: 'start', runId: job.runId, config: { ...state.base, scheme: run.scheme, seed: run.seed } });
}

function start() {
  const { schemes, seeds, base } = readConfig();
  if (!schemes.length) {
    $('status').textContent = 'Select at least one init scheme.';
    return;
  }
  stopAll();
  state.runs.clear();
  state.schemes = schemes;
  state.base = base;
  state.queue = [];
  const updatesPerRun = Math.ceil(base.totalSteps / base.rolloutSteps);
  // Interleave by seed so every scheme gets a first curve early.
  for (let seed = 1; seed <= seeds; seed++) {
    for (const scheme of schemes) {
      const runId = `${scheme}-${seed}`;
      state.runs.set(runId, { runId, scheme, seed, history: [], done: false, expected: updatesPerRun });
      state.queue.push({ runId });
    }
  }
  state.totalUpdates = updatesPerRun * state.runs.size;
  state.doneUpdates = 0;
  state.running = true;
  state.startedAt = performance.now();
  renderLegend($('legend'), schemes);
  viewer.setSchemes(schemes);
  $('start').disabled = true;
  $('stop').disabled = false;
  $('status').textContent = `Training ${state.runs.size} runs, ${poolSize()} at a time…`;
  for (let i = 0; i < poolSize(); i++) launchNext();
  renderLab();
  tickStatus();
}

function tickStatus() {
  if (!state.running) return;
  const done = [...state.runs.values()].filter((r) => r.done).length;
  const secs = (performance.now() - state.startedAt) / 1000;
  const frac = state.totalUpdates ? state.doneUpdates / state.totalUpdates : 0;
  const eta = frac > 0.02 ? Math.max(0, secs / frac - secs) : null;
  $('status').textContent = `Training: ${done}/${state.runs.size} runs finished · ${Math.round(frac * 100)}%${eta !== null ? ` · about ${Math.ceil(eta)} s left` : ''}`;
  setTimeout(tickStatus, 1000);
}

function stopAll() {
  for (const w of state.workers) w.terminate();
  state.workers.clear();
  state.queue = [];
  state.running = false;
}

function finish() {
  state.running = false;
  $('start').disabled = false;
  $('stop').disabled = true;
  const secs = ((performance.now() - state.startedAt) / 1000).toFixed(0);
  const errors = [...state.runs.values()].filter((r) => r.error);
  $('status').textContent = errors.length
    ? `Finished with ${errors.length} failed run(s): ${errors[0].error}`
    : `Done: ${state.runs.size} runs in ${secs} s. Hover the charts to compare values.`;
  renderLab();
}

$('start').addEventListener('click', start);
$('stop').addEventListener('click', () => {
  stopAll();
  $('start').disabled = false;
  $('stop').disabled = true;
  $('status').textContent = 'Stopped. Partial results are kept below.';
  renderLab();
});

// ---------- CartPole viewer ----------
const viewer = (() => {
  const canvas = $('cartpole');
  const ctx = canvas.getContext('2d');
  const select = $('viewer-run');
  const info = $('viewer-info');
  const env = new CartPole(new Rng(4242));
  const rng = new Rng(99);
  let net = null;
  let pending = null;
  let netStep = 0;
  let obs = env.reset();
  let t = 0;
  let lastLen = null;
  let acc = 0;
  let last = performance.now();

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function draw() {
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const scale = W / (CARTPOLE.xThreshold * 2 * 1.15);
    const groundY = H * 0.72;
    ctx.strokeStyle = css('--axis');
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, groundY);
    ctx.lineTo(W, groundY);
    ctx.stroke();
    // track limits
    for (const s of [-1, 1]) {
      const x = W / 2 + s * CARTPOLE.xThreshold * scale;
      ctx.beginPath();
      ctx.moveTo(x, groundY - 8);
      ctx.lineTo(x, groundY + 8);
      ctx.stroke();
    }
    const [x, , theta] = obs;
    const cx = W / 2 + x * scale;
    const cartW = 64;
    const cartH = 30;
    ctx.fillStyle = css('--text-secondary');
    ctx.fillRect(cx - cartW / 2, groundY - cartH, cartW, cartH);
    const poleLen = 2 * CARTPOLE.length * scale;
    const px = cx + Math.sin(theta) * poleLen;
    const py = groundY - cartH - Math.cos(theta) * poleLen;
    const sel = select.value;
    ctx.strokeStyle = sel ? css(`--series-${SCHEME_IDS.indexOf(sel) + 1}`) : css('--accent');
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx, groundY - cartH);
    ctx.lineTo(px, py);
    ctx.stroke();
    ctx.fillStyle = css('--text-primary');
    ctx.beginPath();
    ctx.arc(cx, groundY - cartH, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = css('--text-muted');
    ctx.font = '22px system-ui, sans-serif';
    ctx.fillText(`t = ${t}`, 14, 30);
  }

  function step() {
    if (!net) return;
    const { out } = forward(net, obs, 1);
    const m = Math.max(out[0], out[1]);
    const p0 = Math.exp(out[0] - m) / (Math.exp(out[0] - m) + Math.exp(out[1] - m));
    const a = rng.next() < p0 ? 0 : 1;
    const r = env.step(a);
    t += 1;
    obs = r.obs;
    if (r.terminated || r.truncated) {
      lastLen = t;
      t = 0;
      obs = env.reset();
      if (pending) {
        net = deserialize(pending.net);
        netStep = pending.step;
        pending = null;
      }
      updateInfo();
    }
  }

  function updateInfo() {
    if (!net) {
      info.textContent = 'No policy yet. Press Train.';
      return;
    }
    info.textContent = `Policy from step ${netStep.toLocaleString('en-US')}${lastLen !== null ? ` · last episode lasted ${lastLen} steps` : ''}.`;
  }

  function loop(now) {
    acc += Math.min(now - last, 100);
    last = now;
    const dt = CARTPOLE.tau * 1000;
    while (acc >= dt) {
      step();
      acc -= dt;
    }
    draw();
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  select.addEventListener('change', () => {
    net = null;
    pending = null;
    t = 0;
    lastLen = null;
    obs = env.reset();
    const run = state.runs.get(`${select.value}-1`);
    if (run && run.snapshot) {
      net = deserialize(run.snapshot.net);
      netStep = run.snapshot.step;
    }
    updateInfo();
  });

  return {
    setSchemes(schemes) {
      const prev = select.value;
      select.innerHTML = schemes.map((id) => `<option value="${id}">${SCHEMES[id].label}</option>`).join('');
      if (schemes.includes(prev)) select.value = prev;
      net = null;
      pending = null;
      lastLen = null;
      updateInfo();
    },
    offer(run) {
      if (run.scheme !== select.value) return;
      if (!net) {
        net = deserialize(run.snapshot.net);
        netStep = run.snapshot.step;
        t = 0;
        obs = env.reset();
        updateInfo();
      } else {
        pending = run.snapshot;
      }
    },
  };
})();

// ---------- signal propagation ----------
const SP_SCHEMES = SCHEME_IDS.filter((id) => id !== 'orthoPlain');
const spLabel = (id) => (id === 'orthoPPO' ? 'Orthogonal (gain √2)' : SCHEMES[id].label);
const spCharts = [
  new LineChart($('sp-charts'), {
    title: 'Activation spread by layer',
    subtitle: 'Standard deviation of each layer’s output. Flat is healthy.',
    logY: true,
    xLabel: 'Layer',
    xMin: 1,
    xFormat: (v) => String(Math.round(v)),
  }),
  new LineChart($('sp-charts'), {
    title: 'Weight-gradient norm by layer',
    subtitle: 'Backpropagated from a random loss at the output. Flat is healthy.',
    logY: true,
    xLabel: 'Layer',
    xMin: 1,
    xFormat: (v) => String(Math.round(v)),
  }),
];

function renderSignal() {
  const opts = {
    activation: $('sp-activation').value,
    depth: +$('sp-depth').value,
    width: +$('sp-width').value,
  };
  const results = SP_SCHEMES.map((id) => ({ id, ...propagate(id, opts) }));
  const series = (key) =>
    results.map((r) => ({
      id: r.id,
      label: spLabel(r.id),
      color: color(r.id),
      points: r[key].map((y, i) => ({ x: i + 1, y })),
    }));
  spCharts[0].setData(series('actStd'));
  spCharts[1].setData(series('gradNorm'));
}

renderLegend($('sp-legend'), SP_SCHEMES, spLabel);
for (const id of ['sp-activation', 'sp-depth', 'sp-width']) $(id).addEventListener('change', renderSignal);
renderSignal();
renderLab();
