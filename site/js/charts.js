// Minimal SVG line chart: 2px lines, optional min-max band, recessive grid,
// end dots with a surface ring, and a crosshair tooltip on hover.
// Series colors are CSS custom properties, so light/dark mode swaps for free.

const SVGNS = 'http://www.w3.org/2000/svg';

function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function niceStep(span, count) {
  const raw = span / Math.max(count, 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1;
  return step * mag;
}

export function niceTicks(lo, hi, count = 4) {
  if (!(hi > lo)) hi = lo + 1;
  const step = niceStep(hi - lo, count);
  const start = Math.floor(lo / step) * step;
  const end = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
  return ticks;
}

function logTicks(lo, hi) {
  const ticks = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) ticks.push(Math.pow(10, e));
  return ticks;
}

export function formatNumber(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e5 || a < 1e-3) return v.toExponential(1).replace('e+', 'e');
  if (a >= 1000) return Math.round(v).toLocaleString('en-US');
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

export function formatSteps(v) {
  if (v >= 1000) return `${+(v / 1000).toFixed(1)}k`;
  return String(v);
}

export class LineChart {
  // opts: { title, subtitle, xLabel, yFormat, xFormat, logY, yMin, yMax, height }
  constructor(container, opts = {}) {
    this.opts = { height: 200, yFormat: formatNumber, xFormat: formatSteps, ...opts };
    this.series = [];
    this.root = document.createElement('figure');
    this.root.className = 'chart';
    const cap = document.createElement('figcaption');
    cap.innerHTML = `<span class="chart-title"></span><span class="chart-sub"></span>`;
    cap.querySelector('.chart-title').textContent = this.opts.title || '';
    cap.querySelector('.chart-sub').textContent = this.opts.subtitle || '';
    this.root.appendChild(cap);
    this.plot = document.createElement('div');
    this.plot.className = 'chart-plot';
    this.root.appendChild(this.plot);
    this.tip = document.createElement('div');
    this.tip.className = 'chart-tip';
    this.tip.hidden = true;
    this.plot.appendChild(this.tip);
    container.appendChild(this.root);
    this.ro = new ResizeObserver(() => this.render());
    this.ro.observe(this.plot);
  }

  // series: [{ id, label, color, points: [{ x, y, lo?, hi? }] }]
  setData(series) {
    this.series = series;
    this.render();
  }

  render() {
    const width = this.plot.clientWidth;
    if (!width) return;
    const height = this.opts.height;
    const m = { top: 8, right: 14, bottom: 26, left: 48 };
    const w = width - m.left - m.right;
    const h = height - m.top - m.bottom;
    const logY = !!this.opts.logY;

    const pts = this.series.flatMap((s) => s.points);
    let xMax = Math.max(1, ...pts.map((p) => p.x));
    if (this.opts.xMax) xMax = Math.max(xMax, this.opts.xMax);
    const xMin = this.opts.xMin ?? 0;
    const ys = pts.flatMap((p) => [p.y, p.lo, p.hi]).filter((v) => Number.isFinite(v) && (!logY || v > 0));
    let yLo = this.opts.yMin ?? (ys.length ? Math.min(...ys) : 0);
    let yHi = this.opts.yMax ?? (ys.length ? Math.max(...ys) : 1);
    if (!logY && this.opts.yMin === undefined) yLo = Math.min(yLo, 0);
    if (yHi === yLo) yHi = yLo + 1;
    let ticks;
    if (logY) {
      yLo = Math.max(yLo, 1e-12);
      ticks = logTicks(yLo, yHi);
      yLo = ticks[0];
      yHi = ticks[ticks.length - 1];
      if (ticks.length > 7) ticks = ticks.filter((_, i) => i % Math.ceil(ticks.length / 6) === 0);
    } else {
      ticks = niceTicks(yLo, yHi, 4);
      if (this.opts.yMin === undefined) yLo = ticks[0];
      if (this.opts.yMax === undefined) yHi = ticks[ticks.length - 1];
      ticks = ticks.filter((t) => t >= yLo - 1e-12 && t <= yHi + 1e-12);
    }
    const sx = (x) => m.left + ((x - xMin) / (xMax - xMin || 1)) * w;
    const sy = logY
      ? (y) => m.top + h - ((Math.log10(Math.max(y, yLo)) - Math.log10(yLo)) / (Math.log10(yHi) - Math.log10(yLo))) * h
      : (y) => m.top + h - ((Math.min(Math.max(y, yLo), yHi) - yLo) / (yHi - yLo)) * h;
    this.scale = { sx, sy, m, w, h, xMin, xMax };

    const old = this.plot.querySelector('svg');
    if (old) old.remove();
    const svg = el('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img' });
    svg.setAttribute('aria-label', this.opts.title || 'chart');
    const grid = el('g', { class: 'grid' }, svg);
    for (const t of ticks) {
      const y = sy(t);
      el('line', { x1: m.left, x2: m.left + w, y1: y, y2: y }, grid);
      const label = el('text', { x: m.left - 6, y: y + 3.5, 'text-anchor': 'end', class: 'tick' }, grid);
      label.textContent = this.opts.yFormat(t);
    }
    el('line', { x1: m.left, x2: m.left + w, y1: m.top + h, y2: m.top + h, class: 'baseline' }, svg);
    const xt = niceTicks(xMin, xMax, Math.max(2, Math.floor(w / 80))).filter((t) => t >= xMin && t <= xMax);
    for (const t of xt) {
      const label = el('text', { x: sx(t), y: m.top + h + 17, 'text-anchor': 'middle', class: 'tick' }, svg);
      label.textContent = this.opts.xFormat(t);
    }

    for (const s of this.series) {
      const ps = s.points.filter((p) => Number.isFinite(p.y) && (!logY || p.y > 0));
      if (!ps.length) continue;
      const banded = ps.filter((p) => Number.isFinite(p.lo) && Number.isFinite(p.hi));
      if (banded.length > 1 && banded.some((p) => p.hi !== p.lo)) {
        const top = banded.map((p) => `${sx(p.x).toFixed(1)},${sy(p.hi).toFixed(1)}`);
        const bot = banded.map((p) => `${sx(p.x).toFixed(1)},${sy(p.lo).toFixed(1)}`).reverse();
        el('polygon', { points: [...top, ...bot].join(' '), class: 'band', style: `fill: ${s.color}` }, svg);
      }
      const d = ps.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join('');
      el('path', { d, class: 'line', style: `stroke: ${s.color}` }, svg);
      const lastP = ps[ps.length - 1];
      el('circle', { cx: sx(lastP.x), cy: sy(lastP.y), r: 4, class: 'dot', style: `fill: ${s.color}` }, svg);
    }

    this.cross = el('line', { y1: m.top, y2: m.top + h, class: 'crosshair', visibility: 'hidden' }, svg);
    const hit = el('rect', { x: m.left, y: m.top, width: w, height: h, class: 'hit' }, svg);
    hit.addEventListener('pointermove', (e) => this.hover(e, svg));
    hit.addEventListener('pointerleave', () => this.unhover());
    this.plot.insertBefore(svg, this.tip);
  }

  hover(e, svg) {
    const { sx, m, w, xMin, xMax } = this.scale;
    const rect = svg.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const x = xMin + ((px - m.left) / w) * (xMax - xMin);
    const rows = [];
    let snapX = null;
    for (const s of this.series) {
      let best = null;
      for (const p of s.points) if (!best || Math.abs(p.x - x) < Math.abs(best.x - x)) best = p;
      if (!best) continue;
      if (snapX === null || Math.abs(best.x - x) < Math.abs(snapX - x)) snapX = best.x;
      rows.push({ s, p: best });
    }
    if (snapX === null) return this.unhover();
    this.cross.setAttribute('x1', sx(snapX));
    this.cross.setAttribute('x2', sx(snapX));
    this.cross.setAttribute('visibility', 'visible');
    const f = this.opts.yFormat;
    this.tip.innerHTML = `<div class="tip-head">${this.opts.xLabel || 'Step'} ${this.opts.xFormat(snapX)}</div>` +
      rows
        .sort((a, b) => b.p.y - a.p.y)
        .map(({ s, p }) => {
          const range = Number.isFinite(p.lo) && p.hi !== p.lo ? ` <span class="tip-range">${f(p.lo)}–${f(p.hi)}</span>` : '';
          return `<div class="tip-row"><i style="background:${s.color}"></i><span>${s.label}</span><b>${f(p.y)}</b>${range}</div>`;
        })
        .join('');
    this.tip.hidden = false;
    const tipW = this.tip.offsetWidth;
    const left = sx(snapX) + 12 + tipW > rect.width ? sx(snapX) - 12 - tipW : sx(snapX) + 12;
    this.tip.style.left = `${Math.max(0, left)}px`;
    this.tip.style.top = `${m.top}px`;
  }

  unhover() {
    this.tip.hidden = true;
    if (this.cross) this.cross.setAttribute('visibility', 'hidden');
  }
}
