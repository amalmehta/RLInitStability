// Literature page: themed paper list, search and theme filters, and a
// year-by-theme dot timeline. Data comes from data/papers.json.

const SVGNS = 'http://www.w3.org/2000/svg';

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

export function authorLine(authors) {
  if (authors.length <= 3) return authors.join(', ');
  return `${authors.slice(0, 3).join(', ')} et al.`;
}

export function matches(paper, query) {
  if (!query) return true;
  const hay = [paper.title, paper.authors.join(' '), paper.finding, paper.lesson, paper.tags.join(' '), paper.venue, paper.year]
    .join(' ')
    .toLowerCase();
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((w) => hay.includes(w));
}

function paperHTML(p) {
  return `<article class="paper" id="${esc(p.id)}">
    <h3><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a></h3>
    <div class="meta">${esc(authorLine(p.authors))} · ${esc(p.venue)}${p.arxiv ? ` · arXiv ${esc(p.arxiv)}` : ''}</div>
    <p class="finding">${esc(p.finding)}</p>
    <p class="lesson"><b>Why it matters:</b> ${esc(p.lesson)}</p>
    <div class="tags">${p.tags.map((t) => `<span class="tag">${esc(t)}</span>`).join('')}</div>
  </article>`;
}

function renderTimeline(host, data) {
  host.innerHTML = '';
  const width = host.clientWidth || 800;
  const labelW = width < 560 ? 0 : 200;
  const rowH = 30;
  const top = 6;
  const years = data.papers.map((p) => +p.year);
  const y0 = Math.min(...years);
  const y1 = Math.max(...years);
  const left = labelW + 12;
  const right = 14;
  const plotW = width - left - right;
  const height = top + data.themes.length * rowH + 24;
  const sx = (y) => left + ((y - y0) / (y1 - y0 || 1)) * plotW;
  const svg = document.createElementNS(SVGNS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('height', height);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Dot timeline of papers by year and theme');
  const add = (tag, attrs, text) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    svg.appendChild(n);
    return n;
  };
  data.themes.forEach((t, i) => {
    const cy = top + i * rowH + rowH / 2;
    add('line', { x1: left, x2: left + plotW, y1: cy, y2: cy });
    if (labelW) add('text', { x: 0, y: cy + 4, class: 'row-label' }, t.short || t.title);
    const inRow = data.papers.filter((p) => p.theme === t.id);
    const perYear = {};
    for (const p of inRow) {
      const k = perYear[p.year] = (perYear[p.year] || 0) + 1;
      // spread same-year papers sideways so dots don't overlap
      const offset = (k % 2 === 0 ? 1 : -1) * Math.floor(k / 2) * 9;
      const c = add('circle', {
        cx: Math.max(left, Math.min(left + plotW, sx(+p.year) + offset)),
        cy,
        r: 5,
        style: `fill: var(--series-${(i % 7) + 1})`,
        tabindex: 0,
        'data-id': p.id,
      });
      const title = document.createElementNS(SVGNS, 'title');
      title.textContent = `${p.title} (${p.year})`;
      c.appendChild(title);
      const go = () => {
        location.hash = p.id;
      };
      c.addEventListener('click', go);
      c.addEventListener('keydown', (e) => e.key === 'Enter' && go());
    }
  });
  for (let y = y0; y <= y1; y++) {
    if ((y - y0) % (width < 560 ? 3 : 1)) continue;
    add('text', { x: sx(y), y: height - 6, 'text-anchor': 'middle', class: 'tick' }, String(y));
  }
  host.appendChild(svg);
  if (!labelW) {
    const legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML = data.themes
      .map((t, i) => `<span><i style="background: var(--series-${(i % 7) + 1})"></i>${esc(t.title)}</span>`)
      .join('');
    host.appendChild(legend);
  }
}

async function main() {
  const data = await (await fetch('data/papers.json')).json();
  const active = new Set(data.themes.map((t) => t.id));
  const chips = document.getElementById('theme-chips');
  chips.innerHTML = data.themes
    .map(
      (t, i) => `<label class="chip"><input type="checkbox" value="${t.id}" checked>
        <i style="background: var(--series-${(i % 7) + 1})"></i>${esc(t.title)}</label>`,
    )
    .join('');
  const search = document.getElementById('search');
  const list = document.getElementById('papers');
  const count = document.getElementById('result-count');

  function render() {
    const q = search.value.trim();
    let shown = 0;
    list.innerHTML = data.themes
      .filter((t) => active.has(t.id))
      .map((t) => {
        const ps = data.papers.filter((p) => p.theme === t.id && matches(p, q));
        if (!ps.length) return '';
        shown += ps.length;
        return `<section class="theme-block" id="theme-${t.id}">
          <div class="theme-head"><h2>${esc(t.title)}</h2><span class="count">${ps.length}</span></div>
          <p>${esc(t.summary)}</p>
          ${ps.map(paperHTML).join('')}
        </section>`;
      })
      .join('');
    count.textContent = `${shown} of ${data.papers.length} papers`;
    if (!shown) list.innerHTML = '<p class="muted">No papers match. Try a shorter search or turn more themes on.</p>';
  }

  chips.addEventListener('change', (e) => {
    if (e.target.checked) active.add(e.target.value);
    else active.delete(e.target.value);
    render();
  });
  search.addEventListener('input', render);
  render();

  const tl = document.getElementById('timeline');
  renderTimeline(tl, data);
  let lastW = tl.clientWidth;
  new ResizeObserver(() => {
    if (tl.clientWidth !== lastW) {
      lastW = tl.clientWidth;
      renderTimeline(tl, data);
    }
  }).observe(tl);

  // Jumping to a paper: clear filters first if it's hidden.
  function reveal() {
    const id = decodeURIComponent(location.hash.slice(1));
    if (!id) return;
    if (!document.getElementById(id)) {
      search.value = '';
      data.themes.forEach((t) => active.add(t.id));
      chips.querySelectorAll('input').forEach((c) => (c.checked = true));
      render();
    }
    const target = document.getElementById(id);
    if (target) target.scrollIntoView({ block: 'center' });
  }
  window.addEventListener('hashchange', reveal);
  reveal();
}

if (typeof document !== 'undefined') main();
