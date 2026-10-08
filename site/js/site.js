// Shared page chrome: the light / dark / system theme toggle.

const ORDER = ['system', 'light', 'dark'];
const LABEL = { system: 'Theme: system', light: 'Theme: light', dark: 'Theme: dark' };

function read() {
  try {
    return localStorage.getItem('theme') || 'system';
  } catch {
    return 'system';
  }
}

function apply(mode) {
  if (mode === 'system') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', mode);
  const btn = document.querySelector('.theme-toggle');
  if (btn) btn.textContent = LABEL[mode];
}

apply(read());

document.addEventListener('DOMContentLoaded', () => {
  const btn = document.querySelector('.theme-toggle');
  if (!btn) return;
  apply(read());
  btn.addEventListener('click', () => {
    const next = ORDER[(ORDER.indexOf(read()) + 1) % ORDER.length];
    try {
      localStorage.setItem('theme', next);
    } catch {
      /* storage unavailable: theme still applies for this page view */
    }
    apply(next);
    // Charts read colors from CSS variables; re-render so SVG picks them up.
    window.dispatchEvent(new Event('themechange'));
  });
});
