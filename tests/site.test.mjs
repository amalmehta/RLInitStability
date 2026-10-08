// Checks on the site's data and pages: the literature dataset is well formed,
// every in-page paper link resolves, and every page links its assets.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';

import { matches, authorLine } from '../site/js/literature.js';

const site = new URL('../site/', import.meta.url);
const read = (p) => readFileSync(new URL(p, site), 'utf8');
const data = JSON.parse(read('data/papers.json'));

test('papers.json is well formed', () => {
  const themeIds = new Set(data.themes.map((t) => t.id));
  const ids = new Set();
  assert.ok(data.papers.length >= 40, `only ${data.papers.length} papers`);
  for (const p of data.papers) {
    for (const k of ['id', 'title', 'year', 'venue', 'url', 'finding', 'lesson']) {
      assert.ok(typeof p[k] === 'string' && p[k].length, `${p.id}: missing ${k}`);
    }
    assert.ok(Array.isArray(p.authors) && p.authors.length, `${p.id}: authors`);
    assert.ok(themeIds.has(p.theme), `${p.id}: unknown theme ${p.theme}`);
    assert.ok(!ids.has(p.id), `duplicate id ${p.id}`);
    ids.add(p.id);
    if (p.arxiv) assert.equal(p.url, `https://arxiv.org/abs/${p.arxiv}`);
    assert.match(p.year, /^(19|20)\d\d$/);
  }
  for (const t of data.themes) assert.ok(data.papers.some((p) => p.theme === t.id), `empty theme ${t.id}`);
});

test('every literature.html#id link on the pages points at a real paper', () => {
  const ids = new Set(data.papers.map((p) => p.id));
  for (const page of ['index.html', 'demo.html']) {
    for (const [, id] of read(page).matchAll(/literature\.html#([^"]+)"/g)) {
      assert.ok(ids.has(id), `${page} links to unknown paper ${id}`);
    }
  }
});

test('the overview paper count matches the data', () => {
  const m = read('index.html').match(/id="paper-count">(\d+)</);
  assert.equal(+m[1], data.papers.length);
  const themes = read('index.html').match(/papers in (\d+) themes/);
  assert.equal(+themes[1], data.themes.length);
});

test('pages reference assets that exist', () => {
  for (const page of ['index.html', 'literature.html', 'demo.html']) {
    const html = read(page);
    for (const [, src] of html.matchAll(/(?:src|href)="((?:js|css|data)\/[^"]+)"/g)) {
      assert.ok(existsSync(new URL(src, site)), `${page}: missing ${src}`);
    }
    assert.match(html, /<title>[^<]+<\/title>/);
    assert.match(html, /name="viewport"/);
  }
});

test('search matches every word across fields', () => {
  const p = data.papers.find((x) => x.arxiv === '2302.12902');
  assert.ok(matches(p, 'dormant sokar'));
  assert.ok(matches(p, 'ReDo'));
  assert.ok(!matches(p, 'dormant transformer'));
  assert.equal(authorLine(['A', 'B', 'C', 'D']), 'A, B, C et al.');
});
