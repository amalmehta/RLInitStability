// Generates docs/LITERATURE.md from site/data/papers.json, so the review can
// be read on GitHub without running the site. Run: npm run literature
import { readFileSync, writeFileSync } from 'node:fs';

const root = new URL('../', import.meta.url);
const data = JSON.parse(readFileSync(new URL('site/data/papers.json', root), 'utf8'));

const authors = (a) => (a.length <= 3 ? a.join(', ') : `${a.slice(0, 3).join(', ')} et al.`);

const lines = [
  '# Literature Review: Initialization and Training Stability in Deep RL',
  '',
  `${data.papers.length} papers in ${data.themes.length} themes. Titles, authors and years come from the arXiv API; summaries are this project's own.`,
  'The same data powers the searchable [Literature page](../site/literature.html) of the website.',
  '',
  '_Generated from `site/data/papers.json` by `npm run literature`. Edit the JSON, not this file._',
  '',
  '## Themes',
  '',
  ...data.themes.map((t) => `- [${t.title}](#${t.title.toLowerCase().replace(/[^a-z0-9 -]/g, '').replace(/ /g, '-')}) (${data.papers.filter((p) => p.theme === t.id).length})`),
  '',
];

for (const t of data.themes) {
  lines.push(`## ${t.title}`, '', t.summary, '');
  for (const p of data.papers.filter((x) => x.theme === t.id)) {
    lines.push(`### [${p.title}](${p.url})`, '');
    lines.push(`${authors(p.authors)} · ${p.venue}${p.arxiv ? ` · arXiv ${p.arxiv}` : ''}`, '');
    lines.push(p.finding, '');
    lines.push(`**Why it matters:** ${p.lesson}`, '');
  }
}

writeFileSync(new URL('docs/LITERATURE.md', root), lines.join('\n'));
console.log(`Wrote docs/LITERATURE.md (${data.papers.length} papers)`);
