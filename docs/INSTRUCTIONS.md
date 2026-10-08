# Instructions

## Requirements

- A modern browser (Chrome, Edge, Safari 15+ or Firefox 114+) with module Web Workers.
- Python 3.7+ or any other static file server, to serve the site.
- Node.js 20+, only for the tests and the literature generator.

There is nothing to install: no `npm install`, no build step.

## Run the website

```bash
npm start
```

Then open <http://localhost:8765>.
`npm start` is just `python3 -m http.server 8765 --directory site`, so any static server pointed at `site/` works too.
Opening the HTML files directly (`file://`) does not work, because browsers block ES modules and workers there.

## Use it

**Overview** (`index.html`): seven findings from the literature, each linked to its papers.

**Literature** (`literature.html`):
- Hover a dot on the timeline to see a paper's title, and click it to jump to the paper.
- Search matches every word you type across titles, authors, findings and tags.
- Theme chips show or hide whole sections.
- Every card links to the paper itself.

**Live lab** (`demo.html`):
1. Tick the init schemes to compare. Hover a chip to see that scheme's exact formula.
2. Choose the activation, network depth and width, learning rate, step budget, number of seeds and whether to clip gradients.
3. Press **Train**. Runs execute in parallel Web Workers, and the charts fill in live: the line is the mean over seeds, and the shaded band is the range from lowest to highest seed. Hover any chart for exact values.
4. **Watch a policy** plays the latest snapshot of seed 1 for the chosen scheme.
5. The **Summary table** lists final return, steps until return first reached 475, starting entropy, and final health metrics per scheme.
6. **Stop** ends all runs but keeps partial results.

Suggested experiments:
- **Default settings:** compare start-of-training entropy and weight norm between "Orthogonal + small heads" and "Large normal N(0, 1)".
- **ReLU:** switch the activation to ReLU and watch the dormant-neuron share for N(0, 1) start near 50%.
- **Signal propagation:** set ReLU and depth 40, then compare He with the PyTorch default.

A default comparison (4 schemes × 3 seeds × 50k steps) takes about a minute on an idle 8-core laptop.

## Test

```bash
npm test
```

This runs the unit, integration and site-consistency tests with Node's built-in runner, in about 15 s.

## Update the literature

1. Edit `site/data/papers.json`. Each paper needs `id`, `theme`, `title`, `authors`, `year`, `venue`, `url`, `tags`, `finding` and `lesson`. Set `arxiv` to the arXiv ID, or to `null` for other sources.
2. Regenerate the Markdown copy:

   ```bash
   npm run literature
   ```

3. If the paper count changed, update the number in `site/index.html` (`id="paper-count"`). `npm test` fails until you do.

## Regenerate the CartPole test fixture

The fixture is checked in, so this is only needed if you change the reference trajectory. It requires `pip install gym==0.21`:

```bash
python3 scripts/make-cartpole-fixture.py
```
