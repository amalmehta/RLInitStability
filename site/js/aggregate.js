// Pure helpers that combine per-seed metric histories into chart series and
// summary rows. Runs of the same config share rollout length, so the i-th
// record of every seed sits at the same environment step.

const ok = (v) => typeof v === 'number' && Number.isFinite(v);

function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}

// Mean with min-max band across seeds, at each update index any seed reached.
export function aggregate(histories, key) {
  const n = Math.max(0, ...histories.map((h) => h.length));
  const points = [];
  for (let i = 0; i < n; i++) {
    const recs = histories.map((h) => h[i]).filter(Boolean);
    const vals = recs.map((r) => r[key]).filter(ok);
    if (!vals.length) continue;
    points.push({ x: recs[0].step, y: mean(vals), lo: Math.min(...vals), hi: Math.max(...vals), n: vals.length });
  }
  return points;
}

export const SOLVED = 475;

export function summarize(histories) {
  const hs = histories.filter((h) => h.length);
  const first = (key) => mean(hs.map((h) => h[0][key]).filter(ok));
  const final = (key) => mean(hs.map((h) => h[h.length - 1][key]).filter(ok));
  const finals = hs.map((h) => h[h.length - 1].return).filter(ok);
  const solveSteps = hs
    .map((h) => h.find((r) => r.return >= SOLVED))
    .filter(Boolean)
    .map((r) => r.step);
  return {
    runs: hs.length,
    finalReturn: mean(finals),
    finalReturnRange: finals.length > 1 ? [Math.min(...finals), Math.max(...finals)] : null,
    solveStep: solveSteps.length ? mean(solveSteps) : null,
    solvedRuns: solveSteps.length,
    entropy0: first('entropy'),
    dormant: final('dormant'),
    saturated: final('saturated'),
    srank: final('srank'),
    weightNorm0: first('weightNorm'),
    weightNorm: final('weightNorm'),
  };
}
