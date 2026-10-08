// PPO integration tests and the aggregation helpers used by the lab page.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { PPO, runPPO } from '../site/js/core/ppo.js';
import { aggregate, summarize } from '../site/js/aggregate.js';

test('PPO is deterministic for a given seed', () => {
  const a = new PPO({ seed: 3, totalSteps: 1024 }).update();
  const b = new PPO({ seed: 3, totalSteps: 1024 }).update();
  assert.deepEqual(a, b);
});

test('first update reports every metric with sane values', () => {
  const m = new PPO({ seed: 1, scheme: 'orthoPPO' }).update();
  for (const k of ['return', 'entropy', 'gradNorm', 'weightNorm', 'dormant', 'saturated', 'srank', 'valueLoss', 'approxKL', 'clipFrac']) {
    assert.ok(Number.isFinite(m[k]), `${k} = ${m[k]}`);
  }
  assert.ok(Math.abs(m.entropy - Math.LN2) < 0.01, `uniform start, entropy ${m.entropy}`);
  assert.equal(m.step, 512);
});

test('PPO learns CartPole with the standard recipe', { timeout: 300000 }, () => {
  const { history } = runPPO({ seed: 1, scheme: 'orthoPPO', totalSteps: 15360 });
  const best = Math.max(...history.map((h) => h.return));
  assert.ok(best > 150, `best mean return ${best}`);
  assert.ok(history[0].return < 60, `starts near random, got ${history[0].return}`);
});

test('aggregate gives mean with min-max band and tolerates ragged seeds', () => {
  const h1 = [{ step: 512, return: 10 }, { step: 1024, return: 30 }];
  const h2 = [{ step: 512, return: 20 }];
  const pts = aggregate([h1, h2], 'return');
  assert.deepEqual(pts[0], { x: 512, y: 15, lo: 10, hi: 20, n: 2 });
  assert.deepEqual(pts[1], { x: 1024, y: 30, lo: 30, hi: 30, n: 1 });
  assert.deepEqual(aggregate([[{ step: 512, saturated: null }]], 'saturated'), []);
});

test('summarize computes solve step and final values', () => {
  const h1 = [
    { step: 512, return: 20, entropy: 0.69, weightNorm: 10, dormant: 0, saturated: null, srank: 5 },
    { step: 1024, return: 480, entropy: 0.5, weightNorm: 12, dormant: 0.1, saturated: null, srank: 7 },
  ];
  const h2 = [
    { step: 512, return: 30, entropy: 0.69, weightNorm: 10, dormant: 0, saturated: null, srank: 5 },
    { step: 1024, return: 100, entropy: 0.6, weightNorm: 11, dormant: 0.3, saturated: null, srank: 9 },
  ];
  const s = summarize([h1, h2]);
  assert.equal(s.finalReturn, 290);
  assert.deepEqual(s.finalReturnRange, [100, 480]);
  assert.equal(s.solveStep, 1024);
  assert.equal(s.solvedRuns, 1);
  assert.equal(s.saturated, null);
  assert.equal(s.srank, 8);
  assert.equal(s.weightNorm0, 10);
});
