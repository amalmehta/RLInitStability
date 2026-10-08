// Unit tests for the shared core: RNG, MLP backprop, Adam, init schemes,
// CartPole dynamics, health metrics and signal propagation.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { Rng } from '../site/js/core/rng.js';
import { createMLP, forward, backward, Adam, serialize, deserialize } from '../site/js/core/nn.js';
import { SCHEMES, SCHEME_IDS, initNet, orthogonal } from '../site/js/core/init.js';
import { CartPole, CARTPOLE } from '../site/js/core/cartpole.js';
import { symmetricEigenvalues, srank, dormantFraction, saturatedFraction } from '../site/js/core/metrics.js';
import { propagate } from '../site/js/core/signal.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} expected ${b}, got ${a}`);

test('rng is deterministic per seed and roughly standard normal', () => {
  const a = new Rng(5);
  const b = new Rng(5);
  for (let i = 0; i < 10; i++) assert.equal(a.next(), b.next());
  const r = new Rng(1);
  let s = 0;
  let s2 = 0;
  const n = 20000;
  for (let i = 0; i < n; i++) {
    const x = r.normal();
    s += x;
    s2 += x * x;
  }
  close(s / n, 0, 0.03, 'mean');
  close(s2 / n, 1, 0.05, 'variance');
});

for (const act of ['tanh', 'relu']) {
  test(`backward matches finite differences (${act})`, () => {
    const rng = new Rng(3);
    const net = initNet(createMLP([3, 5, 4, 2], act), 'xavier', 'policy', rng);
    net.layers.forEach((L) => L.b.forEach((_, i) => (L.b[i] = rng.uniform(-0.3, 0.3))));
    const n = 4;
    const X = Float64Array.from({ length: n * 3 }, () => rng.normal());
    const C = Float64Array.from({ length: n * 2 }, () => rng.normal());
    const loss = () => forward(net, X, n).out.reduce((acc, v, i) => acc + v * C[i], 0);
    const { cache } = forward(net, X, n);
    const grads = backward(net, cache, C);
    const eps = 1e-6;
    net.layers.forEach((L, k) => {
      for (const key of ['W', 'b']) {
        for (let j = 0; j < L[key].length; j++) {
          const orig = L[key][j];
          L[key][j] = orig + eps;
          const up = loss();
          L[key][j] = orig - eps;
          const down = loss();
          L[key][j] = orig;
          close(grads[k][key][j], (up - down) / (2 * eps), 1e-5, `layer ${k} ${key}[${j}]`);
        }
      }
    });
  });
}

test('Adam fits a linear regression', () => {
  const rng = new Rng(9);
  const net = initNet(createMLP([2, 1]), 'xavier', 'value', rng);
  const opt = new Adam(net, { lr: 0.05 });
  const n = 32;
  const X = Float64Array.from({ length: n * 2 }, () => rng.normal());
  const y = Array.from({ length: n }, (_, s) => 3 * X[s * 2] - 2 * X[s * 2 + 1] + 0.5);
  for (let it = 0; it < 500; it++) {
    const { out, cache } = forward(net, X, n);
    const d = Float64Array.from(out, (v, s) => (v - y[s]) / n);
    opt.step(backward(net, cache, d));
  }
  close(net.layers[0].W[0], 3, 0.02);
  close(net.layers[0].W[1], -2, 0.02);
  close(net.layers[0].b[0], 0.5, 0.02);
});

test('serialize round-trips a network', () => {
  const net = initNet(createMLP([4, 8, 2]), 'he', 'policy', new Rng(2));
  const copy = deserialize(JSON.parse(JSON.stringify(serialize(net))));
  const X = Float64Array.from([0.1, -0.2, 0.3, 0.05]);
  assert.deepEqual(forward(copy, X, 1).out, forward(net, X, 1).out);
});

for (const [rows, cols] of [[8, 8], [16, 4], [4, 16]]) {
  test(`orthogonal init is orthogonal (${rows}x${cols})`, () => {
    const g = 1.7;
    const W = orthogonal(rows, cols, g, new Rng(11));
    // Rows are orthonormal when wide, columns when tall.
    const short = Math.min(rows, cols);
    for (let a = 0; a < short; a++) {
      for (let b = 0; b < short; b++) {
        let dot = 0;
        if (rows >= cols) for (let r = 0; r < rows; r++) dot += W[r * cols + a] * W[r * cols + b];
        else for (let c = 0; c < cols; c++) dot += W[a * cols + c] * W[b * cols + c];
        close(dot, a === b ? g * g : 0, 1e-9);
      }
    }
  });
}

test('init schemes have the documented scales', () => {
  const rng = new Rng(4);
  const fanIn = 256;
  const variance = (id, role = 'hidden') => {
    const net = createMLP([fanIn, 256]);
    SCHEMES[id].init(net.layers[0], role, rng);
    const W = net.layers[0].W;
    return W.reduce((a, v) => a + v * v, 0) / W.length;
  };
  close(variance('he'), 2 / fanIn, 0.1 * (2 / fanIn), 'He');
  close(variance('lecun'), 1 / fanIn, 0.1 / fanIn, 'LeCun');
  close(variance('xavier'), 2 / (fanIn + 256), 0.1 * (2 / 512), 'Xavier');
  close(variance('pytorch'), 1 / (3 * fanIn), 0.1 / (3 * fanIn), 'PyTorch');
  close(variance('large'), 1, 0.05, 'Large');
  close(variance('orthoPPO', 'policy'), 0.01 ** 2 / fanIn, 1e-9, 'small policy head');
  assert.deepEqual([...SCHEME_IDS].sort(), Object.keys(SCHEMES).sort());
});

test('small-head init starts PPO with a near-uniform policy', () => {
  const rng = new Rng(1);
  const net = initNet(createMLP([4, 64, 64, 2]), 'orthoPPO', 'policy', rng);
  const big = initNet(createMLP([4, 64, 64, 2]), 'large', 'policy', rng);
  const obs = Float64Array.from([0.5, -1, 0.1, 1.5]);
  const gap = (n) => {
    const o = forward(n, obs, 1).out;
    return Math.abs(o[0] - o[1]);
  };
  assert.ok(gap(net) < 0.05, `small head logit gap ${gap(net)}`);
  assert.ok(gap(big) > 1, `large init logit gap ${gap(big)}`);
});

test('CartPole matches a Gym CartPole-v1 reference trajectory', () => {
  const ref = JSON.parse(readFileSync(new URL('./fixtures/cartpole-gym.json', import.meta.url)));
  const env = new CartPole(new Rng(1));
  env.reset();
  env.state.set(ref.start);
  ref.actions.forEach((a, i) => {
    const r = env.step(a);
    ref.states[i].forEach((v, j) => close(r.obs[j], v, 1e-6, `step ${i} dim ${j}`));
    assert.equal(r.terminated, ref.dones[i], `done at step ${i}`);
  });
});

test('CartPole truncates at 500 steps', () => {
  const env = new CartPole(new Rng(1));
  env.reset();
  let r;
  for (let i = 0; i < CARTPOLE.maxSteps; i++) {
    env.state.fill(0); // hold the pole upright so it never terminates
    r = env.step(i % 2);
  }
  assert.equal(r.truncated, true);
  assert.equal(r.terminated, false);
});

test('Jacobi eigenvalues of a known symmetric matrix', () => {
  // [[2,1],[1,2]] has eigenvalues 1 and 3
  const ev = Array.from(symmetricEigenvalues([2, 1, 1, 2], 2)).sort();
  close(ev[0], 1, 1e-9);
  close(ev[1], 3, 1e-9);
});

test('srank recovers the rank of a low-rank feature matrix', () => {
  const rng = new Rng(8);
  const n = 100;
  const d = 12;
  const k = 3;
  const A = Array.from({ length: n * k }, () => rng.normal());
  const B = Array.from({ length: k * d }, () => rng.normal());
  const H = new Float64Array(n * d);
  for (let s = 0; s < n; s++)
    for (let j = 0; j < d; j++) for (let r = 0; r < k; r++) H[s * d + j] += A[s * k + r] * B[r * d + j];
  assert.equal(srank(H, n, d, 0.001), k);
});

test('dormant and saturated fractions', () => {
  // 2 samples x 4 units: unit 3 is silent, unit 2 tiny.
  const h = Float64Array.from([1, 1, 0.01, 0, 1, -1, 0.01, 0]);
  assert.equal(dormantFraction([{ h, width: 4 }], 2, 0.1), 0.5);
  const t = Float64Array.from([0.995, -0.999, 0.2, 0.0]);
  assert.equal(saturatedFraction([{ h: t, width: 4 }], 1), 0.5);
});

test('signal propagation: He preserves ReLU signals, PyTorch default shrinks to its bias floor, N(0,1) explodes', () => {
  const opts = { activation: 'relu', depth: 20, width: 64, batch: 32 };
  const he = propagate('he', opts).actStd;
  const pt = propagate('pytorch', opts).actStd;
  const big = propagate('large', opts).actStd;
  assert.ok(he.at(-1) > 0.2 && he.at(-1) < 5, `He final std ${he.at(-1)}`);
  // Weight variance 1/(3 fan_in) shrinks the signal each layer until the
  // uniform biases dominate, so it plateaus well below He instead of hitting 0.
  assert.ok(pt.at(-1) < 0.15 * he.at(-1), `PyTorch final std ${pt.at(-1)} vs He ${he.at(-1)}`);
  assert.ok(big.at(-1) > 1e6, `N(0,1) final std ${big.at(-1)}`);
});
