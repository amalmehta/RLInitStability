// Signal propagation through a deep random network at initialization:
// per-layer activation spread on the way forward, and per-layer weight
// gradient norm on the way back (Poole et al. 2016; Schoenholz et al. 2017).

import { Rng } from './rng.js';
import { createMLP, forward, backward } from './nn.js';
import { initNet } from './init.js';

function std(a) {
  let m = 0;
  for (const v of a) m += v;
  m /= a.length;
  let s = 0;
  for (const v of a) s += (v - m) ** 2;
  return Math.sqrt(s / a.length);
}

export function propagate(schemeId, { activation = 'tanh', depth = 10, width = 128, batch = 64, seed = 1 } = {}) {
  const rng = new Rng(seed);
  const sizes = Array(depth + 1).fill(width);
  const net = initNet(createMLP(sizes, activation), schemeId, 'hidden', rng);
  const X = new Float64Array(batch * width);
  for (let j = 0; j < X.length; j++) X[j] = rng.normal();
  const { out, cache } = forward(net, X, batch);
  // Layer k's output is the input cached for layer k + 1; the last is `out`.
  const outputs = [...cache.inputs.slice(1), out];
  const actStd = outputs.map(std);
  const dOut = new Float64Array(out.length);
  for (let j = 0; j < dOut.length; j++) dOut[j] = rng.normal() / batch;
  const grads = backward(net, cache, dOut);
  const gradNorm = grads.map((g) => {
    let s = 0;
    for (const v of g.W) s += v * v;
    return Math.sqrt(s);
  });
  return { actStd, gradNorm };
}
