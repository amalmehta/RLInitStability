// Proximal Policy Optimization (Schulman et al. 2017) for CartPole, with
// separate actor and critic MLPs, GAE (Schulman et al. 2016) and the usual
// implementation details (Huang et al. 2022). One call to update() collects a
// rollout, measures network health, then optimizes.

import { Rng } from './rng.js';
import { CartPole } from './cartpole.js';
import { createMLP, forward, backward, zeroGrads, gradNorm, scaleGrads, weightNorm, Adam, serialize } from './nn.js';
import { initNet } from './init.js';
import { dormantFraction, saturatedFraction, srank } from './metrics.js';

export const DEFAULT_CONFIG = {
  seed: 1,
  scheme: 'orthoPPO',
  activation: 'tanh',
  hidden: [64, 64],
  lr: 1e-3,
  totalSteps: 50000,
  rolloutSteps: 512,
  epochs: 4,
  minibatch: 128,
  gamma: 0.99,
  lambda: 0.95,
  clip: 0.2,
  entCoef: 0.0,
  maxGradNorm: 0.5, // null disables clipping
  normAdv: true,
  dormantTau: 0.1,
};

const OBS = 4;
const ACTIONS = 2;

function softmax2(z0, z1) {
  const m = Math.max(z0, z1);
  const e0 = Math.exp(z0 - m);
  const e1 = Math.exp(z1 - m);
  const s = e0 + e1;
  return [e0 / s, e1 / s];
}

export class PPO {
  constructor(config = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...config };
    const c = this.cfg;
    this.rng = new Rng(c.seed * 7919 + 17);
    const initRng = new Rng(c.seed * 104729 + 3);
    this.actor = initNet(createMLP([OBS, ...c.hidden, ACTIONS], c.activation), c.scheme, 'policy', initRng);
    this.critic = initNet(createMLP([OBS, ...c.hidden, 1], c.activation), c.scheme, 'value', initRng);
    this.actorOpt = new Adam(this.actor, { lr: c.lr });
    this.criticOpt = new Adam(this.critic, { lr: c.lr });
    this.env = new CartPole(new Rng(c.seed * 31337 + 5));
    this.obs = this.env.reset();
    this.epReturn = 0;
    this.steps = 0;
    this.lastMeanReturn = 0;
    this.diverged = false;
  }

  get done() {
    return this.steps >= this.cfg.totalSteps || this.diverged;
  }

  policy(obs) {
    const { out } = forward(this.actor, obs, 1);
    return softmax2(out[0], out[1]);
  }

  value(obs) {
    return forward(this.critic, obs, 1).out[0];
  }

  collect() {
    const T = this.cfg.rolloutSteps;
    const obs = new Float64Array(T * OBS);
    const actions = new Int32Array(T);
    const logp = new Float64Array(T);
    const rewards = new Float64Array(T);
    const dones = new Uint8Array(T);
    const values = new Float64Array(T);
    const returns = [];
    for (let t = 0; t < T; t++) {
      obs.set(this.obs, t * OBS);
      const p = this.policy(this.obs);
      const a = this.rng.next() < p[0] ? 0 : 1;
      actions[t] = a;
      logp[t] = Math.log(Math.max(p[a], 1e-12));
      values[t] = this.value(this.obs);
      const r = this.env.step(a);
      let reward = r.reward;
      this.epReturn += r.reward;
      // Time-limit truncation is not a real terminal: bootstrap from V(s').
      if (r.truncated) reward += this.cfg.gamma * this.value(r.obs);
      rewards[t] = reward;
      if (r.terminated || r.truncated) {
        dones[t] = 1;
        returns.push(this.epReturn);
        this.epReturn = 0;
        this.obs = this.env.reset();
      } else {
        this.obs = r.obs;
      }
    }
    const lastValue = this.value(this.obs);
    // Generalized advantage estimation.
    const adv = new Float64Array(T);
    let gae = 0;
    for (let t = T - 1; t >= 0; t--) {
      const nextV = t === T - 1 ? lastValue : values[t + 1];
      const notDone = dones[t] ? 0 : 1;
      const delta = rewards[t] + this.cfg.gamma * nextV * notDone - values[t];
      gae = delta + this.cfg.gamma * this.cfg.lambda * notDone * gae;
      adv[t] = gae;
    }
    const ret = new Float64Array(T);
    for (let t = 0; t < T; t++) ret[t] = adv[t] + values[t];
    this.steps += T;
    return { obs, actions, logp, adv, ret, values, episodeReturns: returns, T };
  }

  measure(batch) {
    const n = Math.min(256, batch.T);
    const X = batch.obs.subarray(0, n * OBS);
    const { cache } = forward(this.actor, X, n);
    const hidden = cache.inputs.slice(1).map((h, k) => ({ h, width: this.actor.layers[k].outDim }));
    const last = hidden[hidden.length - 1];
    return {
      dormant: dormantFraction(hidden, n, this.cfg.dormantTau),
      saturated: this.cfg.activation === 'tanh' ? saturatedFraction(hidden, n) : null,
      srank: srank(last.h, n, last.width),
      weightNorm: weightNorm(this.actor),
    };
  }

  optimize(batch) {
    const c = this.cfg;
    const { T } = batch;
    const adv = Float64Array.from(batch.adv);
    if (c.normAdv) {
      let mean = 0;
      for (const v of adv) mean += v;
      mean /= T;
      let varr = 0;
      for (const v of adv) varr += (v - mean) ** 2;
      const std = Math.sqrt(varr / T) + 1e-8;
      for (let t = 0; t < T; t++) adv[t] = (adv[t] - mean) / std;
    }
    const idx = Array.from({ length: T }, (_, i) => i);
    const mb = c.minibatch;
    let gnSum = 0;
    let gnCount = 0;
    let entSum = 0;
    let klSum = 0;
    let clipSum = 0;
    let vlossSum = 0;
    let seen = 0;
    for (let epoch = 0; epoch < c.epochs; epoch++) {
      this.rng.shuffle(idx);
      for (let start = 0; start + mb <= T; start += mb) {
        const X = new Float64Array(mb * OBS);
        for (let j = 0; j < mb; j++) X.set(batch.obs.subarray(idx[start + j] * OBS, idx[start + j] * OBS + OBS), j * OBS);

        // Actor: clipped surrogate + entropy bonus.
        const fa = forward(this.actor, X, mb);
        const dLogits = new Float64Array(mb * ACTIONS);
        for (let j = 0; j < mb; j++) {
          const t = idx[start + j];
          const z0 = fa.out[j * 2];
          const z1 = fa.out[j * 2 + 1];
          if (!Number.isFinite(z0) || !Number.isFinite(z1)) {
            this.diverged = true;
            return null;
          }
          const p = softmax2(z0, z1);
          const a = batch.actions[t];
          const lp = Math.log(Math.max(p[a], 1e-12));
          const ratio = Math.exp(lp - batch.logp[t]);
          const A = adv[t];
          const clipped = (A > 0 && ratio > 1 + c.clip) || (A < 0 && ratio < 1 - c.clip);
          const H = -(p[0] * Math.log(Math.max(p[0], 1e-12)) + p[1] * Math.log(Math.max(p[1], 1e-12)));
          entSum += H;
          klSum += ratio - 1 - Math.log(ratio); // low-variance KL estimator
          if (Math.abs(ratio - 1) > c.clip) clipSum += 1;
          seen += 1;
          const dLp = clipped ? 0 : -ratio * A; // d(-min(...))/d logp
          for (let k = 0; k < 2; k++) {
            const dlogp = (k === a ? 1 : 0) - p[k];
            const dH = -p[k] * (Math.log(Math.max(p[k], 1e-12)) + H);
            dLogits[j * 2 + k] = (dLp * dlogp - c.entCoef * dH) / mb;
          }
        }
        const ga = backward(this.actor, fa.cache, dLogits, zeroGrads(this.actor));
        const gn = gradNorm(ga);
        gnSum += gn;
        gnCount += 1;
        if (c.maxGradNorm && gn > c.maxGradNorm) scaleGrads(ga, c.maxGradNorm / gn);
        this.actorOpt.step(ga);

        // Critic: squared error to the GAE return.
        const fc = forward(this.critic, X, mb);
        const dV = new Float64Array(mb);
        for (let j = 0; j < mb; j++) {
          const t = idx[start + j];
          const err = fc.out[j] - batch.ret[t];
          vlossSum += 0.5 * err * err;
          dV[j] = err / mb;
        }
        const gc = backward(this.critic, fc.cache, dV, zeroGrads(this.critic));
        const gcn = gradNorm(gc);
        if (c.maxGradNorm && gcn > c.maxGradNorm) scaleGrads(gc, c.maxGradNorm / gcn);
        this.criticOpt.step(gc);
      }
    }
    if (!Number.isFinite(gnSum)) this.diverged = true;
    return {
      gradNorm: gnSum / Math.max(gnCount, 1),
      entropy: entSum / Math.max(seen, 1),
      approxKL: klSum / Math.max(seen, 1),
      clipFrac: clipSum / Math.max(seen, 1),
      valueLoss: vlossSum / Math.max(seen, 1),
    };
  }

  // One PPO iteration. Returns a metrics record for this point in training.
  update() {
    const batch = this.collect();
    const health = this.measure(batch);
    const opt = this.optimize(batch);
    if (batch.episodeReturns.length) {
      this.lastMeanReturn = batch.episodeReturns.reduce((a, b) => a + b, 0) / batch.episodeReturns.length;
    }
    return {
      step: this.steps,
      return: this.lastMeanReturn,
      episodes: batch.episodeReturns.length,
      ...health,
      ...(opt || { gradNorm: NaN, entropy: NaN, approxKL: NaN, clipFrac: NaN, valueLoss: NaN }),
      diverged: this.diverged,
    };
  }

  snapshot() {
    return serialize(this.actor);
  }
}

// Synchronous helper for tests and scripts.
export function runPPO(config, onUpdate = () => {}) {
  const agent = new PPO(config);
  const history = [];
  while (!agent.done) {
    const m = agent.update();
    history.push(m);
    onUpdate(m, agent);
  }
  return { agent, history };
}
