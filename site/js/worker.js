// Runs one PPO training run off the main thread and streams metrics back.
// Message in:   { type: 'start', runId, config }
// Messages out: { type: 'metrics', runId, m } | { type: 'snapshot', runId, step, net }
//               { type: 'done', runId, diverged } | { type: 'error', runId, message }
// The page stops a run by terminating the worker, so the training loop never
// yields: yielding through timers or message ports gets throttled to seconds
// per tick when the page sits in a background tab.

import { PPO } from './core/ppo.js';

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type !== 'start') return;
  const { runId, config } = msg;
  try {
    const agent = new PPO(config);
    let k = 0;
    while (!agent.done) {
      const m = agent.update();
      self.postMessage({ type: 'metrics', runId, m });
      if (k++ % 3 === 0 || agent.done) {
        self.postMessage({ type: 'snapshot', runId, step: m.step, net: agent.snapshot() });
      }
    }
    self.postMessage({ type: 'done', runId, diverged: agent.diverged });
  } catch (err) {
    self.postMessage({ type: 'error', runId, message: String(err && err.message ? err.message : err) });
  }
};
