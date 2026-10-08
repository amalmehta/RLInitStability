// A small multilayer perceptron with hand-written backprop and Adam.
// Hidden layers use the chosen activation; the last layer is linear.
// Batches are flat row-major Float64Arrays: X[n * dim + i].

export const ACTIVATIONS = {
  tanh: {
    f: (x) => Math.tanh(x),
    // derivative expressed through the output h = f(x)
    df: (h) => 1 - h * h,
  },
  relu: {
    f: (x) => (x > 0 ? x : 0),
    df: (h) => (h > 0 ? 1 : 0),
  },
};

export function createMLP(sizes, activation = 'tanh') {
  if (!ACTIVATIONS[activation]) throw new Error(`Unknown activation: ${activation}`);
  const layers = [];
  for (let k = 0; k < sizes.length - 1; k++) {
    const inDim = sizes[k];
    const outDim = sizes[k + 1];
    layers.push({
      inDim,
      outDim,
      W: new Float64Array(outDim * inDim),
      b: new Float64Array(outDim),
    });
  }
  return { sizes: sizes.slice(), activation, layers };
}

// Forward pass. Returns the output batch and a cache of each layer's input
// (post-activation of the previous layer) for backward().
export function forward(net, X, n) {
  const relu = net.activation === 'relu';
  const layers = net.layers;
  const last = layers.length - 1;
  const inputs = [];
  let h = X;
  for (let k = 0; k <= last; k++) {
    const L = layers[k];
    const W = L.W;
    const bias = L.b;
    const inDim = L.inDim;
    const outDim = L.outDim;
    const out = new Float64Array(n * outDim);
    inputs.push(h);
    for (let s = 0; s < n; s++) {
      const xo = s * inDim;
      const yo = s * outDim;
      for (let o = 0; o < outDim; o++) {
        let z = bias[o];
        const wo = o * inDim;
        for (let i = 0; i < inDim; i++) z += W[wo + i] * h[xo + i];
        if (k === last) out[yo + o] = z;
        else if (relu) out[yo + o] = z > 0 ? z : 0;
        else out[yo + o] = Math.tanh(z);
      }
    }
    h = out;
  }
  return { out: h, cache: { inputs, out: h, n } };
}

export function zeroGrads(net) {
  return net.layers.map((L) => ({
    W: new Float64Array(L.W.length),
    b: new Float64Array(L.b.length),
  }));
}

// Backward pass: dOut is dLoss/dOutput for the batch. Accumulates into grads.
export function backward(net, cache, dOut, grads = zeroGrads(net)) {
  const relu = net.activation === 'relu';
  const { inputs, n } = cache;
  let delta = dOut; // gradient w.r.t. this layer's pre-activation output
  for (let k = net.layers.length - 1; k >= 0; k--) {
    const L = net.layers[k];
    const W = L.W;
    const inDim = L.inDim;
    const outDim = L.outDim;
    const x = inputs[k];
    const gW = grads[k].W;
    const gb = grads[k].b;
    for (let s = 0; s < n; s++) {
      const xo = s * inDim;
      const yo = s * outDim;
      for (let o = 0; o < outDim; o++) {
        const d = delta[yo + o];
        if (d === 0) continue;
        gb[o] += d;
        const wo = o * inDim;
        for (let i = 0; i < inDim; i++) gW[wo + i] += d * x[xo + i];
      }
    }
    if (k === 0) break;
    // x is the previous layer's activation output: chain through f'(x).
    const dx = new Float64Array(n * inDim);
    for (let s = 0; s < n; s++) {
      const xo = s * inDim;
      const yo = s * outDim;
      for (let o = 0; o < outDim; o++) {
        const d = delta[yo + o];
        if (d === 0) continue;
        const wo = o * inDim;
        for (let i = 0; i < inDim; i++) dx[xo + i] += d * W[wo + i];
      }
      for (let i = 0; i < inDim; i++) {
        const hv = x[xo + i];
        dx[xo + i] *= relu ? (hv > 0 ? 1 : 0) : 1 - hv * hv;
      }
    }
    delta = dx;
  }
  return grads;
}

export function gradNorm(grads) {
  let s = 0;
  for (const g of grads) {
    for (const v of g.W) s += v * v;
    for (const v of g.b) s += v * v;
  }
  return Math.sqrt(s);
}

export function scaleGrads(grads, c) {
  for (const g of grads) {
    for (let j = 0; j < g.W.length; j++) g.W[j] *= c;
    for (let j = 0; j < g.b.length; j++) g.b[j] *= c;
  }
}

export function weightNorm(net) {
  let s = 0;
  for (const L of net.layers) {
    for (const v of L.W) s += v * v;
    for (const v of L.b) s += v * v;
  }
  return Math.sqrt(s);
}

// Hidden-layer activations for a batch: one Float64Array (n x width) per
// hidden layer. Used for dormant-neuron and feature-rank metrics.
export function hiddenActivations(net, X, n) {
  const { cache } = forward(net, X, n);
  return cache.inputs.slice(1).map((h, k) => ({ h, width: net.layers[k].outDim }));
}

export function serialize(net) {
  return {
    sizes: net.sizes,
    activation: net.activation,
    layers: net.layers.map((L) => ({ W: Array.from(L.W), b: Array.from(L.b) })),
  };
}

export function deserialize(data) {
  const net = createMLP(data.sizes, data.activation);
  data.layers.forEach((l, k) => {
    net.layers[k].W.set(l.W);
    net.layers[k].b.set(l.b);
  });
  return net;
}

export class Adam {
  constructor(net, { lr = 3e-4, beta1 = 0.9, beta2 = 0.999, eps = 1e-5 } = {}) {
    this.net = net;
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.t = 0;
    this.m = zeroGrads(net);
    this.v = zeroGrads(net);
  }

  step(grads) {
    this.t += 1;
    const { beta1, beta2, eps } = this;
    const c1 = 1 - Math.pow(beta1, this.t);
    const c2 = 1 - Math.pow(beta2, this.t);
    const lr = this.lr;
    this.net.layers.forEach((L, k) => {
      for (const key of ['W', 'b']) {
        const p = L[key];
        const g = grads[k][key];
        const m = this.m[k][key];
        const v = this.v[k][key];
        for (let j = 0; j < p.length; j++) {
          m[j] = beta1 * m[j] + (1 - beta1) * g[j];
          v[j] = beta2 * v[j] + (1 - beta2) * g[j] * g[j];
          p[j] -= (lr * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + eps);
        }
      }
    });
  }
}
