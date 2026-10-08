// Weight initialization schemes. Each layer is initialized according to its
// role: "hidden" (feature layers), "policy" (actor output) or "value" (critic
// output). Weights are row-major: W[o * fanIn + i].

const SQRT2 = Math.SQRT2;

function fill(W, f) {
  for (let k = 0; k < W.length; k++) W[k] = f();
  return W;
}

// Orthogonal matrix (rows x cols) via modified Gram-Schmidt on a Gaussian
// matrix, with the sign fix that makes the result uniform over the
// orthogonal group (as in Saxe et al. 2014 and torch.nn.init.orthogonal_).
export function orthogonal(rows, cols, gain, rng) {
  const tall = rows >= cols;
  const n = tall ? rows : cols; // long side
  const m = tall ? cols : rows; // short side
  // m column vectors of length n
  const Q = [];
  for (let j = 0; j < m; j++) {
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = rng.normal();
    Q.push(v);
  }
  for (let j = 0; j < m; j++) {
    const v = Q[j];
    for (let p = 0; p < j; p++) {
      const q = Q[p];
      let dot = 0;
      for (let i = 0; i < n; i++) dot += v[i] * q[i];
      for (let i = 0; i < n; i++) v[i] -= dot * q[i];
    }
    let norm = 0;
    for (let i = 0; i < n; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    // Gram-Schmidt's R diagonal is positive by construction, which already
    // matches the sign-corrected QR, so no further flip is needed.
    for (let i = 0; i < n; i++) v[i] /= norm;
  }
  const W = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      W[r * cols + c] = gain * (tall ? Q[c][r] : Q[r][c]);
    }
  }
  return W;
}

export const SCHEMES = {
  pytorch: {
    label: 'PyTorch default',
    note: 'Kaiming-uniform with a = sqrt(5): U(±1/sqrt(fan_in)) for weights and biases.',
    init(layer, role, rng) {
      const b = 1 / Math.sqrt(layer.inDim);
      fill(layer.W, () => rng.uniform(-b, b));
      fill(layer.b, () => rng.uniform(-b, b));
    },
  },
  xavier: {
    label: 'Xavier (Glorot) uniform',
    note: 'U(±sqrt(6/(fan_in+fan_out))), zero bias. Glorot & Bengio 2010.',
    init(layer, role, rng) {
      const b = Math.sqrt(6 / (layer.inDim + layer.outDim));
      fill(layer.W, () => rng.uniform(-b, b));
      layer.b.fill(0);
    },
  },
  he: {
    label: 'He (Kaiming) normal',
    note: 'N(0, 2/fan_in), zero bias. He et al. 2015.',
    init(layer, role, rng) {
      const s = Math.sqrt(2 / layer.inDim);
      fill(layer.W, () => s * rng.normal());
      layer.b.fill(0);
    },
  },
  lecun: {
    label: 'LeCun normal',
    note: 'N(0, 1/fan_in), zero bias.',
    init(layer, role, rng) {
      const s = Math.sqrt(1 / layer.inDim);
      fill(layer.W, () => s * rng.normal());
      layer.b.fill(0);
    },
  },
  orthoPPO: {
    label: 'Orthogonal + small heads',
    note: 'The common PPO recipe: orthogonal gain sqrt(2) hidden, 0.01 policy head, 1.0 value head, zero bias (Andrychowicz et al. 2021; Huang et al. 2022).',
    init(layer, role, rng) {
      const gain = role === 'policy' ? 0.01 : role === 'value' ? 1.0 : SQRT2;
      layer.W.set(orthogonal(layer.outDim, layer.inDim, gain, rng));
      layer.b.fill(0);
    },
  },
  orthoPlain: {
    label: 'Orthogonal, no small heads',
    note: 'Orthogonal gain sqrt(2) everywhere, including both output heads: an ablation of the small policy head.',
    init(layer, role, rng) {
      layer.W.set(orthogonal(layer.outDim, layer.inDim, SQRT2, rng));
      layer.b.fill(0);
    },
  },
  large: {
    label: 'Large normal N(0, 1)',
    note: 'Unit-variance weights regardless of fan-in: activations explode or saturate. A deliberately bad init.',
    init(layer, role, rng) {
      fill(layer.W, () => rng.normal());
      layer.b.fill(0);
    },
  },
};

// Fixed display order. A scheme's position sets its chart color, so colors
// follow the scheme, not how many are selected.
export const SCHEME_IDS = ['orthoPPO', 'pytorch', 'orthoPlain', 'large', 'xavier', 'he', 'lecun'];

// Initialize every layer of an MLP; the last layer takes the given head role.
export function initNet(net, schemeId, headRole, rng) {
  const scheme = SCHEMES[schemeId];
  if (!scheme) throw new Error(`Unknown init scheme: ${schemeId}`);
  net.layers.forEach((layer, k) => {
    const role = k === net.layers.length - 1 ? headRole : 'hidden';
    scheme.init(layer, role, rng);
  });
  return net;
}
