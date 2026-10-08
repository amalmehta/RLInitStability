// Network-health metrics used in the stability literature.

// Eigenvalues of a symmetric d x d matrix (flat row-major) by cyclic Jacobi.
export function symmetricEigenvalues(A, d, sweeps = 30, tol = 1e-12) {
  const a = Float64Array.from(A);
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) off += a[p * d + q] ** 2;
    if (off < tol) break;
    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) {
        const apq = a[p * d + q];
        if (Math.abs(apq) < 1e-300) continue;
        const app = a[p * d + p];
        const aqq = a[q * d + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < d; k++) {
          const akp = a[k * d + p];
          const akq = a[k * d + q];
          a[k * d + p] = c * akp - s * akq;
          a[k * d + q] = s * akp + c * akq;
        }
        for (let k = 0; k < d; k++) {
          const apk = a[p * d + k];
          const aqk = a[q * d + k];
          a[p * d + k] = c * apk - s * aqk;
          a[q * d + k] = s * apk + c * aqk;
        }
      }
    }
  }
  const ev = new Float64Array(d);
  for (let i = 0; i < d; i++) ev[i] = a[i * d + i];
  return ev;
}

// Singular values of an n x d matrix, largest first.
export function singularValues(H, n, d) {
  const G = new Float64Array(d * d);
  for (let s = 0; s < n; s++) {
    const o = s * d;
    for (let i = 0; i < d; i++) {
      const hi = H[o + i];
      if (hi === 0) continue;
      for (let j = i; j < d; j++) G[i * d + j] += hi * H[o + j];
    }
  }
  for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) G[i * d + j] = G[j * d + i];
  return Array.from(symmetricEigenvalues(G, d))
    .map((l) => Math.sqrt(Math.max(l, 0)))
    .sort((x, y) => y - x);
}

// Effective feature rank, srank_delta (Kumar et al. 2021, "Implicit
// Under-Parameterization"): the fewest singular values that hold
// (1 - delta) of the total singular-value mass.
export function srank(H, n, d, delta = 0.01) {
  const sv = singularValues(H, n, d);
  const total = sv.reduce((a, b) => a + b, 0);
  if (total === 0) return 0;
  let acc = 0;
  for (let k = 0; k < sv.length; k++) {
    acc += sv[k];
    if (acc >= (1 - delta) * total) return k + 1;
  }
  return sv.length;
}

// Fraction of tau-dormant neurons (Sokar et al. 2023): neuron i is dormant
// when its mean |activation| divided by the layer average is <= tau.
// Pools all hidden layers given as [{ h, width }].
export function dormantFraction(layers, n, tau = 0.1) {
  let dormant = 0;
  let total = 0;
  for (const { h, width } of layers) {
    const score = new Float64Array(width);
    for (let s = 0; s < n; s++) for (let i = 0; i < width; i++) score[i] += Math.abs(h[s * width + i]);
    let mean = 0;
    for (let i = 0; i < width; i++) mean += score[i];
    mean /= width;
    for (let i = 0; i < width; i++) {
      if (mean === 0 || score[i] / mean <= tau) dormant += 1;
    }
    total += width;
  }
  return total ? dormant / total : 0;
}

// Share of tanh activations pinned near ±1, where the gradient vanishes.
export function saturatedFraction(layers, n, threshold = 0.99) {
  let sat = 0;
  let total = 0;
  for (const { h } of layers) {
    for (let j = 0; j < h.length; j++) if (Math.abs(h[j]) > threshold) sat += 1;
    total += h.length;
  }
  return total ? sat / total : 0;
}
