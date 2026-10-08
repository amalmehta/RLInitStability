// Seeded random numbers so every run is reproducible from its seed.

export class Rng {
  constructor(seed = 1) {
    this.state = (seed >>> 0) || 1;
    this.spare = null;
  }

  // mulberry32: fast 32-bit generator, uniform in [0, 1).
  next() {
    let t = (this.state = (this.state + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  uniform(lo, hi) {
    return lo + (hi - lo) * this.next();
  }

  // Standard normal via Box-Muller, caching the second sample.
  normal() {
    if (this.spare !== null) {
      const s = this.spare;
      this.spare = null;
      return s;
    }
    let u = 0;
    while (u === 0) u = this.next();
    const v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }

  int(n) {
    return Math.floor(this.next() * n);
  }

  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}
