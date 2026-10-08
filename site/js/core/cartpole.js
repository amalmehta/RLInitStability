// CartPole-v1, ported from Gymnasium's classic_control/cartpole.py
// (Barto, Sutton & Anderson 1983 dynamics, Euler integration).

export const CARTPOLE = {
  gravity: 9.8,
  massCart: 1.0,
  massPole: 0.1,
  length: 0.5, // half the pole's length
  forceMag: 10.0,
  tau: 0.02,
  thetaThreshold: (12 * 2 * Math.PI) / 360,
  xThreshold: 2.4,
  maxSteps: 500,
};

export class CartPole {
  constructor(rng) {
    this.rng = rng;
    this.state = new Float64Array(4);
    this.steps = 0;
  }

  reset() {
    for (let i = 0; i < 4; i++) this.state[i] = this.rng.uniform(-0.05, 0.05);
    this.steps = 0;
    return Float64Array.from(this.state);
  }

  // action: 0 = push left, 1 = push right.
  // Returns { obs, reward, terminated, truncated }.
  step(action) {
    const c = CARTPOLE;
    let [x, xDot, theta, thetaDot] = this.state;
    const force = action === 1 ? c.forceMag : -c.forceMag;
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const totalMass = c.massCart + c.massPole;
    const poleMassLength = c.massPole * c.length;
    const temp = (force + poleMassLength * thetaDot * thetaDot * sin) / totalMass;
    const thetaAcc =
      (c.gravity * sin - cos * temp) /
      (c.length * (4 / 3 - (c.massPole * cos * cos) / totalMass));
    const xAcc = temp - (poleMassLength * thetaAcc * cos) / totalMass;
    x += c.tau * xDot;
    xDot += c.tau * xAcc;
    theta += c.tau * thetaDot;
    thetaDot += c.tau * thetaAcc;
    this.state[0] = x;
    this.state[1] = xDot;
    this.state[2] = theta;
    this.state[3] = thetaDot;
    this.steps += 1;
    const terminated =
      x < -c.xThreshold || x > c.xThreshold || theta < -c.thetaThreshold || theta > c.thetaThreshold;
    const truncated = !terminated && this.steps >= c.maxSteps;
    return { obs: Float64Array.from(this.state), reward: 1, terminated, truncated };
  }
}
