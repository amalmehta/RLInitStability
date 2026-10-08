"""Record a reference CartPole-v1 trajectory from OpenAI Gym for tests/cartpole.test.mjs.

Usage: python3 scripts/make-cartpole-fixture.py  (needs `pip install gym==0.21`)
"""
import json
import gym

env = gym.make("CartPole-v1").unwrapped
env.reset()
start = [0.01, -0.02, 0.03, 0.04]
env.state = list(start)
actions = [(i * 7 + i // 3) % 2 for i in range(60)]
states, dones = [], []
for a in actions:
    obs, r, done, _ = env.step(a)
    states.append([float(v) for v in obs])
    dones.append(bool(done))
    if done:
        break
json.dump({"source": f"gym {gym.__version__} CartPole-v1", "start": start, "actions": actions[: len(states)],
           "states": states, "dones": dones}, open("tests/fixtures/cartpole-gym.json", "w"), indent=1)
print(len(states), "steps, terminated:", dones[-1])
