# File Structure

```
.
├── README.md                     What it is, screenshots, links
├── package.json                  npm scripts: start, test, literature (no dependencies)
├── rl_initialization_training_stability.md   Project brief, decisions and changelog
├── site/                         The website, served as static files
│   ├── index.html                Overview and key findings
│   ├── literature.html           Literature review page
│   ├── demo.html                 Live lab: PPO training and signal propagation
│   ├── css/style.css             All styles, with light and dark tokens
│   ├── data/papers.json          The literature dataset (single source of truth)
│   └── js/
│       ├── site.js               Theme toggle shared by every page
│       ├── literature.js         Timeline, search, filters and paper cards
│       ├── demo.js               Lab controller: worker pool, charts, table, viewer, signal explorer
│       ├── worker.js             Runs one PPO training run in a Web Worker
│       ├── charts.js             SVG line chart with band and tooltip
│       ├── aggregate.js          Seed aggregation and summary rows
│       └── core/                 Shared by the browser and the Node tests
│           ├── rng.js            Seeded random numbers
│           ├── nn.js             MLP, backprop, Adam
│           ├── init.js           The seven init schemes and orthogonal init
│           ├── cartpole.js       CartPole-v1 environment
│           ├── ppo.js            PPO agent and runPPO helper
│           ├── metrics.js        Dormant neurons, saturation, srank
│           └── signal.js         Signal propagation through deep untrained nets
├── tests/
│   ├── core.test.mjs             Unit tests for core/
│   ├── ppo.test.mjs              PPO integration tests and aggregation
│   ├── site.test.mjs             Literature data and page consistency
│   └── fixtures/cartpole-gym.json  Reference trajectory recorded from Gym
├── scripts/
│   ├── build-literature.mjs      papers.json → docs/LITERATURE.md
│   └── make-cartpole-fixture.py  Records the Gym reference trajectory
└── docs/
    ├── INSTRUCTIONS.md           Setup, run, use, test
    ├── SYSTEM-DESIGN.md          Architecture, flows, decisions, limits
    ├── FILE-STRUCTURE.md         This file
    ├── LITERATURE.md             Generated readable copy of the review
    └── images/                   Screenshots used by the README
```
