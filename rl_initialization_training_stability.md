PROJECT NAME: rl_initialization_training_stability

META-INSTRUCTIONS:

<Read it all before acting. Ask about anything unclear, contradictory or
 underspecified — before starting and mid-build. Ask in the question widget
 (AskUserQuestion): related questions batched, concrete options, your
 recommendation first. Plain text only if the widget isn't available.>

<Don't expand scope. Anything not listed here is a proposal, including changes
 to this file — propose it, don't do it.>

<Prefer doing over describing: run the code, write the files, test it.>

<Always in scope, no proposal needed: when it goes on GitHub, a README that is
 easy to read at a glance — a line on what it is, then clear visuals
 (screenshots, a diagram or a chart), then links. Everything else goes in
 linked files: docs/INSTRUCTIONS.md (setup, run, use),
 docs/SYSTEM-DESIGN.md (see below) and docs/FILE-STRUCTURE.md (what's where). If what you're
 building is an application rather than a script, also a small unobtrusive feedback tab, and a settings
 button or tab (⌘, on the Mac) that gathers its preferences in one place.>

<If what you're building is an application, build it as a Mac app first; the
 website comes after, as its own step.>

<Name things the way a person would say them — "Goal Tracker", not
 goal_tracker — for the app, its windows, titles, files people open, repo
 descriptions and README headings. When you create the GitHub repo, name it
 with no "_" or "-": one word or joined words, e.g. GoalTracker.>

<Always in scope: a system design doc in the codebase, docs/SYSTEM-DESIGN.md,
 kept current as the build changes. Cover the architecture (with a Mermaid
 diagram), each component's job, the main flows, where data lives, the key
 design decisions and their trade-offs, how it's tested, and known limits.>

<Finish by listing every deliverable: path, what it is, how to check it works.>

<Git rules (no Claude attribution, never commit .claude/) are in
 ~/.claude/CLAUDE.md and apply on their own — nothing to repeat here.>

<Keep the changelog at the bottom current.>

CONTEXT:

create a full project on training stability/rl vizualization that covers a full literature sweep + demo

DELIVERABLES:

website

OPEN QUESTIONS / ASSUMPTIONS:

<Agent fills in: what it guessed, what it decided without asking.>

Asked and answered (2026-10-08):
- Platform: website only, with no Mac app step. It is treated as an educational site, not an application, so there is no feedback tab or settings panel.
- Literature scope: RL initialization and stability. The result is 68 papers in 7 themes, with titles, authors and years verified against the arXiv API.
- Demo: live in-browser RL. PPO on CartPole in plain JavaScript Web Workers, comparing 7 init schemes across seeds.
- GitHub: local only for now (git init plus a commit). Creating the repo and deploying are proposals below.

Decided without asking:
- PPO defaults are cheaper than CleanRL's (512-step rollouts, 4 epochs, minibatch 128, lr 1e-3, 50k steps) so a comparison takes about a minute. The 10-epoch, 100k-step version took about 150 s per run.
- Added a signal-propagation explorer to the demo page. It visualizes the init theory directly and costs under a second to compute.
- CartPole is easy, so init effects show mainly in health metrics and early learning. The site says so, and the claims match measured runs: N(0,1) gives entropy 0.21 at step 0, 18% saturated tanh units and weight norm 68; ReLU with N(0,1) starts about 50% dormant and lags early. A 6-layer tanh net did not separate PyTorch-default from orthogonal init, so the site doesn't claim depth does.
- The site is served at port 8765 (`npm start`), because 8000 was already taken on this machine.

Proposals (not done):
- Create a GitHub repo named RLInitStability and deploy the site with GitHub Pages.
- A harder environment (Acrobot or LunarLander-like) or a non-stationary task, to show init and plasticity effects in return as well.

CHANGELOG:

- 2026-10-08 — created
- 2026-09-15 — added meta-instruction: built-out applications include a small feedback tab
- 2026-09-15 — added meta-instruction: no "Claude" attribution in commits, PRs, or branches
- 2026-09-16 — added meta-instruction: always include a README when adding to GitHub
- 2026-09-16 — changed meta-instruction: ask clarifying questions in the question widget
- 2026-09-17 — added meta-instructions: Claude never a contributor; never commit .claude/
- 2026-09-26 — compressed the meta-instructions and every field prompt; git rules moved to the global instruction file
- 2026-09-27 — added meta-instruction: applications are built as a Mac app first, then a website
- 2026-09-28 — folded inputs, instructions, constraints, deliverables and done criteria into one free-form CONTEXT
- 2026-09-28 — changed meta-instruction: a README on GitHub always includes a visual
- 2026-09-28 — added meta-instruction: name things like a person would, never snake_case
- 2026-09-28 — changed meta-instruction: README leads with visuals; instructions live in a linked guide
- 2026-09-28 — changed meta-instruction: README is visuals and links; details in docs/INSTRUCTIONS.md and docs/FILE-STRUCTURE.md
- 2026-09-29 — changed meta-instruction: GitHub repo names have no "_" or "-"
- 2026-10-02 — added a DELIVERABLES field after CONTEXT
- 2026-10-02 — added meta-instruction: every project has a system design doc at docs/SYSTEM-DESIGN.md
- 2026-10-05 — added meta-instruction: applications include a settings button or tab
- 2026-10-08 — built the website: literature review (68 papers), live PPO lab, signal-propagation explorer, tests and docs
