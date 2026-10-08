# Literature Review: Initialization and Training Stability in Deep RL

68 papers in 7 themes. Titles, authors and years come from the arXiv API; summaries are this project's own.
The same data powers the searchable [Literature page](../site/literature.html) of the website.

_Generated from `site/data/papers.json` by `npm run literature`. Edit the JSON, not this file._

## Themes

- [Initialization and signal propagation](#initialization-and-signal-propagation) (10)
- [Policy-gradient implementation details](#policy-gradient-implementation-details) (9)
- [Value learning, divergence and the deadly triad](#value-learning-divergence-and-the-deadly-triad) (12)
- [Plasticity loss and dormant neurons](#plasticity-loss-and-dormant-neurons) (10)
- [Resets and plasticity interventions](#resets-and-plasticity-interventions) (11)
- [Normalization and architectures for stable scaling](#normalization-and-architectures-for-stable-scaling) (14)
- [Evaluation and reproducibility](#evaluation-and-reproducibility) (2)

## Initialization and signal propagation

How the scale and structure of initial weights decide whether signals and gradients survive a forward and backward pass. This is the general deep-learning theory that RL initialization recipes build on.

### [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)

Xavier Glorot, Yoshua Bengio · AISTATS 2010

Shows that with standard random init, activations and gradients shrink or saturate with depth (especially with sigmoid), and derives a fan-in/fan-out-balanced variance: the "Xavier" init.

**Why it matters:** Weight variance should depend on layer width. This is the baseline idea behind every init scheme in the demo.

### [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120)

Andrew M. Saxe, James L. McClelland, Surya Ganguli · ICLR 2014 · arXiv 1312.6120

Solves the learning dynamics of deep linear networks exactly and shows that random orthogonal initialization gives depth-independent learning times, unlike scaled Gaussian init.

**Why it matters:** This is the theoretical root of the orthogonal init used in almost every PPO and A2C implementation.

### [All you need is a good init](https://arxiv.org/abs/1511.06422)

Dmytro Mishkin, Jiri Matas · ICLR 2016 · arXiv 1511.06422

LSUV: start from orthogonal weights, then rescale each layer on a data batch so its output has unit variance.

**Why it matters:** Data-dependent init is a cheap way to fix scale when analytic formulas don't match the architecture.

### [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

Kaiming He, Xiangyu Zhang, Shaoqing Ren et al. · ICCV 2015 · arXiv 1502.01852

Derives a weight variance of 2/fan_in that preserves activation variance through ReLU layers, which made very deep rectifier networks trainable from scratch.

**Why it matters:** The right scale depends on the activation; ReLU networks need roughly twice the Xavier variance.

### [Deep Information Propagation](https://arxiv.org/abs/1611.01232)

Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli et al. · ICLR 2017 · arXiv 1611.01232

Shows that signals propagate only to a finite depth scale away from the order-to-chaos boundary, and networks deeper than that scale cannot be trained.

**Why it matters:** Initializing at the "edge of chaos" is what makes depth trainable.

### [Exponential expressivity in deep neural networks through transient chaos](https://arxiv.org/abs/1606.05340)

Ben Poole, Subhaneil Lahiri, Maithra Raghu et al. · NeurIPS 2016 · arXiv 1606.05340

Uses mean-field theory to show that random deep networks have an ordered and a chaotic phase, set by the weight and bias variances.

**Why it matters:** Init variance moves a network between vanishing and exploding signal regimes. The signal-propagation explorer visualizes exactly this.

### [Layer Normalization](https://arxiv.org/abs/1607.06450)

Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton · arXiv 2016 · arXiv 1607.06450

Layer normalization normalizes each sample's activations across features, so it works with any batch size, and online.

**Why it matters:** LayerNorm is now the most common stabilizer in RL networks because it doesn't depend on non-stationary batch statistics.

### [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868)

Tim Salimans, Diederik P. Kingma · NeurIPS 2016 · arXiv 1602.07868

Weight normalization reparameterizes each weight vector as a direction and a learned length, which decouples the norm from the direction and speeds optimization.

**Why it matters:** Controlling weight norm explicitly reappears in recent RL work such as hyperspherical normalization and batch-plus-weight normalization.

### [Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321)

Hongyi Zhang, Yann N. Dauphin, Tengyu Ma · ICLR 2019 · arXiv 1901.09321

Fixup rescales the initialization of residual branches, zero-initializing the last layer of each branch, so very deep residual networks train without normalization layers.

**Why it matters:** Zero or small init of a branch's final layer makes each block start as the identity. RL's small policy-head init uses the same idea.

### [Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522)

Greg Yang, Edward J. Hu · ICML 2021 · arXiv 2011.14522

Introduces the Maximal Update Parameterization (muP), which scales init and per-layer learning rates with width so features keep learning as width grows to infinity.

**Why it matters:** Init scale and learning rate are coupled. A good scheme at one width can be wrong at another.

## Policy-gradient implementation details

Large ablation studies of PPO-style agents, where small choices such as last-layer init, orthogonal weights, gradient clipping and advantage normalization turn out to matter as much as the algorithm.

### [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

John Schulman, Philipp Moritz, Sergey Levine et al. · ICLR 2016 · arXiv 1506.02438

Generalized Advantage Estimation is an exponentially weighted mix of n-step advantages, and lambda trades bias against variance.

**Why it matters:** Lower-variance advantages make policy-gradient updates steadier. The demo uses GAE(0.95).

### [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

John Schulman, Filip Wolski, Prafulla Dhariwal et al. · arXiv 2017 · arXiv 1707.06347

Proximal Policy Optimization optimizes a clipped surrogate objective so it can take several minibatch epochs per batch while keeping each policy update small.

**Why it matters:** PPO is the algorithm in the live demo. Its clipping is itself a stability mechanism.

### [A Closer Look at Deep Policy Gradients](https://arxiv.org/abs/1811.02553)

Andrew Ilyas, Logan Engstrom, Shibani Santurkar et al. · ICLR 2020 · arXiv 1811.02553

Measures how far deep policy-gradient algorithms depart from the theory that motivates them: gradient estimates are poorly correlated with the true gradient and value baselines barely reduce variance.

**Why it matters:** Noisy updates make training sensitive to scale and init. Stability tricks partly compensate for this noise.

### [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729)

Logan Engstrom, Andrew Ilyas, Shibani Santurkar et al. · ICLR 2020 · arXiv 2005.12729

Shows that much of PPO's advantage over TRPO comes from code-level optimizations (orthogonal init, reward scaling, value clipping, learning-rate annealing), not from the clipped objective.

**Why it matters:** Init and other unreported details can dominate algorithm comparisons.

### [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990)

Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk et al. · ICLR 2021 · arXiv 2006.05990

Trains more than 250,000 agents across implementation choices. One of the strongest recommendations: initialize the policy's last layer with weights about 100x smaller, so the initial action distribution is near-uniform and independent of the observation.

**Why it matters:** This is direct evidence that output-layer init matters in on-policy RL. The demo's "Orthogonal + small heads" vs "no small heads" comparison reproduces it in miniature.

### [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

Shengyi Huang, Rousslan Fernand Julien Dossa, Antonin Raffin et al. · ICLR Blog Track 2022

Catalogs 37 implementation details of the reference PPO code, including orthogonal init with gain sqrt(2) for hidden layers, 0.01 for the policy head and 1 for the value head, plus Adam epsilon 1e-5 and global gradient clipping at 0.5.

**Why it matters:** This is the source of the exact recipe in the demo's "Orthogonal + small heads" scheme.

### [The Phenomenon of Policy Churn](https://arxiv.org/abs/2206.00730)

Tom Schaul, André Barreto, John Quan et al. · NeurIPS 2022 · arXiv 2206.00730

Finds that in value-based agents the greedy policy changes on a large fraction of states after a single gradient step, and argues this churn acts as implicit exploration.

**Why it matters:** Small updates can still cause large behavioral changes. Policy stability is not the same as parameter stability.

### [A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning](https://arxiv.org/abs/2405.19153)

Arthur Juliani, Jordan T. Ash · NeurIPS 2024 · arXiv 2405.19153

Studies plasticity loss in on-policy RL under distribution shift. Many methods that help off-policy agents fail here; regenerative methods such as shrink-and-perturb-style resets and regularization toward the initialization help.

**Why it matters:** Interventions that keep the network close to a good initialization are a common thread.

### [No Representation, No Trust: Connecting Representation, Collapse, and Trust Issues in PPO](https://arxiv.org/abs/2405.00662)

Skander Moalla, Andrea Miele, Daniil Pyatko et al. · NeurIPS 2024 · arXiv 2405.00662

Shows that PPO agents also lose plasticity: feature rank collapses over training, which then breaks the trust region. Proposes Proximal Feature Optimization, a regularizer on feature change.

**Why it matters:** On-policy RL is not immune to plasticity loss. The demo tracks the actor's feature rank for this reason.

## Value learning, divergence and the deadly triad

Why bootstrapped value learning with function approximation is fragile: divergence, overestimation, feature collapse, and the regularizers that counter them.

### [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

Volodymyr Mnih, Koray Kavukcuoglu, David Silver et al. · NeurIPS Deep Learning Workshop 2013 · arXiv 1312.5602

Deep Q-Networks learned Atari from pixels, using experience replay to stabilize Q-learning with a convolutional network.

**Why it matters:** Replay buffers, and later target networks, were the first stability fixes for deep value learning.

### [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)

Scott Fujimoto, Herke van Hoof, David Meger · ICML 2018 · arXiv 1802.09477

TD3 counters overestimation bias in actor-critic methods with clipped double-Q learning, delayed policy updates and target policy smoothing.

**Why it matters:** Value-estimation error compounds through bootstrapping. Stability comes from controlling it.

### [Deep Reinforcement Learning and the Deadly Triad](https://arxiv.org/abs/1812.02648)

Hado van Hasselt, Yotam Doron, Florian Strub et al. · arXiv 2018 · arXiv 1812.02648

Empirically studies the deadly triad (function approximation, bootstrapping, off-policy learning) in DQN variants. Unbounded divergence is rare, but soft divergence is common and depends on target networks, multi-step returns and prioritization.

**Why it matters:** Divergence risk is a measurable function of design choices, not a fixed fact of the algorithm.

### [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel et al. · ICML 2018 · arXiv 1801.01290

Soft Actor-Critic is an off-policy maximum-entropy actor-critic method that is notably more stable across seeds than earlier off-policy methods.

**Why it matters:** An entropy bonus keeps the policy from collapsing early, much like the small policy-head init does at step zero.

### [Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning](https://arxiv.org/abs/2010.14498)

Aviral Kumar, Rishabh Agarwal, Dibya Ghosh et al. · ICLR 2021 · arXiv 2010.14498

Shows that bootstrapped value learning progressively reduces the effective rank (srank) of learned features, and that this rank collapse correlates with poor performance.

**Why it matters:** Introduces the srank metric that the demo plots as "feature rank".

### [Interference and Generalization in Temporal Difference Learning](https://arxiv.org/abs/2003.06350)

Emmanuel Bengio, Joelle Pineau, Doina Precup · ICML 2020 · arXiv 2003.06350

Finds that TD-trained networks show lower interference between states and worse generalization than regression-trained ones, and links this to how targets evolve over training.

**Why it matters:** How the targets are produced shapes the representation the network learns.

### [Revisiting Fundamentals of Experience Replay](https://arxiv.org/abs/2007.06700)

William Fedus, Prajit Ramachandran, Rishabh Agarwal et al. · ICML 2020 · arXiv 2007.06700

Disentangles replay capacity from the age of the oldest policy in the buffer, and shows that n-step returns are key to benefiting from larger replay.

**Why it matters:** Replay ratio and data age are stability levers as important as the network itself.

### [DR3: Value-Based Deep Reinforcement Learning Requires Explicit Regularization](https://arxiv.org/abs/2112.04716)

Aviral Kumar, Rishabh Agarwal, Tengyu Ma et al. · ICLR 2022 · arXiv 2112.04716

DR3 identifies implicit regularization in TD learning that makes features of consecutive state-action pairs co-adapt, and adds an explicit regularizer on their dot product.

**Why it matters:** Value-based RL needs explicit regularization that supervised learning does not.

### [The Difficulty of Passive Learning in Deep Reinforcement Learning](https://arxiv.org/abs/2110.14020)

Georg Ostrovski, Pablo Samuel Castro, Will Dabney · NeurIPS 2021 · arXiv 2110.14020

The tandem experiment: a passive agent trained on exactly the same data as an active agent still fails badly. The paper traces this mainly to function approximation interacting with the data distribution.

**Why it matters:** Identical data and objectives can still give very different results: optimization dynamics matter, not only the data.

### [Dissecting Deep RL with High Update Ratios: Combatting Value Divergence](https://arxiv.org/abs/2403.05996)

Marcel Hussing, Claas Voelcker, Igor Gilitschenski et al. · RLC 2024 · arXiv 2403.05996

At high update-to-data ratios, the main failure is not the primacy bias but divergence of Q-values driven by out-of-distribution action prediction. Normalizing the output features fixes much of it without resets.

**Why it matters:** Some failures that resets were fixing can be prevented by architecture instead.

### [Simplifying Deep Temporal Difference Learning](https://arxiv.org/abs/2407.04811)

Matteo Gallici, Mattie Fellows, Benjamin Ellis et al. · ICLR 2025 · arXiv 2407.04811

Proves that LayerNorm plus L2 regularization makes TD learning stable even off-policy, and builds PQN: a Q-learning agent with no target network or replay buffer, running in vectorized environments.

**Why it matters:** Normalization can replace target networks as a stabilizer, which is a strong argument for architecture-level fixes.

### [Stop Regressing: Training Value Functions via Classification for Scalable Deep RL](https://arxiv.org/abs/2403.03950)

Jesse Farebrother, Jordi Orbay, Quan Vuong et al. · ICML 2024 · arXiv 2403.03950

Training value functions with a cross-entropy loss over binned returns (HL-Gauss) instead of MSE regression scales better and is more robust to noisy targets.

**Why it matters:** Changing the loss's geometry can remove a whole class of value-learning instabilities.

## Plasticity loss and dormant neurons

Networks trained on RL's shifting targets gradually lose the ability to fit new targets. This shows up as dormant neurons, collapsing feature rank, growing weight norms and sharp curvature.

### [Transient Non-Stationarity and Generalisation in Deep Reinforcement Learning](https://arxiv.org/abs/2006.05826)

Maximilian Igl, Gregory Farquhar, Jelena Luketina et al. · ICLR 2021 · arXiv 2006.05826

Shows that transient non-stationarity early in RL training permanently hurts generalization. ITER periodically distills the agent into a freshly initialized network.

**Why it matters:** Re-initializing and distilling can undo damage that early training left behind.

### [Understanding and Preventing Capacity Loss in Reinforcement Learning](https://arxiv.org/abs/2204.09560)

Clare Lyle, Mark Rowland, Will Dabney · ICLR 2022 · arXiv 2204.09560

Shows that networks trained on sequences of targets lose capacity to fit new ones, measured by feature rank. InFeR regresses some outputs toward their values at initialization to preserve capacity.

**Why it matters:** Staying close to initialization's function preserves plasticity, which makes the init a resource to protect.

### [DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization](https://arxiv.org/abs/2310.19668)

Guowei Xu, Ruijie Zheng, Yongyuan Liang et al. · ICLR 2024 · arXiv 2310.19668

DrM shows that the dormant ratio in visual RL agents tracks exploration and performance, and uses it to drive perturbation, exploration and exploitation schedules.

**Why it matters:** Network-health metrics can act as control signals, not just diagnostics.

### [Loss of Plasticity in Continual Deep Reinforcement Learning](https://arxiv.org/abs/2303.07507)

Zaheer Abbas, Rosie Zhao, Joseph Modayil et al. · CoLLAs 2023 · arXiv 2303.07507

Shows plasticity loss in continual deep RL on Atari and connects it to activation-footprint changes. Concatenated ReLUs (CReLU) mitigate it.

**Why it matters:** Activation choice affects how quickly units die. The demo lets you switch tanh and ReLU.

### [Maintaining Plasticity in Deep Continual Learning](https://arxiv.org/abs/2306.13812)

Shibhansh Dohare, J. Fernando Hernandez-Garcia, Parash Rahman et al. · Nature 2024 · arXiv 2306.13812

Demonstrates large-scale loss of plasticity under continual learning with standard deep learning (including on ImageNet-derived tasks and in RL) and proposes continual backpropagation, which keeps re-initializing a small fraction of low-utility units.

**Why it matters:** Initialization is not a one-time event. Injecting fresh random units keeps a network trainable.

### [PLASTIC: Improving Input and Label Plasticity for Sample Efficient Reinforcement Learning](https://arxiv.org/abs/2306.10711)

Hojoon Lee, Hanseul Cho, Hyunseung Kim et al. · NeurIPS 2023 · arXiv 2306.10711

Separates input plasticity (adapting to new inputs) from label plasticity (adapting to new targets) and combines SAM, LayerNorm, CReLU and resets in PLASTIC.

**Why it matters:** Different remedies target different parts of the problem, so they combine well.

### [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://arxiv.org/abs/2302.12902)

Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro et al. · ICML 2023 · arXiv 2302.12902

Defines tau-dormant neurons (normalized mean activation at or below tau) and shows their fraction grows during deep RL training. ReDo recycles them by re-initializing their incoming weights and zeroing their outgoing weights.

**Why it matters:** The demo's dormant-fraction chart uses this definition directly.

### [Understanding plasticity in neural networks](https://arxiv.org/abs/2303.01486)

Clare Lyle, Zeyu Zheng, Evgenii Nikishin et al. · ICML 2023 · arXiv 2303.01486

Finds that plasticity loss is best explained by changes in loss-landscape curvature rather than by weight norm or rank alone, and shows LayerNorm is a strong mitigation.

**Why it matters:** No single scalar metric predicts plasticity loss. The demo plots several for this reason.

### [Disentangling the Causes of Plasticity Loss in Neural Networks](https://arxiv.org/abs/2402.18762)

Clare Lyle, Zeyu Zheng, Khimya Khetarpal et al. · ICML 2024 · arXiv 2402.18762

Disentangles several independent causes of plasticity loss (preactivation shift, unit saturation and dormancy, growing parameter norms) and shows that layer normalization plus weight decay together are robust across settings.

**Why it matters:** Combine a normalization layer with norm control instead of looking for a single fix.

### [Normalization and effective learning rates in reinforcement learning](https://arxiv.org/abs/2407.01800)

Clare Lyle, Zeyu Zheng, Khimya Khetarpal et al. · NeurIPS 2024 · arXiv 2407.01800

Shows that with normalization layers, growing parameter norms silently shrink the effective learning rate. Normalize-and-Project keeps the effective step size constant.

**Why it matters:** The weight-norm chart in the demo is about this effect, which is invisible in the loss curve.

## Resets and plasticity interventions

Remedies that partly re-initialize or re-randomize the network during training: full or partial resets, shrink-and-perturb, plasticity injection, continual backprop and regularizing toward the initialization.

### [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

Diederik P. Kingma, Jimmy Ba · ICLR 2015 · arXiv 1412.6980

Adam combines momentum with per-parameter adaptive step sizes from bias-corrected first and second moment estimates.

**Why it matters:** Adam's epsilon and moment state interact with init scale. RL code commonly uses epsilon = 1e-5, as the demo does.

### [On Warm-Starting Neural Network Training](https://arxiv.org/abs/1910.08475)

Jordan T. Ash, Ryan P. Adams · NeurIPS 2020 · arXiv 1910.08475

Warm-starting from a model trained on part of the data generalizes worse than training from scratch. Shrink-and-perturb (scale weights down, add noise) recovers most of the gap.

**Why it matters:** Partially moving weights back toward a random init is a simple, general plasticity fix.

### [Fortuitous Forgetting in Connectionist Networks](https://arxiv.org/abs/2202.00155)

Hattie Zhou, Ankit Vani, Hugo Larochelle et al. · ICLR 2022 · arXiv 2202.00155

Shows that repeatedly forgetting and relearning, by resetting later layers, steers networks toward better generalizing solutions.

**Why it matters:** Controlled re-initialization can act as a regularizer.

### [The Primacy Bias in Deep Reinforcement Learning](https://arxiv.org/abs/2205.07802)

Evgenii Nikishin, Max Schwarzer, Pierluca D'Oro et al. · ICML 2022 · arXiv 2205.07802

Identifies the primacy bias, a tendency to overfit early experience, and shows that periodically resetting the last layers of the network, while keeping the replay buffer, improves sample efficiency substantially.

**Why it matters:** Re-initialization during training can beat continuous training. This is the foundation for most later reset work.

### [Bigger, Better, Faster: Human-level Atari with human-level efficiency](https://arxiv.org/abs/2305.19452)

Max Schwarzer, Johan Obando-Ceron, Aaron Courville et al. · ICML 2023 · arXiv 2305.19452

Bigger, Better, Faster reaches human-level Atari 100k performance by scaling the network, using shrink-and-perturb resets, annealing n-step and discount, and using weight decay.

**Why it matters:** Resets are what make larger networks and high replay ratios usable.

### [Deep Reinforcement Learning with Plasticity Injection](https://arxiv.org/abs/2305.15555)

Evgenii Nikishin, Junhyuk Oh, Georg Ostrovski et al. · NeurIPS 2023 · arXiv 2305.15555

Plasticity injection adds a freshly initialized copy of the head and subtracts a frozen copy, so plasticity increases without changing the network's current output or adding trainable parameters permanently.

**Why it matters:** Fresh random weights can be added without a performance drop.

### [Maintaining Plasticity in Continual Learning via Regenerative Regularization](https://arxiv.org/abs/2308.11958)

Saurabh Kumar, Henrik Marklund, Benjamin Van Roy · arXiv 2023 · arXiv 2308.11958

L2 Init regularizes parameters toward their initial values instead of toward zero, maintaining plasticity in continual learning more simply than resets.

**Why it matters:** The initial weights are a good anchor to return to.

### [Resetting the Optimizer in Deep RL: An Empirical Study](https://arxiv.org/abs/2306.17833)

Kavosh Asadi, Rasool Fakoor, Shoham Sabach · NeurIPS 2023 · arXiv 2306.17833

Resetting Adam's moment estimates whenever the target network updates significantly improves Rainbow on Atari, because stale moments from earlier targets corrupt new updates.

**Why it matters:** Optimizer state is part of what has to be re-initialized.

### [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://arxiv.org/abs/2404.00781)

Mohamed Elsayed, A. Rupam Mahmood · ICLR 2024 · arXiv 2404.00781

Utility-based perturbed gradient descent protects useful units and perturbs less useful ones, addressing both plasticity loss and catastrophic forgetting.

**Why it matters:** Targeted, utility-aware perturbation beats uniform resets when old knowledge matters.

### [Learning Continually by Spectral Regularization](https://arxiv.org/abs/2406.06811)

Alex Lewandowski, Michał Bortkiewicz, Saurabh Kumar et al. · arXiv 2024 · arXiv 2406.06811

Spectral regularization keeps each layer's largest singular value near 1, preserving the gradient-propagation properties of the initialization and so maintaining trainability over long task sequences.

**Why it matters:** Instead of resetting weights, preserve the property of the initialization that matters: its spectrum.

### [Non-Stationary Learning of Neural Networks with Automatic Soft Parameter Reset](https://arxiv.org/abs/2411.04034)

Alexandre Galashov, Michalis K. Titsias, András György et al. · arXiv 2024 · arXiv 2411.04034

Models parameter drift toward the initialization as an Ornstein-Uhlenbeck process with a learned drift rate, which gives soft, adaptive resets under non-stationarity.

**Why it matters:** Reset strength can be learned instead of scheduled by hand.

## Normalization and architectures for stable scaling

Layer, batch, weight and spectral normalization, residual blocks and other architecture choices that let RL networks grow larger and train with higher replay ratios without diverging.

### [D2RL: Deep Dense Architectures in Reinforcement Learning](https://arxiv.org/abs/2010.09163)

Samarth Sinha, Homanga Bharadhwaj, Aravind Srinivas et al. · arXiv 2020 · arXiv 2010.09163

Feeds the state input into every hidden layer (dense connections), which helps deeper actor-critic networks train in continuous control.

**Why it matters:** Skip connections ease the gradient-flow problems that limit depth in RL.

### [Spectral Normalisation for Deep Reinforcement Learning: an Optimisation Perspective](https://arxiv.org/abs/2105.05246)

Florin Gogianu, Tudor Berariu, Mihaela Rosca et al. · ICML 2021 · arXiv 2105.05246

Shows that spectral normalization's benefit in DQN-style agents comes mostly from its effect on optimization (an effective step size), not from smoothness. Proposes cheaper alternatives.

**Why it matters:** Many stabilizers work by changing effective learning rates.

### [Towards Deeper Deep Reinforcement Learning with Spectral Normalization](https://arxiv.org/abs/2106.01151)

Johan Bjorck, Carla P. Gomes, Kilian Q. Weinberger · ICML 2021 · arXiv 2106.01151

Spectral normalization lets much deeper and wider networks be used in SAC-style agents, where unnormalized large networks would otherwise destabilize.

**Why it matters:** Smoothness constraints, not just init, unlock depth in RL.

### [Training Larger Networks for Deep Reinforcement Learning](https://arxiv.org/abs/2102.07920)

Kei Ota, Devesh K. Jha, Asako Kanezaki · arXiv 2021 · arXiv 2102.07920

Shows that naively enlarging RL networks hurts, and gets gains by decoupling representation learning from RL and using DenseNet-style connections plus distributed training.

**Why it matters:** Scaling RL networks needs both architecture and training-procedure changes.

### [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba et al. · Nature 2025 · arXiv 2301.04104

DreamerV3 masters over 150 tasks with fixed hyperparameters using robustness techniques such as symlog predictions, LayerNorm, return normalization and zero-initialized output layers for some heads.

**Why it matters:** One set of hyperparameters can work everywhere once scale is controlled at every interface.

### [Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control](https://arxiv.org/abs/2405.16158)

Michal Nauman, Mateusz Ostaszewski, Krzysztof Jankowski et al. · NeurIPS 2024 · arXiv 2405.16158

BRO scales the critic with LayerNorm and residual blocks, uses strong regularization and resets, and adds optimistic exploration, reaching state-of-the-art continuous control.

**Why it matters:** Scaling up only works once normalization and resets keep training stable.

### [CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity](https://arxiv.org/abs/1902.05605)

Aditya Bhatt, Daniel Palenicek, Boris Belousov et al. · ICLR 2024 · arXiv 1902.05605

CrossQ removes target networks and uses carefully applied batch normalization, joint forward passes of current and next states and wider critics, matching REDQ/DroQ sample efficiency at a low update ratio.

**Why it matters:** With the right normalization, the classic stabilizers (target nets, high UTD) can be dropped.

### [In value-based deep reinforcement learning, a pruned network is a good network](https://arxiv.org/abs/2402.12479)

Johan Obando-Ceron, Aaron Courville, Pablo Samuel Castro · ICML 2024 · arXiv 2402.12479

Gradual magnitude pruning makes value-based agents perform better as networks grow, where dense scaling usually hurts.

**Why it matters:** Bigger dense networks are harder to keep stable in RL, so sparsity is an alternative route.

### [Mixtures of Experts Unlock Parameter Scaling for Deep RL](https://arxiv.org/abs/2402.08609)

Johan Obando-Ceron, Ghada Sokar, Timon Willi et al. · ICML 2024 · arXiv 2402.08609

Soft mixture-of-experts layers in value-based agents give performance that improves with parameter count, along with fewer dormant neurons and more stable feature rank.

**Why it matters:** Architecture changes show up directly in the plasticity metrics.

### [Overestimation, Overfitting, and Plasticity in Actor-Critic: the Bitter Lesson of Reinforcement Learning](https://arxiv.org/abs/2403.00514)

Michal Nauman, Michał Bortkiewicz, Piotr Miłoś et al. · ICML 2024 · arXiv 2403.00514

Benchmarks over 60 off-policy agents built from combinations of regularization and plasticity interventions across 14 tasks. Generic network regularization such as LayerNorm, together with resets, does as well as RL-specific tricks.

**Why it matters:** Practical RL stability benefits most from standard deep-learning regularizers.

### [SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning](https://arxiv.org/abs/2410.09754)

Hojoon Lee, Dongyoon Hwang, Donghu Kim et al. · ICLR 2025 · arXiv 2410.09754

SimBa combines running observation normalization, residual feedforward blocks and LayerNorm. The result has a simplicity bias that lets RL networks scale up in parameters without losing performance.

**Why it matters:** The architecture's inductive bias, not only its init, decides whether bigger networks help in RL.

### [Hyperspherical Normalization for Scalable Deep Reinforcement Learning](https://arxiv.org/abs/2502.15280)

Hojoon Lee, Youngdo Lee, Takuma Seno et al. · ICML 2025 · arXiv 2502.15280

SimbaV2 constrains weights and features to a hypersphere and uses distributional value estimation with reward scaling, which keeps effective learning rates stable as models and compute scale.

**Why it matters:** Projecting weights onto a fixed norm directly removes the growing-weight-norm failure mode.

### [Scaling Off-Policy Reinforcement Learning with Batch and Weight Normalization](https://arxiv.org/abs/2502.07523)

Daniel Palenicek, Florian Vogt, Joe Watson et al. · arXiv 2025 · arXiv 2502.07523

Adds weight normalization to CrossQ so it scales reliably to higher update-to-data ratios without resets.

**Why it matters:** Norm control on weights complements batch normalization on activations.

### [Stable Gradients for Stable Learning at Scale in Deep Reinforcement Learning](https://arxiv.org/abs/2506.15544)

Roger Creus Castanyer, Johan Obando-Ceron, Lu Li et al. · arXiv 2025 · arXiv 2506.15544

Attributes the failure of naively scaled deep RL networks to non-stationarity combined with poor gradient flow, and stabilizes scaling with gradient-propagation-preserving architecture and optimizer changes.

**Why it matters:** Signal propagation, the theme of the first section, returns as the main issue at scale.

## Evaluation and reproducibility

How to tell whether a stability improvement is real: seed variance, confidence intervals and reporting practice.

### [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560)

Peter Henderson, Riashat Islam, Philip Bachman et al. · AAAI 2018 · arXiv 1709.06560

Shows that deep RL results vary hugely with random seeds, hyperparameters, network architecture and code base, so reported improvements are often not significant.

**Why it matters:** Init effects must be judged across many seeds. The demo runs several seeds per scheme and shades their range.

### [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro et al. · NeurIPS 2021 · arXiv 2108.13264

Argues that point estimates from few runs mislead, and recommends interquartile mean, stratified bootstrap confidence intervals and performance profiles (the rliable library).

**Why it matters:** Report intervals over seeds, not single curves.
