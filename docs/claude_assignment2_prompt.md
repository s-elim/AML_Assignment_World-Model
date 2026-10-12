# Master Prompt for Claude: Assignment 2 Technical Report Overhaul

Copy and paste the prompt below directly into Claude.

```markdown
You are a Senior Research Scientist and NeurIPS/CoRL Area Chair specializing in visual world models and robot manipulation. You are tasked with overhauling the implementation and authoring a comprehensive, publication-grade technical report for Assignment 2. The report must rigorously fulfill every requirement of Assignment 2 under Setting 3 (Simulator-Based Evaluation), maintain strict continuity with the research scope established in Assignment 1, adopt a single-column technical report format, and span between 13 and 16 compiled PDF pages.

---

### Context and Research Scope Continuity

The research area established in Assignment 1 is learned world models used as pretrained foundation models for robot control, specifically investigating whether pretraining a predictive model of environment dynamics produces a transferable substrate for downstream robot control.

For Assignment 2, follow Setting 3 (Simulator-Based Evaluation) by implementing, benchmarking, and comparatively analyzing four models in a unified MuJoCo-Gymnasium robotic manipulation suite based on the LIBERO benchmark:
1. Action-Conditioned V-JEPA 2 (Assran et al., 2025): Self-supervised video foundation model predicting latent future features conditioned on continuous actions without pixel reconstruction.
2. DINO-WM (Zhou et al., ICML 2025): World model operating over self-supervised patch tokens from a frozen DINOv2 visual backbone.
3. TD-MPC2 (Hansen et al., ICLR 2024): Scalable model-based reinforcement learning baseline utilizing a task-oriented latent dynamics model with value and reward heads.
4. Direct Behavioral Cloning (BC) Baseline: Closed-loop policy mapping visual observations directly to Cartesian actions without predictive rollouts.

---

### Strict Format, Volume, and Stylistic Constraints

1. Format: Single-column technical report using LaTeX (`\documentclass[11pt,a4paper]{article}` with `\usepackage[margin=1in]{geometry}`). Use standard 11pt type, single spacing, standard section styling, and clear float placement.
2. Length: The compiled document must strictly span 13 to 16 pages. A document under 13 pages lacks necessary depth; a document over 16 pages violates assignment limits. Adjust narrative depth, mathematical exposition, qualitative analysis, and figure sizing to ensure the compiled PDF falls squarely in the 13 to 16 page window.
3. Prohibited Vocabulary and Constructs:
   - Do not use em dashes or en dashes anywhere. Use a comma, colon, semicolon, or period instead.
   - Do not use inflated filler vocabulary, speculative buzzwords, or vague compliments.
   - Do not use formulaic phrases such as "not only X but also Y", "X is not just A; it is B", or repetitive transitions like "Moreover", "Furthermore", "Additionally".
   - Active voice and publication-grade flowing prose at PAMI, IJCV, and NeurIPS standards. Every claim must carry a concrete mechanism, quantity, or formal citation.
4. Rigorous Seeding and Protocol:
   - Deterministic evaluation seed: 0.
   - Episode budget: Exactly 10 rollout episodes per task across 4 continuous manipulation tasks (`reach`, `push`, `pick_place`, `drawer_open`).
   - Planning horizon: 10 steps per episode.

---

### Required Document Architecture and Section Breakdown

Structure the LaTeX manuscript to span 13 to 16 single-column pages following this detailed plan:

#### Section 1: Introduction and Research Problem Formulation (Target: ~2 pages)
- Formal statement of visual continuous control under partial observability as a POMDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \Omega, \mathcal{R}, \gamma)$.
- The core dilemma: Pixel-level reconstruction (high computational overhead, capacity wasted on task-irrelevant high frequencies) versus latent-space dynamics prediction (information loss, potential representation collapse, non-invertibility).
- Direct bridge to Assignment 1: Evaluating whether joint-embedding predictive architectures (JEPA) and self-supervised visual representations provide superior inductive biases for zero-shot and few-shot planning compared to task-specific reconstruction and value-equivalent models.
- Explicit statements of the four research questions investigated, followed by a concise summary of empirical contributions.

#### Section 2: Theoretical Foundations and Corpus Taxonomy (Target: ~2 pages)
- Systematic categorization grounded in the Assignment 1 corpus:
  1. Recurrent state-space models with pixel reconstruction (DayDreamer, DreamerV3) optimizing an ELBO objective.
  2. Latent predictive architectures without reconstruction (V-JEPA 2, DINO-WM, Navigation World Models) optimizing feature prediction losses under stop-gradient operators.
  3. Value-equivalent predictive dynamics (TD-MPC2) optimizing latent transition consistency jointly with reward and value bellman targets.
  4. Non-predictive direct imitation baselines.
- Comparative theoretical table summarizing mathematical objectives, loss formulations, computational complexity per forward pass, and representation collapse mitigation mechanisms across all paradigms.

#### Section 3: Model Architectures and Algorithmic Formulations (Target: ~3 pages)
- Provide full mathematical formulations with formal numbered equations for all four models:
  - Action-Conditioned V-JEPA 2: Encoder $s_\phi(o_t)$, cross-attention action conditioning module $g_\psi(z_t, a_t)$, multi-step predictor $f_\theta$, stop-gradient target update, and $L_1$/cosine latent loss.
  - DINO-WM: Frozen ViT patch tokenization, spatial register handling, transformer-based latent dynamics transition $T_\omega(z_t, a_t)$, and mean squared error loss in DINOv2 latent space.
  - TD-MPC2: Encoder $h_\xi$, latent dynamics $d_\chi(z_t, a_t)$, reward head $r_\psi$, value head $q_\nu$, and multi-task loss.
  - Direct Behavioral Cloning: Visual backbone with MLP policy head trained via supervised negative log-likelihood on expert action distributions.
- Vectorized Model Predictive Control (MPC) with the Cross-Entropy Method (CEM):
  - Formal algorithm box detailing population initialization, elite selection, Gaussian parameter updates, action bounding, and temporal covariance smoothing.
  - Latent trajectory cost formulation: Cosine distance and Euclidean metric to goal latent embedding $z_g = s_\phi(o_g)$.

#### Section 4: Experimental Setup and Simulation Benchmark (Target: ~2 pages)
- Unified simulation platform: MuJoCo-Gymnasium manipulation suite adhering to LIBERO benchmark semantics.
- Task definitions:
  1. `reach`: 4-DoF continuous Cartesian end-effector control to target coordinates.
  2. `push`: Dynamic non-prehensile object translation across frictional table surfaces.
  3. `pick_place`: Multi-stage prehensile manipulation involving contact discovery, grasping, lifting, and targeted placement.
  4. `drawer_open`: Articulated mechanism interaction against joint limits and non-linear friction.
- Expert demonstration dataset: 200 demonstrations (50 per task) collected via operational space controllers and scripted experts.
- Observation and action spaces: Dual camera views (wrist and exterior $128 \times 128 \times 3$), Cartesian velocity commands with continuous gripper state $[-0.05, 0.05]^3 \times [-1, 1]$.
- Training configuration: Exact learning rates, batch sizes, optimizers, epoch schedules, hardware specifications, and wall-clock training times.

#### Section 5: Quantitative Results and Comparative Analysis (Target: ~2.5 pages)
- Benchmark Performance Table: Success rate (%), final Euclidean goal distance (cm), cumulative reward, and episode completion time across all four models and four tasks under seed 0 (10 rollouts per task).
- Multi-Step Latent Prediction Accuracy Table and Curves: Open-loop rollout error at horizons $k \in \{1, 3, 5, 10\}$ measuring compound dynamics drift in latent space.
- Inference Latency and Computational Profiling: Latency per planning step (ms), throughput (steps/s), peak GPU VRAM allocation during planning, and parameter counts.
- Statistical and failure significance: Detailed discussion of why latent planners succeed on reach and push but encounter severe degradation on contact-rich articulated tasks.

#### Section 6: Qualitative Results and Visual Rollout Analysis (Target: ~2.5 pages)
- Visual Rollout Filmstrips: Render high-resolution side-by-side rollout sequences (8 discrete time-steps per episode) comparing successful and failed executions of V-JEPA 2, DINO-WM, and BC on `reach` and `push`.
- Latent Space Manifold Analysis: 2D t-SNE and PCA trajectory projections comparing the imagined latent rollouts from the CEM optimizer against the realized ground-truth environment trajectories.
- Energy Landscapes and Cost Surfaces: 2D contour heatmaps of the latent cost surface over action perturbations, demonstrating local minima and optimizer convergence behavior.
- Failure Mode Taxonomy: Structural visual inspection and classification into three specific failure regimes: contact discontinuity errors, goal embedding ambiguity, and compounding open-loop drift.

#### Section 7: Ablation Studies and Sensitivity Analysis (Target: ~1.5 pages)
- Planning Horizon Sensitivity: Evaluating planning performance across horizons $H \in \{3, 5, 10, 15\}$.
- CEM Optimizer Capacity: Ablating the number of samples $N \in \{32, 64, 128, 256, 512\}$ and elite fractions $K \in \{0.05, 0.1, 0.2\}$.
- Latent Cost Metric Formulation: Direct comparison between Cosine Distance, Euclidean $L_2$ Distance, and Mahalanobis distance in latent space.
- Representation Backbone Freezing vs Fine-Tuning: Evaluating the trade-off between frozen pre-trained representations (DINOv2, V-JEPA) and end-to-end policy learning.

#### Section 8: Discussion, Limitations, and Future Trajectories (Target: ~1 page)
- Critical synthesis of why pure latent world models without physical contact grounding struggle with rigid-body boundaries.
- Implications for physical foundation models and 5D/6D manipulation policies.
- Roadmap toward hybrid world models integrating geometric priors (3D Gaussians, depth tokens) with action-conditioned video transformers.

#### References and Bibliographic Hygiene (Target: ~1 to 1.5 pages)
- Full, unabridged bibliographic records with authors, complete titles, conference/journal venues, years, and official DOIs or arXiv identifiers.
- Must include the full 20-article corpus from Assignment 1 along with all relevant simulation benchmarks and foundation model papers.

---

### Required Visuals, Figures, and Tables

Generate and embed the following publication-quality vector figures (using Matplotlib with PDF/PNG dual export, colorblind-safe palettes, LaTeX typography) and LaTeX tables:
1. Figure 1: Architectural diagram of the Action-Conditioned V-JEPA 2 model and comparative latent dynamics pipelines.
2. Figure 2: Vectorized CEM latent MPC trajectory optimizer flowchart showing candidate generation, latent unrolling, cost evaluation, and elite updating.
3. Figure 3: Multi-step open-loop latent prediction error ($k \in \{1, 3, 5, 10\}$) across models and tasks.
4. Figure 4: Pareto trade-off curve comparing execution success rate against planning latency per step (ms).
5. Figure 5: 2D latent cost energy landscape and CEM optimizer convergence paths.
6. Figure 6: Visual robot rollout filmstrips (8-step horizontal subfigure sequences) comparing planned versus executed trajectories in the simulator.
7. Figure 7: t-SNE and PCA latent trajectory manifolds showing distribution alignment between imagined and actual paths.
8. Figure 8: Failure case diagnostic taxonomy detailing visual snapshots of contact slip, boundary collision, and goal hallucination.
9. Table 1: Model architectural taxonomy, parameter counts, latent dimensionality, and training objectives.
10. Table 2: Benchmark manipulation quantitative results across 4 tasks (Success Rate %, Goal Distance cm, Reward).
11. Table 3: Multi-horizon latent prediction error ($k \in \{1, 3, 5, 10\}$ MSE) and drift rate.
12. Table 4: Computational profiling, inference latency breakdown (encoder, dynamics rollout, action extraction), and VRAM footprint.
13. Table 5: CEM planning hyperparameter ablation results (horizon $H$, sample count $N$, elite fraction $K$, cost metric).

---

### Step-by-Step Execution Plan

1. Verify and inspect existing codebase modules in `src/`, checkpoints in `results/checkpoints/`, dataset trajectories in `data/`, and evaluation logs in `results/tables/`.
2. Generate all visual rollout filmstrips from rollout recordings and render high-resolution vector figures in `results/figures/`.
3. Author the complete LaTeX manuscript `report/assignment2_report.tex` with thorough mathematical formulations, empirical tables, embedded figures, and complete bibliography.
4. Compile the manuscript using `tectonic` or `pdflatex` and verify that the resulting PDF page count is strictly within 13 to 16 pages. If the page count is outside this range, adjust narrative depth and figure sizing until the requirement is satisfied.
5. Provide a clear summary of verification results, including exact compiled page count, figure rendering status, and benchmark reproducibility commands.
```
