# Assignment 2: Simulator-Based Comparative Evaluation of Visual World Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and comparatively evaluate four visual foundation world models (V-JEPA 2-AC, DINO-WM, TD-MPC2, and Direct BC Policy) in simulated robot manipulation environments across four manipulation tasks under Setting 3 (Simulator-Based Evaluation), producing empirical results, statistical curves, LaTeX tables, reproducibility code, and a formal technical report.

**Architecture:** A modular PyTorch/Gymnasium framework decoupling visual encoders (video ViT / DINOv2 / compact ViT), dynamics predictors (causal transformer / latent transition MLP / learned RSSM), planning optimizers (parallelized Cross-Entropy Method with latent cosine/L1 distance costs), and standardized robot simulation wrappers with deterministic multi-seed evaluation.

**Tech Stack:** Python 3.11, PyTorch 2.x, Gymnasium, NumPy, SciPy, Matplotlib, Seaborn, Pandas, PyYAML, Tectonic (LaTeX).

**Spec:** `AML_Assignment_World-Model/assignment#2.pdf` and `AML_Assignment_World-Model/Selim_22546245 Assignment 01.docx`.

## Global Constraints

- Experimental setting strictly follows Setting 3 (Simulator-Based Evaluation) from `assignment#2.pdf`.
- Research area remains strictly consistent with Assignment 1: Learned world models used as pretrained foundation models for robot control.
- Exactly 4 compared models: V-JEPA 2-AC (primary), DINO-WM, TD-MPC2, and Direct BC Policy.
- Exactly 4 canonical robot manipulation tasks: Reach, Push, Pick-Place, and Drawer-Open.
- Evaluation metrics: Task Success Rate (%) over 3 random seeds (seeds 42, 100, 2026), Latent MSE across horizons $k \in \{1, 3, 5, 10\}$, Goal Tracking Error (cm), and Planning Latency (ms/step).
- Deliverables include full code, configs, demonstration collector, training script, evaluation script, plotting script, README, and formal technical report compiled to PDF.
- Writing style prohibitions: no em dashes, no filler, active voice, rigorous NeurIPS/CoRL scientific register.

---

### Task 1: Environment & Simulation Scaffolding (`src/env`)

**Files:**
- Create: `AML_Assignment_World-Model/src/env/camera_utils.py`
- Create: `AML_Assignment_World-Model/src/env/manipulation_env.py`
- Create: `AML_Assignment_World-Model/src/env/__init__.py`
- Test: `AML_Assignment_World-Model/tests/test_env.py`

**Interfaces:**
- Consumes: Gymnasium environment API, NumPy arrays.
- Produces: `ManipulationEnv` exposing `reset() -> (obs, info)`, `step(action) -> (obs, reward, terminated, truncated, info)`, `render() -> np.ndarray (H, W, 3)`, `get_goal_image() -> np.ndarray (H, W, 3)`.

- [ ] **Step 1: Write failing test for simulation environment**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement camera utilities and standardized manipulation environment with Reach, Push, Pick-Place, and Drawer-Open tasks**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Verify environment observation shapes (224, 224, 3) and action shapes (4,)**

---

### Task 2: Deterministic Seeding, Logging, and Benchmark Metrics (`src/utils`)

**Files:**
- Create: `AML_Assignment_World-Model/src/utils/seed.py`
- Create: `AML_Assignment_World-Model/src/utils/metrics.py`
- Create: `AML_Assignment_World-Model/src/utils/__init__.py`
- Test: `AML_Assignment_World-Model/tests/test_utils.py`

**Interfaces:**
- Consumes: `torch`, `numpy`, `random`.
- Produces: `set_seed(seed: int)`, `compute_latent_mse(z_pred, z_true)`, `compute_goal_distance(ee_pose, goal_pose)`, `aggregate_metrics(results_list)`.

- [ ] **Step 1: Write test for deterministic reproducibility and metrics calculations**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement seed utility and metric calculation module**
- [ ] **Step 4: Run test to verify it passes**

---

### Task 3: Latent Trajectory Planning via Cross-Entropy Method (`src/planning`)

**Files:**
- Create: `AML_Assignment_World-Model/src/planning/cost_functions.py`
- Create: `AML_Assignment_World-Model/src/planning/cem.py`
- Create: `AML_Assignment_World-Model/src/planning/mpc_controller.py`
- Create: `AML_Assignment_World-Model/src/planning/__init__.py`
- Test: `AML_Assignment_World-Model/tests/test_planning.py`

**Interfaces:**
- Consumes: Latent representations $(B, N, D)$, action sequences $(B, H, d_a)$, goal latent $z_g$.
- Produces: Vectorized `CEMPlanner.plan(initial_latent, goal_latent, dynamics_fn) -> torch.Tensor (H, d_a)`, `LatentMPCController.step(obs, goal_obs) -> action`.

- [ ] **Step 1: Write test for CEM planning algorithm and latent cost functions**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement cosine and L1 cost functions, vectorized CEM optimizer, and receding horizon controller**
- [ ] **Step 4: Run test to verify it passes**

---

### Task 4: World Model Architectures and Baselines (`src/models`)

**Files:**
- Create: `AML_Assignment_World-Model/src/models/vjepa_ac_interface.py`
- Create: `AML_Assignment_World-Model/src/models/dino_wm.py`
- Create: `AML_Assignment_World-Model/src/models/tdmpc2_model.py`
- Create: `AML_Assignment_World-Model/src/models/bc_policy.py`
- Create: `AML_Assignment_World-Model/src/models/__init__.py`
- Test: `AML_Assignment_World-Model/tests/test_models.py`

**Interfaces:**
- Consumes: Image tensor $(B, C, H, W)$ or trajectory $(B, T, C, H, W)$, action tensor $(B, T, d_a)$.
- Produces: Uniform model interface with `encode(obs)`, `predict(latent, action)`, `compute_loss(obs_seq, action_seq)`.

- [ ] **Step 1: Write test verifying input/output contracts for all 4 models**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement V-JEPA 2-AC, DINO-WM, TD-MPC2, and BC Policy models**
- [ ] **Step 4: Run test to verify it passes**

---

### Task 5: Trajectory Data Collection & Preprocessing Engine (`scripts/collect_demonstrations.py`)

**Files:**
- Create: `AML_Assignment_World-Model/scripts/collect_demonstrations.py`
- Create: `AML_Assignment_World-Model/configs/env/reach.yaml`
- Create: `AML_Assignment_World-Model/configs/env/push.yaml`
- Create: `AML_Assignment_World-Model/configs/env/pick_place.yaml`
- Create: `AML_Assignment_World-Model/configs/env/drawer_open.yaml`
- Test: `AML_Assignment_World-Model/tests/test_data_collection.py`

**Interfaces:**
- Consumes: Task name, number of episodes, image resolution, seed.
- Produces: Compressed `.npz` datasets containing observations $(N, T, 224, 224, 3)$, actions $(N, T, 4)$, states $(N, T, 7)$, rewards $(N, T)$.

- [ ] **Step 1: Write test for trajectory collection and dataset verification**
- [ ] **Step 2: Implement demonstration collector script with scripted expert controllers for 4 tasks**
- [ ] **Step 3: Run data collection for Reach, Push, Pick-Place, Drawer-Open (100 episodes each)**
- [ ] **Step 4: Verify generated dataset integrity and shapes**

---

### Task 6: Action-Conditioned World Model Training Pipeline (`scripts/train_ac_predictor.py`)

**Files:**
- Create: `AML_Assignment_World-Model/scripts/train_ac_predictor.py`
- Create: `AML_Assignment_World-Model/configs/model/vjepa2_ac.yaml`
- Create: `AML_Assignment_World-Model/configs/model/dino_wm.yaml`
- Create: `AML_Assignment_World-Model/configs/model/tdmpc2.yaml`
- Create: `AML_Assignment_World-Model/configs/model/bc_policy.yaml`

**Interfaces:**
- Consumes: Pre-collected dataset path, model configuration YAML, hyperparameters.
- Produces: Trained model checkpoints in `results/checkpoints/` and training curves in `results/logs/`.

- [ ] **Step 1: Implement unified training pipeline supporting all 4 model paradigms**
- [ ] **Step 2: Train all 4 models across the manipulation tasks with validation tracking**
- [ ] **Step 3: Save checkpoints and training metrics logs**

---

### Task 7: Closed-Loop Multi-Seed Comparative Evaluation Benchmark (`scripts/evaluate_latent_mpc.py`)

**Files:**
- Create: `AML_Assignment_World-Model/scripts/evaluate_latent_mpc.py`
- Create: `AML_Assignment_World-Model/configs/planning/cem_default.yaml`

**Interfaces:**
- Consumes: Trained checkpoints, test environments, evaluation seeds (42, 100, 2026).
- Produces: Evaluation metrics (Success Rate %, Latent MSE, Goal Distance cm, Latency ms) saved to JSON/CSV in `results/tables/`.

- [ ] **Step 1: Implement closed-loop evaluation harness with multi-seed execution**
- [ ] **Step 2: Execute benchmark across all 4 models and 4 tasks over 3 seeds**
- [ ] **Step 3: Verify output CSV and JSON files in `results/tables/`**

---

### Task 8: Quantitative Metrics Compilation, Figures, and LaTeX Tables (`scripts/generate_comparative_plots.py`)

**Files:**
- Create: `AML_Assignment_World-Model/scripts/generate_comparative_plots.py`
- Output: `AML_Assignment_World-Model/results/figures/success_rate_comparison.pdf`
- Output: `AML_Assignment_World-Model/results/figures/latent_mse_horizon.pdf`
- Output: `AML_Assignment_World-Model/results/figures/latency_pareto.pdf`
- Output: `AML_Assignment_World-Model/results/figures/latent_energy_landscape.pdf`
- Output: `AML_Assignment_World-Model/results/tables/comparative_results.tex`

- [ ] **Step 1: Implement script to generate vector figures and publication-grade LaTeX tables**
- [ ] **Step 2: Run script and verify figure generation (both PDF and PNG)**
- [ ] **Step 3: Verify LaTeX table syntax and data consistency**

---

### Task 9: Formal Technical Report & Overleaf-Compatible LaTeX Manuscript (`report/`)

**Files:**
- Create: `AML_Assignment_World-Model/report/assignment2_report.tex`
- Create: `AML_Assignment_World-Model/report/references.bib`
- Output: `AML_Assignment_World-Model/report/assignment2_report.pdf`

**Requirements:**
- Publication-grade NeurIPS/CoRL/CVPR formatting.
- Complete mathematical formulations, methodology, comparative results, ablations, failure modes, and reproducibility details.
- Adhere strictly to writing rules: no em dashes, no filler, active voice.
- Compile with `tectonic` and verify PDF output.

- [ ] **Step 1: Write comprehensive LaTeX technical report and BibTeX bibliography**
- [ ] **Step 2: Compile report using `tectonic report/assignment2_report.tex`**
- [ ] **Step 3: Verify PDF compilation and check formatting**

---

### Task 10: Complete Repository Polish & Reproducibility Documentation

**Files:**
- Modify: `AML_Assignment_World-Model/README.md`

- [ ] **Step 1: Update README.md with detailed instructions for reproducing all experiments**
- [ ] **Step 2: Run pytest across the entire test suite**
- [ ] **Step 3: Verify all deliverables required by `assignment#2.pdf` are present and complete**
