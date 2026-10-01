# Action-Conditioned V-JEPA (V-JEPA-AC) for Latent World Modeling in Robot Manipulation

[![Course](https://img.shields.io/badge/Course-Advanced%20Machine%20Learning-blue.svg)](https://github.com/s-elim/AML_Assignment_World-Model)
[![Assignment](https://img.shields.io/badge/Assignment-02%20Model%20Evaluation-green.svg)](assignment%232.pdf)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

---

## 1. Overview and Scope

This repository hosts the implementation, experimental configurations, and comparative evaluation framework for **Assignment 2** in Advanced Machine Learning. Consistent with the foundation established in Assignment 1 (*"Learned world models used as pretrained foundation models for robot control"*), this project adopts **Setting 3: Simulator-Based Evaluation** to implement and benchmark modern visual world models.

The primary model under investigation is **Action-Conditioned V-JEPA (V-JEPA 2-AC)**, an action-conditioned latent dynamics model built on the Joint-Embedding Predictive Architecture. Unlike generative world models that reconstruct raw pixels (e.g., DreamerV3, DayDreamer) or policy-only architectures that lack predictive lookahead, V-JEPA-AC learns predictive representations directly in the feature space of a frozen self-supervised video transformer. Planning operates by running Model Predictive Control (Cross-Entropy Method) directly on latent trajectories to minimize distance to a visual goal embedding.

```
                    +------------------------------------------+
                    |        Pretrained Video Encoder          |
Observation o_t --->|  s_phi(o_t) [Frozen ViT-H/16 or ViT-L/16]|---> Latent State z_t
                    +------------------------------------------+        |
                                                                        v
Planned Action a_t:t+k-1 -----------------------------------------> [ Causal AC Predictor ]
                                                                        | f_theta(z_t, a)
                                                                        v
                                                                Predicted State z_hat_{t+k}
                                                                        |
                                         Target z_{t+k} = sg(s_phi(o_{t+k}))
                                                                        |
                                         Loss: || z_hat_{t+k} - z_{t+k} ||^2
```

---

## 2. Mathematical Formulation

### 2.1 Latent Prediction Objective Without Reconstruction

Let $o_t \in \mathbb{R}^{H \times W \times 3}$ denote the visual observation at time $t$, and $a_t \in \mathcal{A} \subset \mathbb{R}^{d_a}$ denote the continuous robot action. The frozen or exponential-moving-average visual encoder $s_\phi: \mathcal{O} \to \mathcal{Z}$ maps input frames into patch token embeddings $z_t = s_\phi(o_t) \in \mathbb{R}^{N \times D}$.

The action-conditioned predictor $f_\theta$ is a causal transformer parameterized by $\theta$. Given past visual context $z_{\le t}$ and a planned action sequence $a_{t:t+k-1}$, the predictor estimates future latent tokens without decoding back to pixel space:

$$\hat{z}_{t+k} = f_\theta(z_{\le t}, a_{t:t+k-1})$$

The training loss is an $L_2$ error in representation space with a stop-gradient operator $\text{sg}(\cdot)$ on the target representation:

$$\mathcal{L}_{\text{JEPA}}(\theta) = \mathbb{E}_{(o, a) \sim \mathcal{D}} \left[ \left\| f_\theta(s_\phi(o_{\le t}), a_{t:t+k-1}) - \text{sg}\left(s_\phi(o_{t+k})\right) \right\|_2^2 \right]$$

Eliminating pixel reconstruction eliminates the allocation of model capacity to task-irrelevant visual background distractors, while retaining geometric and semantic scene structure.

### 2.2 Latent Goal-Directed Planning via Model Predictive Control (CEM)

Given a current visual observation $o_t$ and a target goal image $o_g$, the robot infers actions by optimizing trajectory sequences over planning horizon $H$:

$$a_{t:t+H-1}^* = \arg\min_{a_{t:t+H-1}} \mathcal{D}_{\text{cost}}\left( \hat{z}_{t+H}, z_g \right)$$

where $z_g = s_\phi(o_g)$ is the latent representation of the goal, and $\mathcal{D}_{\text{cost}}(u, v) = 1 - \frac{u^\top v}{\|u\|_2 \|v\|_2}$ measures cosine distance in latent space.

Optimization proceeds via the Cross-Entropy Method (CEM):
1. Sample $N$ candidate action sequences from a Gaussian distribution: $A^{(0)} \sim \mathcal{N}(\mu^{(0)}, \Sigma^{(0)})$.
2. Forward simulate each sequence through $f_\theta$ in parallel across batches.
3. Compute the terminal latent distance to $z_g$ for each trajectory.
4. Select the top $K$ elite sequences ($K < N$).
5. Update distribution parameters $\mu^{(m+1)}, \Sigma^{(m+1)}$ with momentum $\alpha$.
6. Repeat for $M$ iterations, executing the first action step $a_t^*$ in the simulator (receding horizon control).

---

## 3. Compared Architectures (Assignment 2 Requirement)

To satisfy the comparative mandate (implementing and comparing 3 to 4 models under identical simulation conditions), this benchmark evaluates four distinct paradigms reviewed in Assignment 1:

| Model ID | Paradigm | Feature Space | Reconstruction Loss | Planning / Action Method | Reference Paper |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V-JEPA 2-AC** (Primary) | Latent World Model | Self-supervised Video ViT | None ($L_2$ latent) | Zero-shot CEM MPC | Assran et al., Meta FAIR 2025 |
| **DINO-WM** | Latent World Model | Self-supervised DINOv2 | None ($L_2$ latent) | Zero-shot CEM MPC | Zhou et al., ICML 2025 |
| **TD-MPC2** | Task-Driven World Model | Learned Task Latent | Decoder-free (Q-value) | Task-Reward MPPI / CEM | Hansen et al., ICLR 2024 |
| **Diffusion Policy / BC** | Direct Policy Execution | Vision Encoder + MLP/DDPM | N/A (Policy only) | Open-loop closed-loop rollout | Chi et al., RSS 2023 |

### Architectural Contrast

1. **V-JEPA 2-AC vs. DINO-WM**: DINO-WM predicts next patch features frame-by-frame from static DINOv2 representations. V-JEPA 2-AC uses spatio-temporal tubelet tokens (16 frames, tubelet size 2) with 3D rotary embeddings, providing explicit motion cues over temporal windows.
2. **V-JEPA 2-AC vs. TD-MPC2**: TD-MPC2 requires explicit environmental reward supervision during dynamics learning. V-JEPA-AC learns dynamics purely self-supervised from unannotated visual-action trajectories, enabling zero-shot goal specification via goal images.
3. **V-JEPA 2-AC vs. Direct Policy**: Policy baselines do not simulate alternative counterfactuals. V-JEPA-AC evaluates 400 candidate action sequences concurrently in latent space prior to commitment.

---

## 4. Simulation Environment and Benchmark Tasks

Evaluations are conducted on standardized robotic manipulation environments using **Meta-World (v2)** and **Robosuite** (Franka Emika Panda 7-DoF arm with parallel-jaw gripper).

### Simulation Specifications

- **Simulator**: MuJoCo 3.x / Gymnasium
- **Control Rate**: 20 Hz
- **Action Space**: Continuous 4-DoF / 7-DoF Cartesian end-effector control:
  $$\Delta x, \Delta y, \Delta z, \Delta \text{yaw}, \text{gripper state} \in [-1, 1]$$
- **Observation Space**: RGB image $224 \times 224 \times 3$ captured from an off-axis third-person camera.
- **State Feedback**: End-effector Cartesian pose and gripper aperture (used by the action conditioner).

### Task Suite

1. **Reach**: Navigate the end-effector to a variable 3D goal coordinate.
2. **Push**: Slide a target block across the table surface to a marked zone.
3. **Pick-Place**: Grasp an object, lift against gravity, and transport to a target receptacle.
4. **Drawer-Open**: Engage the handle of a cabinet drawer and pull along a linear slide constraint.

---

## 5. Repository Layout

```
AML_Assignment_World-Model/
├── assignment#2.pdf                   # Official Assignment 2 task description and rubric
├── Selim_22546245 Assignment 01.docx   # Assignment 1 comprehensive literature review
├── README.md                          # Project documentation and reproducibility guide
│
├── configs/                           # Experiment configuration files
│   ├── env/
│   │   ├── metaworld_reach.yaml
│   │   ├── metaworld_push.yaml
│   │   └── metaworld_pick_place.yaml
│   ├── model/
│   │   ├── vjepa2_ac_vitl16.yaml
│   │   ├── dino_wm.yaml
│   │   └── tdmpc2.yaml
│   └── planning/
│       └── cem_default.yaml
│
├── src/                               # Core implementation
│   ├── env/                           # Simulation wrappers and vectorization
│   │   ├── metaworld_wrapper.py
│   │   └── camera_utils.py
│   ├── models/                        # Dynamics predictors and adapters
│   │   ├── vjepa_ac_interface.py
│   │   └── baseline_wrappers.py
│   ├── planning/                      # Latent trajectory optimization
│   │   ├── cem.py
│   │   ├── cost_functions.py
│   │   └── mpc_controller.py
│   └── utils/                         # Logging, seeding, metrics computation
│       ├── seed.py
│       └── metrics.py
│
├── scripts/                           # Reproducibility entrypoints
│   ├── collect_demonstrations.py      # Generate expert/scripted trajectories
│   ├── train_ac_predictor.py          # Train action-conditioned predictor
│   ├── evaluate_latent_mpc.py         # Run closed-loop evaluation
│   └── generate_comparative_plots.py  # Compile metrics tables and curves
│
├── data/                              # Offline trajectory datasets (.npz / .h5)
│   └── README.md                      # Dataset specifications and split instructions
│
├── results/                           # Evaluation outputs, tables, and trajectory plots
│   ├── figures/
│   └── tables/
│
├── V-JEPA/                            # Meta FAIR V-JEPA upstream codebase
├── V-JEPA-2/                          # Meta FAIR V-JEPA 2 upstream codebase (AC predictor)
└── JEPA/                              # Meta FAIR I-JEPA upstream codebase
```

---

## 6. Installation and Hardware Setup

### System Prerequisites

- OS: Linux (Ubuntu 22.04 LTS or compatible)
- GPU: NVIDIA GPU with CUDA 12.1+ (Minimum 16 GB VRAM for ViT-L/16 inference; 24 GB+ recommended for predictor training)
- Python: 3.10+

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/s-elim/AML_Assignment_World-Model.git
cd AML_Assignment_World-Model

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install simulation and manipulation dependencies
pip install gymnasium mujoco metaworld robosuite

# Install V-JEPA 2 core dependencies
pip install timm einops omegaconf hydra-core h5py pandas matplotlib seaborn
pip install -r V-JEPA-2/requirements.txt
```

---

## 7. Experimental Workflow and Reproducibility

### Step 1: Trajectory Dataset Generation

Collect 500 demonstrations per task using scripted expert or operational space controllers:

```bash
python scripts/collect_demonstrations.py \
    --env metaworld_push \
    --num_episodes 500 \
    --save_dir data/metaworld_push \
    --image_size 224 \
    --seed 42
```

Data is stored as compressed NumPy archives (`.npz`) containing:
- `observations`: Shape `(T, 224, 224, 3)`, uint8
- `actions`: Shape `(T, d_a)`, float32
- `states`: End-effector pose `(T, 7)`, float32
- `rewards`: Shape `(T,)`, float32

### Step 2: Training the Action-Conditioned Predictor

Train the causal AC transformer on top of frozen V-JEPA 2 representations:

```bash
python scripts/train_ac_predictor.py \
    --config configs/model/vjepa2_ac_vitl16.yaml \
    --data_dir data/metaworld_push \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --device cuda:0
```

#### Default Training Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| Encoder Backbone | ViT-L/16 (V-JEPA 2) | Frozen visual representation extractor |
| Predictor Depth | 12 Layers | Transformer blocks in AC predictor |
| Predictor Width | 768 | Hidden dimension |
| Attention Heads | 12 | Multi-head self-attention |
| Action Dimension | 4 or 7 | End-effector velocity + gripper command |
| Optimizer | AdamW | $\beta_1=0.9, \beta_2=0.95$ |
| Weight Decay | 0.05 | Decoupled weight regularization |
| Learning Rate | $1 \times 10^{-4}$ | Cosine decay schedule with 5-epoch warmup |
| Batch Size | 32 | Sequences per gradient step |
| Sequence Length ($T$) | 16 frames | Context window for temporal reasoning |

### Step 3: Closed-Loop Latent Model Predictive Control Evaluation

Evaluate the trained world model in the simulation loop across 50 test episodes with 3 random seeds:

```bash
python scripts/evaluate_latent_mpc.py \
    --env metaworld_push \
    --checkpoint results/checkpoints/vjepa2_ac_best.pt \
    --mpc_samples 400 \
    --mpc_horizon 5 \
    --mpc_iterations 10 \
    --mpc_topk 10 \
    --num_eval_episodes 50 \
    --seed 100
```

---

## 8. Evaluation Metrics and Reporting Protocol

In accordance with Assignment 2 specifications, performance is quantified across four evaluation axes:

1. **Task Success Rate (%)**: Fraction of evaluation episodes satisfying the simulator task completion criteria within the maximum horizon ($T_{\max} = 150$).
2. **Latent Prediction Error (MSE)**: Mean squared error between predicted latent token trajectories $\hat{z}_{t+k}$ and ground-truth encoded frames $s_\phi(o_{t+k})$:
   $$\text{MSE}_k = \frac{1}{N \cdot D} \sum_{i=1}^N \| \hat{z}_{t+k}^{(i)} - z_{t+k}^{(i)} \|_2^2$$
3. **Goal Tracking Error (cm)**: Final Euclidean distance between object position and target coordinates at episode termination.
4. **Planning Latency (ms/step)**: Wall-clock computation time required for CEM trajectory optimization per execution step.

### Target Comparative Results Template

| Model | Reach (SR %) | Push (SR %) | Pick-Place (SR %) | Drawer-Open (SR %) | Latent MSE ($k=5$) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct BC Policy** | -- | -- | -- | -- | N/A | < 10 |
| **TD-MPC2** | -- | -- | -- | -- | -- | ~45 |
| **DINO-WM** | -- | -- | -- | -- | -- | ~85 |
| **V-JEPA 2-AC (Ours)**| -- | -- | -- | -- | -- | ~65 |

*Note: All values will report the mean and standard deviation over 3 independent random seeds (seeds 42, 100, 2026).*

---

## 9. References and Attribution

1. **V-JEPA 2**: M. Assran, A. Bardes, D. Fan, Q. Garrido, et al., *"V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"*, Meta FAIR, 2025.
2. **V-JEPA**: A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, N. Ballas, *"Revisiting Feature Prediction for Learning Visual Representations from Video"*, arXiv:2404.08471, 2024.
3. **I-JEPA**: M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, N. Ballas, *"Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"*, CVPR, 2023.
4. **DINO-WM**: G. Zhou, H. Pan, Y. LeCun, L. Pinto, *"World Models on Pre-trained Visual Features enable Zero-shot Planning"*, ICML, 2025.
5. **TD-MPC2**: N. Hansen, H. Su, X. Wang, *"TD-MPC2: Scalable, Robust World Models for Continuous Control"*, ICLR, 2024.
6. **DreamerV3**: D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, *"Mastering Diverse Control Tasks through World Models"*, Nature, 2025.
7. **Meta-World**: T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, S. Levine, *"Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning"*, CoRL, 2019.
