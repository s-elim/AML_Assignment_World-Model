# Latent World Models as Robot Foundation Models

Assignment 2 (Setting 3, simulator-based evaluation) for the research area of Assignment 1:
*learned world models used as pretrained foundation models for robot control*.

Four models are trained on the same 160 LIBERO-Goal demonstrations and evaluated in the same simulator on
the same 40 held-out episodes (seed 0):

| Model | Representation | Trained here | Control |
| --- | --- | --- | --- |
| **V-JEPA 2-AC** (Assran et al., 2025) | frozen V-JEPA 2 ViT-L/16 (304M) | block-causal action-conditioned predictor, 11.4M | CEM in latent space to a goal image |
| **DINO-WM** (Zhou et al., ICML 2025) | frozen DINOv2 ViT-S/14 (21.6M) | frame-causal ViT predictor | CEM in latent space to a goal image |
| **TD-MPC2** (Hansen et al., ICLR 2024) | learned (pixels + proprioception) | encoder, latent dynamics, reward, 5 Q-functions, policy prior (5.6M) | MPPI with learned value |
| **Behaviour cloning** | ResNet-18 (ImageNet init) | end to end, Gaussian NLL | direct policy |

Behaviour-cloning variants on frozen DINOv2 and V-JEPA 2 tokens, with a fine-tuned DINOv2, and with an
added wrist camera form the representation ablation.

## Repository layout

```
configs/
  model/*.yaml              one file per trained model (all hyperparameters, seed 0)
                            (planner settings: CEMConfig in src/planning/cem.py and the evaluate.py flags)
src/
  env/libero_env.py         Gymnasium wrapper: 4 LIBERO-Goal tasks, 4 Hz macro steps, goal error in cm
  data/libero_data.py       demonstration loading, macro-step clips, state normalisation
  models/encoders.py        frozen V-JEPA 2 ViT-L/16 and DINOv2 ViT-S/14 (8x8 token grids)
  models/vjepa2_ac.py       V-JEPA 2-AC predictor, losses, cached rollout
  models/dino_wm.py         DINO-WM predictor, loss, rollout
  models/tdmpc2.py          TD-MPC2 model, update, MPPI planner
  models/bc_policy.py       behaviour-cloning policies
  planning/cem.py, costs.py cross-entropy method and latent goal costs
  agents.py                 observation -> action wrappers used in evaluation
  third_party/vjepa2/       vendored V-JEPA 2 modules (MIT, see its README)
scripts/
  build_libero_dataset.py   regenerate the 200 demonstrations (images, states, actions)
  encode_features.py        cache frozen encoder features
  train_world_model.py      train V-JEPA 2-AC or DINO-WM
  train_tdmpc2.py           train TD-MPC2 offline
  train_bc.py               train a behaviour-cloning policy
  evaluate.py               closed-loop evaluation (main runs and ablations)
  analyze_prediction.py     open-loop multi-step latent prediction error
  analyze_energy.py         planning-energy landscape versus executed outcome
  profile_latency.py        isolated latency, throughput and GPU memory per decision
  make_report_assets.py     report tables, figures, statistics and number macros
  slurm/gpu_worker.sbatch   GPU job runner used for the reported runs
tests/                      unit tests (models, planner) and a simulator integration test
results/
  logs/                     training histories (JSON) and run logs
  eval/                     per-episode evaluation records (JSON)
  tables/                   aggregated result tables
report/                     technical report sources (not tracked)
```

## 1. Setup

Python 3.11 on Linux. The cluster has four NVIDIA RTX PRO 6000 Blackwell GPUs (96 GB each); the reported runs used
one of them for training, encoding and planning, and CPU MuJoCo for simulation.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
```

**LIBERO.** Clone the official repository and download the LIBERO-Goal demonstrations:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /path/to/LIBERO
cd /path/to/LIBERO && python benchmark_scripts/download_libero_datasets.py --datasets libero_goal
```

Then `export LIBERO_ROOT=/path/to/LIBERO` (default: `./LIBERO`). The code writes LIBERO's path config to
`.libero/config.yaml` from it, so nothing else needs editing.

**Rendering.** MuJoCo renders offscreen: `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` on a GPU machine, or
`MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` with an OSMesa build on `LD_LIBRARY_PATH` on a CPU machine.

**Pretrained weights** (downloaded automatically on first use):

| Weights | Source |
| --- | --- |
| V-JEPA 2 ViT-L/16 (`vitl.pt`, target encoder) | https://dl.fbaipublicfiles.com/vjepa2/vitl.pt |
| DINOv2 ViT-S/14 | `timm/vit_small_patch14_dinov2.lvd142m` (Hugging Face) |
| ResNet-18 | `timm/resnet18.a1_in1k` (Hugging Face) |

## 2. Data

| Item | Value |
| --- | --- |
| Source | LIBERO-Goal human demonstrations (Liu et al., NeurIPS 2023 Datasets and Benchmarks), 50 per task |
| Tasks | `turn_on_the_stove`, `put_the_bowl_on_the_plate`, `push_the_plate_to_the_front_of_the_stove`, `open_the_middle_drawer_of_the_cabinet` |
| Observations | agent-view and wrist RGB 128x128, 7-D proprioception (end-effector position, axis-angle, finger opening) |
| Actions | LIBERO OSC_POSE, 7-D in [-1, 1] (position offset up to 0.05 m, rotation up to 0.5 rad, gripper) |
| Split | 40 training / 10 test demonstrations per task, fixed by seed 0 (`<task>__meta.json`); no validation split, because no checkpoint or hyperparameter is selected on held-out data (held-out losses are logged for monitoring only) |
| Preprocessing | frames re-rendered from the stored MuJoCo states with the evaluation renderer; 5 control steps per model step (4 Hz); macro action = mean of the 5 primitive actions |

```bash
for t in turn_on_the_stove put_the_bowl_on_the_plate push_the_plate_to_the_front_of_the_stove open_the_middle_drawer_of_the_cabinet; do
  python scripts/build_libero_dataset.py --task $t --out data/libero_goal4
done
```

## 3. Reproducing the experiments

All commands run from the repository root with `PYTHONPATH=.`; every script is seeded with 0.

```bash
# features from the frozen encoders
python scripts/encode_features.py --encoder vjepa2
python scripts/encode_features.py --encoder dinov2

# training (one model for all four tasks)
python scripts/train_world_model.py --config configs/model/vjepa2_ac.yaml
python scripts/train_world_model.py --config configs/model/dino_wm.yaml
python scripts/train_tdmpc2.py      --config configs/model/tdmpc2.yaml
python scripts/train_bc.py          --config configs/model/bc_resnet18.yaml
#   ablation policies: bc_resnet18_wrist, bc_dinov2_frozen, bc_vjepa2_frozen, bc_dinov2_finetune

# closed-loop evaluation: 4 tasks x 10 held-out episodes
python scripts/evaluate.py --agent vjepa2_ac   --tag vjepa2_ac --record
python scripts/evaluate.py --agent dino_wm     --tag dino_wm   --record
python scripts/evaluate.py --agent tdmpc2      --tag tdmpc2
python scripts/evaluate.py --agent bc_resnet18 --tag bc_resnet18 --record

# planner ablations (V-JEPA 2-AC), for example
python scripts/evaluate.py --agent vjepa2_ac --tag abl_H5    --horizon 5
python scripts/evaluate.py --agent vjepa2_ac --tag abl_N64   --samples 64
python scripts/evaluate.py --agent vjepa2_ac --tag abl_K0.2  --elite_frac 0.2
python scripts/evaluate.py --agent vjepa2_ac --tag abl_cos   --cost cosine

python scripts/evaluate.py --agent vjepa2_ac --tag vjepa2_ac_dataprior --action_prior data --record
python scripts/evaluate.py --agent vjepa2_ac --tag vjepa2_ac_H3 --horizon 3 --record   # post hoc, all tasks

# analyses: open-loop prediction error, energy landscapes, isolated latency profile
python scripts/analyze_prediction.py
python scripts/analyze_energy.py --agent vjepa2_ac --task turn_on_the_stove --episode 0 --grid 11
python scripts/profile_latency.py

# every table, figure and number of the report (writes report/generated/ and results/figures/)
python scripts/make_report_assets.py

# tests (the simulator test needs the dataset and MUJOCO_GL)
python -m pytest -q tests
```

## 4. Experimental configuration

The full configuration of every model is in `configs/model/`. Shared settings:

| Setting | Value |
| --- | --- |
| Seed | 0 (data split, initialisation, minibatch order, planner sampling, evaluation) |
| Input | agent-view 128x128 RGB + 7-D state; upsampled to 256x256 (V-JEPA 2) or 224x224 (DINOv2); TD-MPC2 stacks 3 frames at 64x64; BC and TD-MPC2 also receive the task identity |
| Loss | V-JEPA 2-AC: L1 teacher forcing + L1 two-step rollout; DINO-WM: MSE on the next-frame visual and proprioceptive embedding; TD-MPC2: 20 x consistency + 0.1 x reward + 0.1 x value, plus the policy-prior objective; BC: Gaussian negative log-likelihood |
| Optimiser | AdamW (world models, BC), Adam (TD-MPC2, as in the paper) |
| World-model training | 40 epochs, batch 64 clips, peak LR 4.25e-4, warm-up 5%, cosine decay, weight decay 0.04 |
| TD-MPC2 training | 40,000 updates, batch 256, LR 3e-4 (encoder x0.3), horizon 3 |
| BC training | 30 epochs, batch 64, LR 1e-4 (end to end) or 3e-4 (frozen tokens), one-cycle schedule |
| Augmentation | random shift (BC: 6 px, TD-MPC2: 4 px); none for the frozen-feature world models |
| Planner (latent models) | CEM, horizon 10 model steps (2.5 s), 256 samples, 10 iterations, 5% elites, momentum 0.15 |
| Planner (TD-MPC2) | MPPI, horizon 3, 512 samples, 64 elites, 24 policy samples, 6 iterations |
| Episode | starts at the first state of a held-out demonstration, goal frame from its final state, 300 control steps maximum |
| Metrics | success (LIBERO BDDL predicate), return (equal to success under LIBERO's sparse reward), final goal error (cm), steps to success, convergence (training and held-out curves in `results/logs/`), planning latency, peak VRAM; collisions are not instrumented in LIBERO, so no safety metric is reported |
| Software | Python 3.11.16, PyTorch 2.11.0 (CUDA 12.8), timm 1.0.30, MuJoCo 2.3.7, robosuite 1.4.0, LIBERO commit `8145de4`, NumPy 1.26.4, Gymnasium 1.4.0 |

## 5. Results

Seed 0, 10 held-out episodes per task (40 per model, identical across models). Successes per task, pooled success
rate with a Wilson 95% interval, and mean final goal error. Full tables, paired tests and figures are in the
report; per-episode records are in `results/eval/<tag>/<task>.json`.

| Model | Stove | Bowl-Plate | Push-Plate | Drawer | Success [95% CI] | Goal error (cm) |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| V-JEPA 2-AC (H=10) | 2 | 0 | 0 | 0 | 5% [1, 17] | 23.0 |
| DINO-WM (H=10) | 4 | 0 | 0 | 0 | 10% [4, 23] | 19.9 |
| TD-MPC2 | 0 | 0 | 0 | 0 | 0% [0, 9] | 31.8 |
| BC (ResNet-18, agent view) | 10 | 0 | 8 | 3 | 52% [37, 67] | 9.2 |
| BC + wrist view | 10 | 9 | 10 | 9 | 95% [83, 99] | 3.2 |
| BC, frozen DINOv2 tokens | 10 | 9 | 9 | 7 | 88% [74, 95] | 3.8 |
| BC, frozen V-JEPA 2 tokens | 9 | 7 | 9 | 0 | 62% [47, 76] | 7.5 |
| BC, fine-tuned DINOv2 | 10 | 10 | 10 | 5 | 88% [74, 95] | 4.3 |

Planner ablations (V-JEPA 2-AC, Stove, successes of 10): horizon H=3: 9, H=5: 5, H=10: 2, H=15: 3; samples
N=32/64/128/512: 1/2/1/2; elite fraction 0.1/0.2: 2/1; cosine/L2/Mahalanobis energy: 1/0/1. A data-informed action
prior gives 0/40 for both planners. Post hoc, H=3 on all tasks: V-JEPA 2-AC 9/40 and DINO-WM 5/40, every success on
Stove (0/30 on the contact-rich tasks for both).

Isolated latency per decision (one RTX PRO 6000 Blackwell): V-JEPA 2-AC 511 ms, DINO-WM 929 ms, TD-MPC2 19 ms,
BC 1.8 ms.

Key observations: the frozen DINOv2 features that give a policy 88% success give a planner 10%; on demonstrated
actions the latent predictors beat a no-change baseline at every horizon, but on the actions the planner selects
they are 25-36% worse than it; the planning energy tracks the arm rather than the manipulated object.

## 6. External code, models and data

| Resource | Use | Licence |
| --- | --- | --- |
| [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | encoder and AC-predictor modules vendored in `src/third_party/vjepa2`; `vitl.pt` weights | MIT |
| [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) via [timm](https://github.com/huggingface/pytorch-image-models) | DINOv2 ViT-S/14 weights | Apache-2.0 |
| [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) | reference for the DINO-WM predictor design (re-implemented) | MIT |
| [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) | reference for TD-MPC2 (re-implemented) | MIT |
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | simulator, BDDL tasks and demonstrations | MIT |
| [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) v1.4.0, [MuJoCo](https://github.com/google-deepmind/mujoco) 2.3.7 | physics and rendering | MIT, Apache-2.0 |

`V-JEPA/`, `V-JEPA-2/` and `JEPA/` are unmodified copies of the upstream repositories kept for reference.
AI coding assistance was used for implementation and documentation; all code was read, tested and verified.

## 7. References

1. M. Assran et al., "V-JEPA 2: Self-supervised video models enable understanding, prediction and planning," arXiv:2506.09985, 2025.
2. G. Zhou, H. Pan, Y. LeCun, L. Pinto, "DINO-WM: World models on pre-trained visual features enable zero-shot planning," ICML, 2025. arXiv:2411.04983.
3. N. Hansen, H. Su, X. Wang, "TD-MPC2: Scalable, robust world models for continuous control," ICLR, 2024. arXiv:2310.16828.
4. B. Liu et al., "LIBERO: Benchmarking knowledge transfer for lifelong robot learning," NeurIPS Datasets and Benchmarks, 2023. arXiv:2306.03310.
5. M. Oquab et al., "DINOv2: Learning robust visual features without supervision," TMLR, 2024. arXiv:2304.07193.
6. K. He, X. Zhang, S. Ren, J. Sun, "Deep residual learning for image recognition," CVPR, 2016.
