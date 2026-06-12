# TAM-xLSTM: Topology-Aware Matrix Memory LSTM for EEG Decoding

> **IEEE SMC 2026 Submission — Code Repository Private**
>
> This repository is a **public showcase** of the architecture, methodology, and benchmark results from our paper currently under double-blind peer review at **IEEE SMC 2026**. The full source code and model weights are held in a private repository and will be released upon acceptance and publication.

---

## The Problem

Decoding motor intentions from EEG signals is central to non-invasive brain-computer interfaces (BCIs) — but real-world deployment is blocked by a deceptively hard problem: **a model trained on one person's brain rarely works on another's**.

Inter-subject variability in EEG is enormous. Electrode placement shifts, individual anatomy, neural strategy differences, and hardware inconsistencies all conspire to make cross-subject generalisation the field's central unsolved challenge. Most prior work sidesteps this with subject-specific fine-tuning or domain adaptation tricks — approaches that don't scale to clinical deployment.

## Our Solution: TAM-xLSTM

**TAM-xLSTM** tackles generalisation at the architecture level. Rather than patching a generic model with alignment losses, we encode what we *know* about EEG signals directly into the network design:

- The brain has **spatial topology** — motor cortex electrodes are neighbours and should be modelled as such.
- Motor-related EEG has **multiple timescales** — fast transient beta-band bursts *and* slower preparatory potentials must both be captured.
- Interpretability should be **built in**, not bolted on after training.

The result is a model that generalises across subjects, hardware types, and datasets — with gate dynamics and saliency maps that are neurophysiologically meaningful.

---

## Key Results

| Dataset | Protocol | TAM-xLSTM | Best Prior Baseline | Δ Improvement |
|---|---|---|---|---|
| **BCIC-IV-2a** (22-ch, 4-class MI, 9 subjects) | LOSO | **50.93%** | 44.80% (Zaremba et al.) | **+6.13 pp** |
| **BCIC-IV-2b** (3-ch, 2-class MI, 9 subjects) | LOSO | **70.78%** | 70.53% (DS-KTL) | **+0.25 pp** |
| **BCIC-IV-2b** | Cross-subject | **65.96%** | — | — |
| **Reach & Grasp — Gel** (45 subjects) | Cross-subject | **64.80 ± 1.92%** | — | — |
| **Reach & Grasp — Water** | Cross-subject | **63.58 ± 4.07%** | — | — |
| **Reach & Grasp — Dry** | Cross-subject | **63.24 ± 2.69%** | — | — |

> **Hardware robustness highlight:** Consistent cross-subject accuracy across three radically different electrode types (Gel, Water, Dry) with less than **1.6 percentage points** spread — demonstrating that the model is not brittle to signal-acquisition variability.

> *LOSO = Leave-One-Subject-Out. Cross-subject splits use five random disjoint train/test partitions.*

---

## Methodology

### 1. Graph Spectral Filtering — Encoding Electrode Topology

Standard deep learning models treat EEG channels as an unordered set. TAM-xLSTM instead represents electrode relationships as a **graph**, with a symmetric adjacency matrix built from pairwise Euclidean distances in the standard 10–20 cap layout.

A learnable **K=2 polynomial graph filter** (inspired by Defferrard et al., NeurIPS 2016) is applied:

```
G_θ = θ₀I + θ₁L + θ₂L²
```

where `L` is the normalised graph Laplacian and `θ` coefficients are learned end-to-end. This lets the network discover whether local (one-hop) or distributed (two-hop) spatial patterns are most predictive for a given task — **without hardcoding any prior about spatial scale**.

### 2. Dual-Timescale Matrix-Memory Recurrence (xLSTM)

At the recurrent core, two **parallel TAM-mLSTM branches** run simultaneously on the same input:

- **Fast Branch (stride = 1):** Processes every timestep. High-frequency input-gate retention (0.7–0.9) with dynamic forget-gate fluctuations — designed to track rapid transient patterns like beta-band event-related desynchronisation (ERD).
- **Slow Branch (stride = 4):** Processes every 4th timestep. Smoother, lower-frequency gate evolution — designed to integrate sustained preparatory dynamics and movement-related cortical potentials.

Each branch uses a **matrix memory** `C_t ∈ ℝ^(d_h × d_h)` updated via outer-product accumulation, enabling richer associative storage than standard vector-state LSTMs. A residual projection skip from the shared input features stabilises training.

The two streams are fused via linear–GELU–Dropout projection before the final classifier — **capturing both what changed and what persisted** in the neural signal.

### 3. Learnable Attention Pooling — Interpretability Without Post-Hoc Analysis

Rather than using the final hidden state, all timestep outputs are aggregated via **learned attention pooling**:

```
e_t = w_a^T · tanh(W_a · h_t)
α_t = softmax(e_t)
z   = Σ α_t · h_t
```

The softmax weights `α_t` are temporal importance scores that can be directly visualised as saliency maps. This means interpretability is a **native output** of the forward pass — no gradient-based attribution or surrogate analysis required.

### 4. Supporting Components

- **Squeeze-and-Excitation Channel Attention:** Per-channel importance weights learned from global temporal averages, allowing the model to suppress irrelevant electrodes.
- **Multi-Scale Temporal Front-End:** Parallel 1-D convolutions with kernels `{5, 15, 31}` extract features at β (~20 Hz), α (~10 Hz), and θ/δ (~4 Hz) scales before the recurrent layers.
- **Training Regularisation:** Label smoothing (ε = 0.1), AdamW with cosine-annealing LR, gradient clipping (max norm 1.0), and mixup augmentation (α = 0.1).

---

## Architecture Overview

```
Raw EEG (C × T)
      │
      ▼
Bandpass Filter (0.5–40 Hz)
      │
      ▼
Graph Spectral Filter  ──►  Channel (SE) Attention
                                    │
                                    ▼
             Multi-Scale Temporal Convolution (k=5, 15, 31)
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
             Fast TAM-mLSTM               Slow TAM-mLSTM
              (stride = 1)                (stride = 4)
             Attention Pool              Attention Pool
                      │                           │
                      └─────────────┬─────────────┘
                                    ▼
                             Fusion (2d_h → d_h)
                                    │
                                    ▼
                           2-Layer MLP + Softmax
                                    │
                                    ▼
                         Class Probabilities (N_c)
```

---

## Visual Evidence

### Architecture Pipeline

> <img width="1448" height="349" alt="architecture (1)" src="https://github.com/user-attachments/assets/b0e151e1-2442-4a96-bfab-24541c53ea08" />
>
> *The complete end-to-end TAM-xLSTM pipeline: raw EEG passes through bandpass filtering and graph spectral encoding, followed by squeeze-and-excitation channel reweighting, multi-scale convolution, dual-timescale recurrent branches, and attention pooling before the final classifier. Residual projection paths connect the shared input features to each LSTM branch for training stability.*

---

### Spatial Interpretability — Channel Importance

> <img width="1247" height="822" alt="channel_importance" src="https://github.com/user-attachments/assets/64896ee2-808a-4d5b-9258-4d9a46c8a243" />


>
> *Channel-level importance weights learned from checkpoint parameters. Motor cortex channels (CP4, C4, Cz) receive significantly higher importance weights than non-motor regions. Quantitatively: **motor cortex mean importance = 0.603** vs. **non-motor mean = 0.397** — a near-60/40 split consistent with the known neurophysiology of motor imagery, achieved without any anatomical supervision signal.*

---

### Temporal Interpretability — Saliency Map

> <img width="2084" height="882" alt="temporal_saliency" src="https://github.com/user-attachments/assets/e6e1279c-61fb-4493-902f-b6c03a48ce29" />

>
> *Attention-derived temporal saliency over a representative motor imagery trial. A sharp peak at **cue onset (~0.5 s)** confirms rapid response to the imagery trigger. Saliency then rises and sustains at **0.8–1.0** through the motor imagery period, peaking near **t = 2.42 s**, before declining post-imagery. The model concentrates discriminative computation in the physiologically relevant window — not uniformly across the trial — with no supervision on timing.*

---

### Dual-Timescale Gate Dynamics

> <img width="2082" height="1197" alt="gate_dynamics" src="https://github.com/user-attachments/assets/eb367f8c-af6a-4fa3-a181-82b622f789fa" />
>
> *Temporal evolution of input and forget gates for the fast (stride = 1) and slow (stride = 4) branches. The fast branch shows high-frequency, volatile gate activity tracking transient EEG fluctuations. The slow branch shows a smooth, gradual trajectory integrating sustained contextual dynamics. This complementary behaviour is an emergent property of the dual-stride design — the two branches do not redundantly encode the same information.*

---

## Implementation Details

| Parameter | Value |
|---|---|
| Hidden dimension `d_h` | 128 (BCIC) / 256 (Reach & Grasp) |
| Graph filter order `K` | 2 |
| Slow stride `s` | 4 |
| Temporal kernel sizes | 5, 15, 31 |
| Dropout | 0.5 |
| Learning rate (initial) | 1 × 10⁻⁴ |
| Optimizer | AdamW |
| LR schedule | Cosine annealing |
| Gradient clip (max norm) | 1.0 |
| Label smoothing `ε` | 0.1 |
| Batch size | 32–64 |
| Max epochs | 100 |
| Early stopping patience | 30 |

**Hardware:** Single NVIDIA Quadro RTX 5000 GPU · CUDA 12.1 · PyTorch 2.1.0

---

## Datasets

| Dataset | Subjects | Channels | Fs | Task | Classes |
|---|---|---|---|---|---|
| BCIC-IV-2a | 9 | 22 | 250 Hz | Motor imagery (LH / RH / Feet) | 3 |
| BCIC-IV-2b | 9 | 3 | 250 Hz | Motor imagery (LH / RH) | 2 |
| Reach & Grasp (BNCI #27) | 45 (15 × 3 electrode types) | Variable | 256 Hz | Reach & grasp movements | 3 |

---

## Limitations & Future Work

This work is an honest step forward, not a final answer. Current limitations include:

- **No ablation study** isolating individual components (graph filter, dual-stride, attention pooling) — planned for the camera-ready version.
- **Protocol heterogeneity** across datasets makes direct competitive comparison on BCIC-IV-2b and Reach & Grasp difficult; reported figures should be treated as reference benchmarks rather than strictly competitive results.
- Future directions include subject-adaptive fine-tuning strategies, class-balanced sampling, and extension to higher-density EEG montages.

---

## Citation

> *This paper is currently under double-blind peer review. The citation block below will be updated upon acceptance.*

```bibtex
@inproceedings{shrivastava2026tamxlstm,
  title     = {TAM-xLSTM: Topology-Aware Matrix Memory LSTM for EEG Decoding},
  author    = {Shrivastava, Soham and Shukla, Shivansh and {Reddy N}, Gowtham and Meena, Yogesh Kumar},
  booktitle = {Proceedings of the IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  year      = {2026},
  note      = {Under review}
}
```

---

## Contact

This project is affiliated with the **Human-AI Interaction (HAIx) Lab, IIT Gandhinagar, India**.

For questions about the methodology, potential collaborations, or early access to the codebase for academic purposes, feel free to reach out:
- **Shivansh Shukla** (First Author) - [[LinkedIn]](https://www.linkedin.com/in/shivansh-shukla-00242028b/) . [Email](mailto:shivansh.shukla@iitgn.ac.in)
- **Soham Shrivastava** (First Author) — [[LinkedIn]](https://www.linkedin.com/in/soham-shrivastava) · [Email](mailto:soham.shrivastava@iitgn.ac.in)
- **Prof. Yogesh Kumar Meena** (Corresponding Author) — [yk.meena@iitgn.ac.in](mailto:yk.meena@iitgn.ac.in)

> *Recruiters and industry researchers are welcome to reach out for technical discussions. A live demo and full codebase will be published upon paper acceptance.*

---

<p align="center">
  <sub>© 2026 HAIx Lab, IIT Gandhinagar · Submitted to IEEE SMC 2026 · Code release pending publication</sub>
</p>
