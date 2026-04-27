# Looped Qwen: A Study in Recurrent Depth

This repository contains the codebase for investigating the parameter efficiency and inference-time capabilities of "looped" (recurrent) Transformer architectures, specifically applied to reasoning tasks. By sharing weights across layers—creating a deep, recurrent virtual depth from a shallow pool of unique parameters—we aim to understand if compute can be decoupled from parameter count to achieve strong reasoning capabilities.

This project is built around the Qwen 2.5 architecture (0.5B and 1.5B variants) and focuses on complex mathematical reasoning (MATH-500).

## 🎯 Research Questions

We are investigating the following core questions:

1. **Parameter Efficiency vs. Compute Depth:** Can a smaller model (e.g., 1.5B parameters) achieve the reasoning performance of a significantly larger model simply by unrolling its layers recursively at inference time (increasing "virtual depth")?
2. **Representation Collapse:** Do shared layers suffer from representation collapse or gradient vanishing when trained via standard backpropagation through time (BPTT)?
3. **Recovery Distillation:** Can we effectively map the representations of a dense, deep teacher model into a shallower, recurrent student model using continuous distillation (minimizing KL-divergence of logits and MSE of hidden states)?
4. **Reinforcement Learning Stability:** How does Group Relative Policy Optimization (GRPO) behave when updating tied parameters? Does the reward signal successfully backpropagate through the recurrent steps without destabilizing the shared layers?

## 💡 Core Hypotheses & Assumptions

*   **Hypothesis 1 (The "Thinking" Hypothesis):** Complex reasoning tasks require more sequential computation steps than simple tasks. Looped architectures naturally allow for variable computation depth, potentially enabling "test-time compute" scaling.
*   **Hypothesis 2 (Distillation as Initialization):** Training a looped model from scratch is difficult due to the optimization challenges of BPTT. We hypothesize that initializing the model via "recovery distillation" (forcing the looped model to mimic a dense teacher) provides a crucial warm-start for subsequent alignment phases.
*   **Assumption (Layer Tie-ing Structure):** We assume a specific recurrent structure. For example, a 28-layer dense model might be mapped to a 7-layer unique model, where the 7 layers are looped 4 times (virtual depth = 28).

## ✨ Advantages of this Approach

If successful, looped architectures offer significant advantages over traditional dense models:

*   **Massive VRAM Reduction:** The model's memory footprint is bound by the number of *unique* parameters. A model with a virtual depth of 28 but only 7 unique layers uses 4x less VRAM to store weights.
*   **Dynamic Inference Compute:** While not fully explored in this baseline codebase, looped models theoretically allow for dynamic halting. The model can exit the loop early for easy questions or iterate longer for hard questions.
*   **Deployment Efficiency:** Smaller parameter footprints make it feasible to deploy highly capable reasoning engines on edge devices or consumer hardware.

## 🔬 Experimental Pipeline

To rigorously test these hypotheses, this repository implements a structured 4-stage experimental pipeline:

1.  **Recovery Distillation:** Align the representations of the randomly initialized looped model to a pre-trained dense teacher.
2.  **Supervised Fine-Tuning (SFT):** Teach the model the required format and style for mathematical reasoning.
3.  **Reinforcement Learning (GRPO):** Optimize the model for accuracy and logical soundness using verifiable rewards.
4.  **Evaluation:** Rigorously evaluate the resulting model on benchmarks like MATH-500 and GSM8K.

## 🚀 Getting Started

The implementation details, environment setup, and instructions for running the complete 4-stage pipeline (including Weights & Biases tracking) are thoroughly documented in the `GUIDE.md`.

Please refer to **[`GUIDE.md`](./GUIDE.md)** for:
*   Installation instructions
*   Dataset preparation
*   Executing the Recovery, SFT, GRPO, and Eval stages
*   Configuration management
*   Troubleshooting

## 📊 Instrumentation

This codebase is heavily instrumented for research. Through Weights & Biases (`wandb`), we track not only standard loss and reward metrics but also:
*   Per-virtual-step hidden state norms (to detect representation collapse).
*   Per-physical-layer gradient norms (to monitor training stability across shared parameters).
*   Interactive prediction tables for qualitative failure analysis.
