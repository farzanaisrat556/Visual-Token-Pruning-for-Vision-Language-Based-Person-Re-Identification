# Visual Token Pruning for Vision-Language-Based Person Re-Identification

An experimental Person Re-Identification (Re-ID) pipeline combining a ResNet-50 retrieval model with Qwen2-VL-2B, LoRA fine-tuning, teacher-guided knowledge distillation, and visual-token pruning.

> **Project status:** Experimental research prototype  
> **Dataset:** Market-1501  
> **Primary environment:** Google Colab with an NVIDIA Tesla T4 GPU  
> **Evaluation scope:** 200-query subset of Market-1501, with a teacher-generated shortlist of 10 gallery candidates for VLM reranking

This repository documents an experimental investigation—not a state-of-the-art claim. The reported results are limited to the evaluation protocol and settings described below.

---

## Table of contents

- [Overview](#overview)
- [Research questions](#research-questions)
- [Pipeline](#pipeline)
- [Methods](#methods)
- [Dataset and protocol](#dataset-and-protocol)
- [Experimental results](#experimental-results)
- [Interpretation and limitations](#interpretation-and-limitations)
- [Repository contents](#repository-contents)
- [Environment and installation](#environment-and-installation)
- [Running the notebook](#running-the-notebook)
- [Gradio demo](#gradio-demo)
- [Reproducibility notes](#reproducibility-notes)
- [Responsible use and privacy](#responsible-use-and-privacy)
- [References](#references)
- [License](#license)

---

## Overview

Person Re-Identification aims to retrieve images of the same person across different camera views. Given a query pedestrian image and a gallery of candidate images, a Re-ID system ranks gallery images by their likelihood of depicting the same identity.

This project explores a two-stage retrieval and verification approach:

1. A ResNet-50 Re-ID model extracts visual embeddings and creates an initial ranking of gallery images.
2. Qwen2-VL-2B acts as a pairwise same-person verifier and reranks a shortlist of candidates.
3. Qwen2-VL is adapted with Low-Rank Adaptation (LoRA), using both hard labels and teacher-generated soft targets.
4. Visual-token pruning is applied to the VLM input to investigate the trade-off between inference cost and retrieval quality.

The project compares dense visual-token processing with random and diversity-oriented token-selection strategies at different token budgets.

## Research questions

The experiments investigate the following questions:

- Can a vision-language model serve as a pairwise verifier in a teacher-assisted Person Re-ID pipeline?
- Does LoRA fine-tuning improve Qwen2-VL's verification performance over zero-shot inference?
- Does teacher-guided knowledge distillation improve the fine-tuned verifier?
- How do visual-token selection strategies and token budgets affect Rank-1 accuracy, mAP, latency, memory use, and decision consistency?
- What accuracy–efficiency trade-offs appear when visual tokens are removed before the language-model portion of the VLM?

---

## Pipeline

```text
                    Market-1501
                         |
             +-----------+-----------+
             |                       |
       Training split          Query / Gallery
             |                       |
       ResNet-50 Re-ID                |
         teachers                    |
        /          \                  |
 Teacher A       Teacher B            |
    |                |                |
 Soft targets     Gallery shortlist  |
    |                |                |
    +------ LoRA-trained Qwen2-VL <---+
                 verifier
                     |
             Visual-token selection
                     |
        Dense / Random / Diversity /
          Difference-based pruning
                     |
             VLM candidate scores
                     |
          Rerank teacher shortlist
                     |
             Rank-1 and mAP
```

### Stage 1: ResNet-50 retrieval teachers

A ResNet-50-based Re-ID network is trained using identity classification and a batch-hard-style metric-learning objective. The notebook uses two teacher models for different roles:

- **Teacher A:** trained on 70% of the training identities and used to produce soft verification targets for identities not used to train this teacher.
- **Teacher B:** trained on all training identities and used to generate the gallery shortlist during retrieval evaluation and in the demo.

The notebook uses ImageNet-pretrained ResNet-50 weights and a batch construction strategy that samples multiple identities and multiple images per identity.

### Stage 2: Qwen2-VL pairwise verification

Qwen2-VL-2B-Instruct receives two pedestrian images and a prompt asking whether they show the same person. The notebook uses the difference between the `Yes` and `No` token logits as the verification score:

```text
verification score = logit(Yes) - logit(No)
```

A higher score indicates stronger model preference for the same-person answer.

### Stage 3: LoRA fine-tuning

The notebook adapts Qwen2-VL with LoRA rather than updating all model parameters. It evaluates two training targets:

- **Hard-label training:** uses binary same-person / different-person labels.
- **Distillation training:** uses calibrated soft targets generated by Teacher A.

The student training pairs are created from identities outside Teacher A's training identity set. This is intended to reduce direct identity overlap between the teacher's training data and the student-pair targets.

### Stage 4: Visual-token pruning

Each resized image produces 128 visual tokens in the notebook's configured image resolution. A pair therefore contains 256 visual tokens before pruning.

The notebook evaluates the following strategies:

| Strategy | Description |
|---|---|
| `random_indep` | Randomly selects tokens independently for each image |
| `random_shared` | Uses shared token indices for the image pair |
| `div_indep` | Diversity-oriented selection performed independently for each image |
| `div_shared` | Diversity-oriented selection using shared indices |
| `diff_shared` | Difference-oriented shared-index selection |
| `dense` | Keeps all visual tokens; no pruning |

The evaluated token budgets are 75%, 50%, and 25% of the original visual tokens. The pruning operation is applied **after the vision encoder**. It reduces the visual-token sequence passed into the language-model portion, but does not reduce the vision encoder's computation.

---

## Dataset and protocol

The notebook uses the **Market-1501** person Re-ID dataset. It downloads the dataset through `kagglehub` using the dataset identifier `pengcw1/market-1501`.

The notebook output reports:

| Split | Images |
|---|---:|
| Training | 12,936 |
| Gallery | 19,732 |
| Query | 3,368 |

The current experiment uses:

- 200 query images (`N_QUERIES = 200`)
- A teacher-generated shortlist of 10 gallery candidates (`SHORTLIST_K = 10`)
- One random seed for the random-pruning configurations (`RANDOM_SEEDS = [0]`)
- A Tesla T4 GPU in the recorded Colab run

**Important evaluation detail:** the VLM does not independently score every gallery image in the reported experiment. Teacher B first ranks the gallery; Qwen2-VL reranks the top 10 candidates, and the remaining candidates retain their teacher ranking. Therefore, the VLM results describe a teacher-assisted reranking pipeline, not standalone full-gallery Qwen2-VL retrieval.

The notebook includes an optional full-query evaluation section, but the recorded run has `RUN_FINAL = False`. The results below are therefore from the 200-query subset, not all 3,368 query images.

---

## Experimental results

### Main model comparison

Recorded results on the 200-query subset:

| Configuration | Rank-1 (%) | mAP (%) |
|---|---:|---:|
| ResNet-50 teacher retrieval baseline | 92.0 | 76.52 |
| Qwen2-VL zero-shot, dense | 81.5 | 72.538 |
| Qwen2-VL LoRA, hard labels, dense | 82.5 | 74.177 |
| Qwen2-VL LoRA, distilled, dense | 83.0 | 74.2 |

The fine-tuned Qwen2-VL variants improve on zero-shot Qwen2-VL in this run. The ResNet-50 teacher remains stronger on both reported retrieval metrics.

### Visual-token pruning results

Results for the distilled Qwen2-VL verifier:

| Method | Token budget | Rank-1 (%) | mAP (%) | LLM time / pair (ms) | LLM speedup | Decision flip vs. dense |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 100% | 83.0 | 74.2 | 156 | 1.00× | — |
| Random independent | 75% | 81.0 | 73.408 | 78 | 2.00× | 4.2% |
| Random shared | 75% | 82.5 | 74.054 | 78 | 1.99× | 4.0% |
| Diversity independent | 75% | 84.0 | 74.213 | 105 | 1.49× | 5.3% |
| Diversity shared | 75% | 84.0 | 74.102 | 92 | 1.70× | 4.3% |
| Difference shared | 75% | 83.0 | 73.987 | 77 | 2.04× | 4.0% |
| Random independent | 50% | 82.0 | 73.294 | 79 | 1.98× | 9.2% |
| Random shared | 50% | 83.5 | 73.051 | 78 | 2.00× | 9.4% |
| Diversity independent | 50% | 83.5 | 73.484 | 99 | 1.58× | 11.4% |
| **Diversity shared** | **50%** | **85.5** | **73.6** | **90** | **1.74×** | **10.0%** |
| Difference shared | 50% | 83.0 | 73.081 | 79 | 1.97× | 12.4% |
| Random independent | 25% | 82.0 | 71.248 | 79 | 1.98× | 18.7% |
| Random shared | 25% | 82.5 | 72.157 | 79 | 1.98× | 17.2% |
| Diversity independent | 25% | 82.0 | 70.792 | 88 | 1.77× | 20.7% |
| Diversity shared | 25% | 81.0 | 71.316 | 83 | 1.89× | 19.5% |
| Difference shared | 25% | 84.0 | 71.520 | 77 | 2.02× | 24.3% |

The table reports the notebook's recorded measurements. Timing is hardware- and implementation-dependent and should not be treated as a universal speed guarantee.

### What the results suggest

- LoRA fine-tuning improves the Qwen2-VL verifier over its zero-shot result in this experiment.
- Teacher distillation gives a small improvement over hard-label fine-tuning in Rank-1, while mAP is effectively similar at the displayed precision.
- The 50% `div_shared` configuration obtains the highest observed Rank-1 score among the reported pruning configurations (85.5%), but its mAP (73.6%) is below the dense distilled model's 74.2%.
- At the 25% budget, mAP generally decreases and decision-flip rates increase, suggesting that aggressive pruning can remove useful visual information.
- Random pruning has low measured LLM time in this run, while diversity-based methods can take longer because token selection itself has computational cost.

The 50% `div_shared` result is an observed result on this subset and configuration. It should not be interpreted as proof that pruning improves retrieval generally.

---

## Interpretation and limitations

This project is an experimental prototype. The following limitations are important when interpreting or reusing the results.

1. **Subset evaluation:** the recorded evaluation uses 200 of the 3,368 Market-1501 query images. The optional full-query run is disabled in the saved notebook (`RUN_FINAL = False`).
2. **Teacher-assisted reranking:** Qwen2-VL reranks only the teacher's top-10 shortlist. It is not evaluated as an independent full-gallery retrieval model in these reported results.
3. **Limited random-seed coverage:** random pruning is recorded with one seed. Repeated runs with multiple seeds are needed to estimate variability more reliably.
4. **No claim of state-of-the-art performance:** published benchmark numbers in the notebook use a different evaluation scope and/or training protocol. They are not directly comparable with this 200-query subset experiment. Verify original papers and reproduce matching protocols before making comparative claims.
5. **Pruning location:** token pruning occurs after the vision encoder. It reduces the sequence processed by the language-model component, not the vision encoder's computation.
6. **Latency scope:** the notebook reports LLM time per pair and estimates total time by adding two vision-encoder passes. End-to-end deployment latency can differ due to image loading, preprocessing, caching, shortlist generation, batching, and hardware.
7. **Dataset and domain limits:** results on Market-1501 do not establish performance in other camera networks, locations, populations, or operational environments.
8. **Identity and appearance variation:** occlusion, lighting, pose, camera differences, and visually similar clothing can affect retrieval and verification.
9. **Prototype demo:** the Gradio interface demonstrates the workflow; it is not evidence of production readiness or reliable real-world identity verification.

### Future work

- Run the full 3,368-query evaluation.
- Repeat random-pruning experiments across multiple seeds.
- Evaluate additional shortlist sizes and report recall of the teacher shortlist.
- Compare pruning strategies under matched latency and token budgets.
- Profile end-to-end latency, including the vision encoder and teacher retrieval stage.
- Evaluate on additional datasets and cross-domain settings.
- Investigate calibration and failure cases for pairwise verification.
- Add stronger privacy safeguards and a controlled evaluation protocol.

---

## Repository contents

A suggested repository layout:

```text
Visual-Token-Pruning-Person-ReID/
├── README.md
├── pruneprobe_reid_resumable_app(1).ipynb
├── results/
│   ├── results_ablation_q200_k10.csv
│   └── pruning_curves_q200_k10.png
├── demo/
│   └── screenshots/
├── requirements.txt
├── .gitignore
└── LICENSE
```

Only include result files that you have actually exported from the notebook. The names above are suggested organization; adjust them to match the files you upload.

Do not commit large model checkpoints or the Market-1501 image files to this repository. Use the dataset's official access route and document checkpoint acquisition separately if you decide to share trained adapters.

---

## Environment and installation

The recorded notebook installs the following packages:

```bash
pip install "transformers==4.51.3" "peft==0.15.2" accelerate kagglehub scikit-learn pandas matplotlib "gradio>=5,<6"
```

The notebook also relies on PyTorch, Torchvision, NumPy, Pillow, and tqdm. In Google Colab these may already be installed. For a reproducible environment, record the exact versions from the runtime in a `requirements.txt` file after a successful run.

### Hardware

The recorded experiment ran on:

- Google Colab
- NVIDIA Tesla T4 GPU
- CUDA-enabled PyTorch
- FP16 inference on the T4

A GPU with sufficient memory is required for the Qwen2-VL-2B workflow. Runtime, memory requirements, and package compatibility may differ on other hardware.

---

## Running the notebook

The notebook is designed for Google Colab and uses Google Drive for resumable checkpoints.

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another compatible GPU).
3. Make sure you have access to the Market-1501 dataset through the configured KaggleHub route.
4. Run the setup and data sections.
5. Allow the ResNet-50 teacher stages to finish or resume from saved checkpoints.
6. Run Qwen2-VL loading, pair generation, and LoRA training.
7. Run the evaluation and pruning grid.
8. Inspect the exported CSV tables and pruning plots.
9. Run the Gradio demo cell if you want to test the interface.

The notebook sets `USE_DRIVE = True` and mounts Google Drive. If you do not want to use Drive, change the configuration and work directory accordingly. The notebook also asserts that CUDA is available.

### Full-query evaluation

The notebook includes an optional full-query evaluation:

```python
RUN_FINAL = False
FINAL_CONFIGS = [
    ("dense", 1.0),
    ("div_shared", 0.5),
]
```

Set `RUN_FINAL = True` only when you have sufficient GPU time and have checked that the required models and checkpoints are available. This evaluates the dense distilled model and the 50%-token `div_shared` configuration over all 3,368 queries. Do not describe this full evaluation as completed unless you run it and record the resulting outputs.

---

## Gradio demo

The notebook contains a Gradio application with two tabs:

- **Verify two images:** submit two pedestrian images and receive the VLM's same-person score, teacher cosine similarity, selected pruning method, token budget, and measured LLM time.
- **Search a gallery:** submit a query image and gallery images, optionally use the teacher prefilter, and inspect the ranked results and scores.

The recorded notebook generated a temporary `gradio.live` share URL. Such URLs expire and should not be treated as permanent project hosting. Run the demo cell again to create a new temporary session, or deploy it separately after reviewing security, privacy, hardware, and hosting requirements.

---

## Reproducibility notes

The notebook uses a fixed main seed (`SEED = 0`) and saves intermediate checkpoints so training and evaluation can resume after a Colab interruption. Results are stored under a run-specific directory based on the query count and shortlist size.

For a new experiment, record:

- Dataset version and source
- Query count and shortlist size
- Training identities and pair-generation settings
- Model and adapter checkpoints
- Token-selection method and budget
- Random seed(s)
- GPU type and package versions
- Evaluation metrics and confidence intervals

The saved outputs reflect the particular run in the notebook. Re-running the notebook may produce different timings or small numerical differences due to hardware, software versions, and nondeterministic operations.

---

## Responsible use and privacy

Person Re-Identification is a biometric and surveillance-adjacent technology. This repository is intended for controlled academic experimentation and model analysis.

- Do not use this prototype to identify, track, monitor, or make decisions about people without a lawful basis, appropriate authorization, and meaningful safeguards.
- Do not upload private, sensitive, or non-consensual images to the demo.
- Treat model scores as uncertain predictions, not verified identity claims.
- Consider privacy, consent, bias, security, and applicable laws before any real-world use.
- Do not present the demo as a production-ready identification system.

---

## References

- **Market-1501:** Zheng et al., *Scalable Person Re-identification: A Benchmark*, IEEE International Conference on Computer Vision (ICCV), 2015.
- **Qwen2-VL:** Qwen Team, *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution*, 2024. [Paper](https://arxiv.org/abs/2409.12191)
- **LoRA:** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR, 2022. [Paper](https://arxiv.org/abs/2106.09685)
- **Knowledge distillation:** Hinton, Vinyals, and Dean, *Distilling the Knowledge in a Neural Network*, 2015. [Paper](https://arxiv.org/abs/1503.02531)
- **ResNet:** He et al., *Deep Residual Learning for Image Recognition*, CVPR, 2016. [Paper](https://arxiv.org/abs/1512.03385)

Please verify dataset access terms and cite the original dataset and model papers in any academic report or publication based on this work.

---

## License

Choose and add a license only after checking the licenses and terms of the code, pretrained models, adapters, and dataset you use. A repository license does not override third-party model or dataset terms.
