# LLM-Evaluation-for-Vietnamese-NLP

## Introduction

Large Language Models (LLMs) have shown strong performance across a wide range of NLP tasks.
However, their effectiveness is highly dependent on the alignment between model architecture
and task formulation.

This project investigates the performance of different LLM architectures on Vietnamese
Question Answering (QA) and Text Generation tasks, with a focus on:

- Architectural suitability (causal LM vs encoder–decoder)
- Parameter-efficient fine-tuning using LoRA
- Practical performance differences under identical experimental conditions

## Objectives

The objectives of this project are:

1. Compare the effectiveness of different LLM architectures on Vietnamese QA and text generation.
2. Analyze the impact of task–model alignment on downstream performance.
3. Evaluate the effectiveness of LoRA as a parameter-efficient fine-tuning method.
4. Provide qualitative and quantitative insights for selecting LLMs in Vietnamese NLP systems.

## Models

| Model | Architecture | Pretraining Objective | Role in This Project |
|------|-------------|-----------------------|---------------------|
| GPT2-Vietnamese | Causal LM | Next-token prediction | Generative baseline for QA & text generation |
| ViT5-base | Encoder–Decoder (T5) | Text-to-text | Main model for QA and structured generation |

## Datasets

### 1. UIT-ViQuAD2.0 (Question Answering)
- Vietnamese extractive QA dataset
- Long context, short and factual answers
- Used for both baseline inference and fine-tuning

### 2. (Optional) Vietnamese Text Generation Dataset
- Used to evaluate free-form generation capability
- Same preprocessing and evaluation constraints applied across models

## Methodology

### Task Formulation

- Question Answering is formulated as:
  - Text completion for GPT-2
  - Input–output mapping for ViT5 (text-to-text)

### Fine-tuning Strategy

- Parameter-efficient fine-tuning using LoRA
- Only attention projection layers are updated
- Base model weights are frozen

### Experimental Control

To ensure fair comparison:
- Identical datasets and splits
- Consistent prompt format across models
- Same evaluation samples before and after fine-tuning

