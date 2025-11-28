# Sparse LoRA — ECE685 Final Project

This folder contains the implementation and experimental results for the Sparse LoRA component of our group project.\
It extends standard LoRA by introducing L1 regularization and magnitude pruning to create sparse low-rank adaptation matrices, reducing trainable parameters and improving efficiency.


### Overview
Sparse LoRA modifies standard LoRA by:
1. Adding L1 penalty on LoRA weights -- encourages many LoRA parameters to shrink toward zero
2. Performing magnitude-based pruning -- after training, the smallest-magnitude LoRA weights are zeroed out, creating sparse matrices
3. Evaluating multiple LoRA ranks as below and compare accuracy, F1-score, sparsity, and time efficiency
  - r = 2
  - r = 4
  - r = 8
  - r = 16
4. Measuring time efficiency: 
  - total training time
  - time per epoch
  - inference latency (ms/sample)


### Dataset
SST-2 (8:1:1 split)


### Workflow
- load DistilBERT
- insert LoRA adapters
- add L1 term to loss
- train with hugggingface Trainer
- evaluate (accuracy, precision, recall, F1)
- prune LoRA weights
- re-evaluate
- measure time efficiency
- save metrics


### Result Summary Table

| **Rank (r)** | **Trainable Params** | **Ratio** | **Val F1** | **Val Acc** | **Test F1** | **Test Acc** | **Sparsity (<1e−3)** |
| ------------ | -------------------- | --------- | ---------- | ----------- | ----------- | ------------ | -------------------- |
| **2**        | 665,858 / 67.6M      | 0.98%     | 0.9072     | 0.9004      | 0.9028      | 0.8915       | 15.55%               |
| **4**        | 739,586 / 67.7M      | 1.09%     | 0.9135     | 0.9068      | 0.9064      | 0.8950       | 21.52%               |
| **8**        | 887,042 / 67.8M      | 1.31%     | 0.9178     | 0.9121      | 0.9122      | 0.9017       | 27.88%               |
| **16**       | 1,181,954 / 68.1M    | 1.73%     | 0.9224     | 0.9170      | 0.9195      | 0.9097       | 35.68%               |

