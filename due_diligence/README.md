This folder contains a implementation to train transformer-based models using the HuggingFace Trainer for sequence classification.

# Data

The dataset needs to be copied from rijsbergen under `/data/volume_2/david/due_diligence/data/due_diligence_kira.hf` to `data_fintech/due_diligence_kira.hf` in this folder.


# Train on Snellius

In order to train the model go to the folder `slurm` and submit the training script via:

```
sbatch gpu train.sh
```

This will train a bert-base-uncased model for max. 10,000 steps using batch size 32, half-precision learning rate 2e-5, weight decay 0.01 and evalaute the model every 200 steps.
