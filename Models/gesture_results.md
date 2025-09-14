# Gesture model results

This file records ensemble evaluation results and metadata for gesture-based models. Tables below provide a concise comparison across models and experiments.

## Models present

| Model file | Notes |
|---|---|
| `gesture_wlasl_mil_finetuned_mil.keras` | MIL finetuned model (saved `.keras`) |
| `gesture_wlasl_mil_finetuned_mil.weights.h5` | MIL weights (companion) |
| `gesture_wlasl_deeper_aug_os.keras` | Deeper model (oversampled, no BiLSTM) |
| `gesture_wlasl_deeper_aug_os_bilstm.keras` | Deeper model (oversampled, BiLSTM) |
| `gesture_wlasl_deeper_encinit.keras` | Deeper quick transfer-init (encoder init, no BiLSTM) |
| `gesture_wlasl_deeper_encinit.bilstm.keras` | Deeper quick transfer-init (encoder init, BiLSTM) |


## Single-run results

| Pair | mil_acc | deeper_acc | ensemble_acc | notes |
|---|---:|---:|---:|---|
| MIL + deeper_aug_os | 0.6047 | 0.2326 | 0.5581 | single-run (bag_size=32) |
| MIL + deeper_aug_os_bilstm | 0.6047 | 0.0233 | 0.5814 | single-run (bag_size=32) |


## 10-seed sweep (ensemble mean ± pstd)

| Pair | mean_acc | pstd | seeds |
|---|---:|---:|---|
| MIL + deeper_aug_os | 0.6163 | 0.0888 | [0.5581,0.7442,0.5581,0.5814,0.5116,0.6279,0.7442,0.6047,0.5116,0.6047] |
| MIL + deeper_aug_os_bilstm | 0.6640 | 0.0905 | [0.5814,0.7442,0.6047,0.6279,0.6279,0.6512,0.8372,0.6744,0.5116,0.6977] |


## Bag-size sweep (single sampling run)

| Pair | bag_size | mil_acc | deeper_acc | ensemble_acc |
|---|---:|---:|---:|---:|
| MIL + deeper_aug_os | 16 | 0.5814 | 0.3488 | 0.5814 |
| MIL + deeper_aug_os | 32 | 0.6047 | 0.2326 | 0.5581 |
| MIL + deeper_aug_os | 64 | 0.5814 | 0.2093 | 0.5349 |
| MIL + deeper_aug_os_bilstm | 16 | 0.5814 | 0.0233 | 0.5814 |
| MIL + deeper_aug_os_bilstm | 32 | 0.6047 | 0.0233 | 0.5814 |
| MIL + deeper_aug_os_bilstm | 64 | 0.5814 | 0.0233 | 0.5814 |


## Sampling-averaged ensembles (sample_runs=5, bag_size=32)

| Pair | sample_runs | mil_acc | deeper_acc | ensemble_acc |
|---|---:|---:|---:|---:|
| MIL + deeper_aug_os | 5 | 0.6047 | 0.2326 | 0.5581 |
| MIL + deeper_aug_os_bilstm | 5 | 0.6047 | 0.0233 | 0.5814 |


## Transfer-init deeper (encoder pretrain) quick runs (5 epochs)

| Model | bilstm | deeper_acc | mil_acc | ensemble_acc | notes |
|---|---:|---:|---:|---:|---|
| gesture_wlasl_deeper_encinit.keras | No | 0.5349 | 0.6047 | 0.6279 | 5-epoch quick run, weights saved |
| gesture_wlasl_deeper_encinit.bilstm.keras | Yes | 0.1395 | 0.6047 | 0.6047 | 5-epoch quick run, weights saved |

## Long retrain (30 epochs) -- pending (will be filled after runs)

| Model | bilstm | deeper_acc | mil_acc | ensemble_acc | notes |
|---|---:|---:|---:|---:|---|
| gesture_wlasl_deeper_encinit_long.keras | No | - | - | - | scheduled: 30-epoch retrain with encoder init |
| gesture_wlasl_deeper_encinit_long.bilstm.keras | Yes | - | - | - | scheduled: 30-epoch retrain with encoder init |

| gesture_wlasl_deeper_encinit_long.keras | No | 0.7442 | 0.6047 | 0.7442 | 30-epoch retrain (encoder init), sample_runs=5, bag_size=32 |
| gesture_wlasl_deeper_encinit_long.bilstm.keras | Yes | 0.6512 | 0.6047 | 0.7209 | 30-epoch retrain (encoder init, BiLSTM), sample_runs=5, bag_size=32 |

---

All experiments were run with the same clip-wise 80/20 split and bag sampling logic; bag_size refers to number of windows sampled/padded per clip.

## Hyperparameter sweep results

| bag_size | embed | lr | val_acc |
|---:|---:|---:|---:|
| 16 | 64 | 0.001 | 0.0990 |
| 16 | 64 | 0.0005 | 0.0844 |
| 16 | 64 | 0.0001 | 0.0638 |
| 16 | 128 | 0.001 | 0.0869 |
| 16 | 128 | 0.0005 | 0.1024 |
| 16 | 128 | 0.0001 | 0.0441 |
| 16 | 256 | 0.001 | 0.0983 |
| 16 | 256 | 0.0005 | 0.0954 |
| 16 | 256 | 0.0001 | 0.0828 |
| 32 | 64 | 0.001 | 0.1105 |
| 32 | 64 | 0.0005 | 0.0913 |
| 32 | 64 | 0.0001 | 0.0622 |
| 32 | 128 | 0.001 | 0.0955 |
| 32 | 128 | 0.0005 | 0.0985 |
| 32 | 128 | 0.0001 | 0.0699 |
| 32 | 256 | 0.001 | 0.0988 |
| 32 | 256 | 0.0005 | 0.0958 |
| 32 | 256 | 0.0001 | 0.0681 |
| 64 | 64 | 0.001 | 0.0926 |
| 64 | 64 | 0.0005 | 0.1000 |
| 64 | 64 | 0.0001 | 0.0488 |
| 64 | 128 | 0.001 | 0.0971 |
| 64 | 128 | 0.0005 | 0.0636 |
| 64 | 128 | 0.0001 | 0.0586 |
| 64 | 256 | 0.001 | 0.0957 |
| 64 | 256 | 0.0005 | 0.0902 |
| 64 | 256 | 0.0001 | 0.0882 |