# PAPER RESULTS

These results take only the fall detection. In test set there are 391 falls.

| Test | TP | FN | FP | Precision | Recall | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| Fall classification (only SVM) | 390 | 1 | 0 | 1 | 0.99 |
| Fall detection algorithm (YOLO + SVM) | 304 | 87 | 9 | 0.97 | 0.77 |
| Fall detection with pose correction (YOLO + SVM)  | 360 | 31 | 17 | 0.95 | 0.92 |

# COMPARE THE FALLEN RESULTS BETWEEN DATASETS
## FPDS
| Test | thr | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Only real | 0.8 | 0.49 | 0.95 | 371 | 20 | 5 | 0.99 | 0.95 | 0.97 |
| Only virtual | 0.4 | 0.40 | 0.72 | 254 | 137 | 94 | 0.73 | 0.65 | 0.69 |
| Fine-tuning virtual than real | 0.6 | 0.41 | 0.84 | 362 | 29 | 73 | 0.83 | 0.93 | 0.88 |
| Fine-tuning virtual and real | 0.6 | 0.51 | 0.94 | 367 | 24 | 18 | 0.95 | 0.94 | 0.95 |
## UP-FALL DETECTION
| Test | thr | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Only real | 0.9 | 0.59 | 0.93 | 179 | 0 | 132 | 0.58 | 1.00 | 0.73 |
| Only virtual | 0.2 | 0.47 | 0.80 | 123 | 56 | 41 | 0.75 | 0.69 | 0.72 |
| Fine-tuning virtual than real | 0.9 | 0.65 | 0.99 | 177 | 2 | 1 | 0.99 | 0.99 | 0.99 |
| Fine-tuning virtual and real | 0.8 | 0.62 | 0.98 | 174 | 5 | 4 | 0.98 | 0.97 | 0.97 |
## ELDERLY
| Test | thr | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Only real | 0.8 | 0.16 | 0.4 | 250 | 22 | 25 | 0.91 | 0.92 | 0.91 |
| Only virtual | 0.3 | 0.07 | 0.18 | 199 | 73 | 194 | 0.51 | 0.73 | 0.60 |
| Fine-tuning virtual than real | 0.8 | 0.11 | 0.3 | 228 | 44 | 61 | 0.79 | 0.84 | 0.81 |
| Fine-tuning virtual and real | 0.8 | 0.14 | 0.33 | 262 | 10 | 121 | 0.68 | 0.96 | 0.80 |

# TEST FPDS RESULTS

## Fall detection (391 fall)

| Test | thr | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Only real | 0.8 | 0.49 | 0.95 | 371 | 20 | 5 | 0.99 | 0.95 | 0.97 |
| Only virtual | 0.4 | 0.40 | 0.72 | 254 | 137 | 94 | 0.73 | 0.65 | 0.69 |
| Fine-tuning virtual than real | 0.6 | 0.41 | 0.84 | 362 | 29 | 73 | 0.83 | 0.93 | 0.88 |
| Fine-tuning virtual and real | 0.6 | 0.51 | 0.94 | 367 | 24 | 18 | 0.95 | 0.94 | 0.95 |

## No-fall detection (830 no-fall)

| Test | thr | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Only real | 0.5 | 0.49 | 0.95 | 789 | 41 | 73 | 0.92 | 0.95 | 0.93 |
| Only virtual | 0.7 | 0.40 | 0.72 | 792 | 38 | 391 | 0.67 | 0.95 | 0.79 |
| Fine-tuning virtual than real | 0.4 | 0.41 | 0.84 | 703 | 127 | 30 | 0.96 | 0.85 | 0.90 |
| Fine-tuning virtual and real | 0.5 | 0.51 | 0.94 | 760 | 70 | 24 | 0.97 | 0.92 | 0.94 |


# TEST UP-FALL DETECTION RESULTS

## Fall detection (179 fall)

| Test | thr | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Only real | 0.9 | 0.59 | 0.93 | 179 | 0 | 132 | 0.58 | 1.00 | 0.73 |
| Only virtual | 0.2 | 0.47 | 0.80 | 123 | 56 | 41 | 0.75 | 0.69 | 0.72 |
| Fine-tuning virtual than real | 0.9 | 0.65 | 0.99 | 177 | 2 | 1 | 0.99 | 0.99 | 0.99 |
| Fine-tuning virtual and real | 0.8 | 0.62 | 0.98 | 174 | 5 | 4 | 0.98 | 0.97 | 0.97 |

## No-fall detection (242 no-fall)

| Test | thr | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Only real | 0.8 | 0.59 | 0.93 | 242 | 0 | 19 | 0.93 | 1.0 | 0.96 |
| Only virtual | 0.9 | 0.47 | 0.80 | 241 | 1 | 179 | 0.57 | 1.0 | 0.73 |
| Fine-tuning virtual than real | 0.9 | 0.65 | 0.99 | 241 | 1 | 2 | 0.99 | 1.0 | 0.99 |
| Fine-tuning virtual and real | 0.8 | 0.62 | 0.98 | 242 | 0 | 19 | 0.93 | 1.0 | 0.96 |


# TEST ELDERLY RESULTS

## Fall detection (272 fall)

| Test | thr | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Only real | 0.8 | 0.16 | 0.4 | 250 | 22 | 25 | 0.91 | 0.92 | 0.91 |
| Only virtual | 0.3 | 0.07 | 0.18 | 199 | 73 | 194 | 0.51 | 0.73 | 0.60 |
| Fine-tuning virtual than real | 0.8 | 0.11 | 0.3 | 228 | 44 | 61 | 0.79 | 0.84 | 0.81 |
| Fine-tuning virtual and real | 0.8 | 0.14 | 0.33 | 262 | 10 | 121 | 0.68 | 0.96 | 0.80 |

## No-fall detection (141 no-fall)*

| Test | thr | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Only real | 0.9 | 0.16 | 0.4 | 139 | 2 | 88 | 0.61 | 0.99 | 0.76 |
| Only virtual | 0.9 | 0.07 | 0.18 | 141 | 0 | 293 | 0.32 | 1.00 | 0.49 |
| Fine-tuning virtual than real | 0.9 | 0.11 | 0.3 | 127 | 14 | 97 | 0.57 | 0.90 | 0.70 |
| Fine-tuning virtual and real | 0.9 | 0.14 | 0.33 | 64 | 77 | 49 | 0.57 | 0.45 | 0.50 |

\*In this case, all the no-falls are annotations without text
