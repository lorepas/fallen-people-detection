# TEST SET RESULTS

| Test | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|:---:|---|---|---|---|
| Baseline | 0.49 | 0.95 | 374 | 17 | 11 | 0.97 | 0.96 | 0.96 |
| Only virtual | 0.40 | 0.72 | 49 | 342 | 0 | 1 | 0.13 | 0.22 |
| Fine-tuning virtual than real | 0.41 | 0.84 | 354 | 37 | 61 | 0.85 | 0.91 | 0.88 |
| Fine-tuning virtual and real | 0.51 | 0.94 | 357 | 34 | 11 | 0.97 | 0.91 | 0.94 |
| Paper approach fall classification (only SVM) | ///////////// | //////// | 390 | 1 | 0 | 1 | 0.99 | /////// |
| Paper approach fall detection algorithm (YOLO) | ///////////// | //////// | 304 | 87 | 9 | 0.97 | 0.77 | /////// |
| Paper approach fall detection with pose correction (YOLO)  | ///////////// | //////// | 360 | 31 | 17 | 0.95 | 0.92 | /////// |


# TEST UP-FALL DETECTION RESULTS

## Fall detection (179 fall)

| Test | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Only real (0.9) (0.8) | 0.59 | 0.93 | 179 | 0 | 132 | 0.58 | 1.00 | 0.73 |
| Only virtual (0.2) (0.9) | 0.47 | 0.80 | 123 | 56 | 41 | 0.75 | 0.69 | 0.72 |
| Fine-tuning virtual than real (0.9) (0.9) | 0.65 | 0.99 | 177 | 2 | 1 | 0.99 | 0.99 | 0.99 |
| Fine-tuning virtual and real (0.8) (0.8) | 0.62 | 0.98 | 174 | 5 | 4 | 0.98 | 0.97 | 0.97 |

## No-fall detection (242 no-fall)

| Test | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Only real (0.9) (0.8) | 0.59 | 0.93 | 242 | 0 | 19 | 0.93 | 1.0 | 0.96 |
| Only virtual (0.2) (0.9) | 0.47 | 0.80 | 241 | 1 | 179 | 0.57 | 1.0 | 0.73 |
| Fine-tuning virtual than real (0.9) (0.9) | 0.65 | 0.99 | 241 | 1 | 2 | 0.99 | 1.0 | 0.99 |
| Fine-tuning virtual and real (0.8) (0.8) | 0.62 | 0.98 | 242 | 0 | 19 | 0.93 | 1.0 | 0.96 |


# TEST ELDERLY RESULTS
