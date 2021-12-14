| Test | mAP@[0.5:0.95] | mAP@[0.5] | TP | FN | FP | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|:---:|---|---|---|---|
| Baseline | 0.49 | 0.95 | 374 | 17 | 11 | 0.97 | 0.96 | 0.96 |
| Fine-tuning virtual than real | 0.41 | 0.84 | 354 | 37 | 61 | 0.85 | 0.91 | 0.88 |
| Fine-tuning virtual and real | 0.51 | 0.94 | 357 | 34 | 11 | 0.97 | 0.91 | 0.94 |
| Paper approach fall classification (only SVM) | ///////////// | //////// | 390 | 1 | 0 | 1 | 0.99 | /////// |
| Paper approach fall detection algorithm (YOLO) | ///////////// | //////// | 304 | 87 | 9 | 0.97 | 0.77 | /////// |
| Paper approach fall detection with pose correction (YOLO)  | ///////////// | //////// | 360 | 31 | 17 | 0.95 | 0.92 | /////// |
