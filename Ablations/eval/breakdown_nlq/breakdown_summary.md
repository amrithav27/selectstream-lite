# P2 Ablation: Standard Metrics & Breakdowns

## Overall Metrics

| Run        | N    | Exact | Contains | Token F1 | R@1 (0.3) | R@1 (0.5) | R@5 (0.3) | R@5 (0.5) | Top-1 IoU |
| ---------- | ---- | ----- | -------- | -------- | --------- | --------- | --------- | --------- | --------- |
| eval_nlq_A | 4552 | 0.000 | 0.000    | 0.922    | 0.038     | 0.021     | 0.112     | 0.058     | 0.032     |
| eval_nlq_B | 4552 | 0.000 | 0.000    | 0.922    | 0.030     | 0.010     | 0.089     | 0.035     | 0.025     |
| eval_nlq_C | 4552 | 0.000 | 0.000    | 0.922    | 0.048     | 0.021     | 0.125     | 0.055     | 0.037     |


## Paired Bootstrap vs Reference `eval_nlq_A` (resampling video streams, 95% CI)

| Run        | Metric   | Diff vs eval_nlq_A | 95% CI             | p-value |
| ---------- | -------- | ------------------ | ------------------ | ------- |
| eval_nlq_B | contains | +0.0000            | [+0.0000, +0.0000] | 1.0000  |
| eval_nlq_B | token_f1 | +0.0000            | [+0.0000, +0.0000] | 1.0000  |
| eval_nlq_B | r1_iou03 | -0.0081            | [-0.0148, -0.0014] | 0.0180  |
| eval_nlq_B | r5_iou03 | -0.0228            | [-0.0335, -0.0122] | 0.0000  |
| eval_nlq_B | top1_iou | -0.0070            | [-0.0107, -0.0033] | 0.0000  |
| eval_nlq_C | contains | +0.0000            | [+0.0000, +0.0000] | 1.0000  |
| eval_nlq_C | token_f1 | +0.0000            | [+0.0000, +0.0000] | 1.0000  |
| eval_nlq_C | r1_iou03 | +0.0097            | [+0.0014, +0.0181] | 0.0200  |
| eval_nlq_C | r5_iou03 | +0.0132            | [+0.0004, +0.0267] | 0.0420  |
| eval_nlq_C | top1_iou | +0.0051            | [+0.0003, +0.0099] | 0.0380  |


## Fine-Grained Breakdowns

### By `delay_bucket`

| Run        | delay_bucket | N    | Contains | Token F1 | R@1 (0.3) | Top-1 IoU |
| ---------- | ------------ | ---- | -------- | -------- | --------- | --------- |
| eval_nlq_A | <0s          | 78   | 0.000    | 0.936    | 0.090     | 0.072     |
| eval_nlq_A | [0,16)s      | 476  | 0.000    | 0.914    | 0.067     | 0.051     |
| eval_nlq_A | [1024,inf)s  | 406  | 0.000    | 0.874    | 0.025     | 0.016     |
| eval_nlq_A | [16,64)s     | 756  | 0.000    | 0.934    | 0.036     | 0.035     |
| eval_nlq_A | [256,1024)s  | 1206 | 0.000    | 0.918    | 0.022     | 0.019     |
| eval_nlq_A | [64,256)s    | 1630 | 0.000    | 0.932    | 0.044     | 0.037     |
| eval_nlq_B | <0s          | 78   | 0.000    | 0.936    | 0.077     | 0.044     |
| eval_nlq_B | [0,16)s      | 476  | 0.000    | 0.914    | 0.057     | 0.046     |
| eval_nlq_B | [1024,inf)s  | 406  | 0.000    | 0.874    | 0.005     | 0.005     |
| eval_nlq_B | [16,64)s     | 756  | 0.000    | 0.934    | 0.041     | 0.028     |
| eval_nlq_B | [256,1024)s  | 1206 | 0.000    | 0.918    | 0.021     | 0.020     |
| eval_nlq_B | [64,256)s    | 1630 | 0.000    | 0.932    | 0.029     | 0.025     |
| eval_nlq_C | <0s          | 78   | 0.000    | 0.936    | 0.026     | 0.027     |
| eval_nlq_C | [0,16)s      | 476  | 0.000    | 0.914    | 0.078     | 0.064     |
| eval_nlq_C | [1024,inf)s  | 406  | 0.000    | 0.874    | 0.020     | 0.014     |
| eval_nlq_C | [16,64)s     | 756  | 0.000    | 0.934    | 0.060     | 0.043     |
| eval_nlq_C | [256,1024)s  | 1206 | 0.000    | 0.918    | 0.034     | 0.026     |
| eval_nlq_C | [64,256)s    | 1630 | 0.000    | 0.932    | 0.053     | 0.041     |


### By `participant_disjoint`

| Run        | participant_disjoint | N    | Contains | Token F1 | R@1 (0.3) | Top-1 IoU |
| ---------- | -------------------- | ---- | -------- | -------- | --------- | --------- |
| eval_nlq_A | disjoint (unseen)    | 328  | 0.000    | 0.765    | 0.040     | 0.030     |
| eval_nlq_A | seen                 | 4224 | 0.000    | 0.934    | 0.038     | 0.032     |
| eval_nlq_B | disjoint (unseen)    | 328  | 0.000    | 0.765    | 0.024     | 0.018     |
| eval_nlq_B | seen                 | 4224 | 0.000    | 0.934    | 0.031     | 0.026     |
| eval_nlq_C | disjoint (unseen)    | 328  | 0.000    | 0.765    | 0.055     | 0.038     |
| eval_nlq_C | seen                 | 4224 | 0.000    | 0.934    | 0.048     | 0.037     |


### By `linked`

| Run        | linked   | N    | Contains | Token F1 | R@1 (0.3) | Top-1 IoU |
| ---------- | -------- | ---- | -------- | -------- | --------- | --------- |
| eval_nlq_A | linked   | 3396 | 0.000    | 1.000    | 0.035     | 0.030     |
| eval_nlq_A | unlinked | 1156 | 0.000    | 0.691    | 0.048     | 0.039     |
| eval_nlq_B | linked   | 3396 | 0.000    | 1.000    | 0.026     | 0.022     |
| eval_nlq_B | unlinked | 1156 | 0.000    | 0.691    | 0.042     | 0.033     |
| eval_nlq_C | linked   | 3396 | 0.000    | 1.000    | 0.042     | 0.033     |
| eval_nlq_C | unlinked | 1156 | 0.000    | 0.691    | 0.067     | 0.049     |


### By `evidence_location`

| Run        | evidence_location | N    | Contains | Token F1 | R@1 (0.3) | Top-1 IoU |
| ---------- | ----------------- | ---- | -------- | -------- | --------- | --------- |
| eval_nlq_A | committed         | 4169 | 0.000    | 0.923    | 0.037     | 0.031     |
| eval_nlq_A | open              | 257  | 0.000    | 0.914    | 0.035     | 0.035     |
| eval_nlq_A | split             | 126  | 0.000    | 0.873    | 0.087     | 0.057     |
| eval_nlq_B | committed         | 4082 | 0.000    | 0.922    | 0.029     | 0.024     |
| eval_nlq_B | open              | 324  | 0.000    | 0.951    | 0.022     | 0.021     |
| eval_nlq_B | split             | 146  | 0.000    | 0.856    | 0.075     | 0.063     |
| eval_nlq_C | committed         | 4234 | 0.000    | 0.921    | 0.048     | 0.036     |
| eval_nlq_C | open              | 116  | 0.000    | 0.931    | 0.017     | 0.025     |
| eval_nlq_C | split             | 202  | 0.000    | 0.931    | 0.074     | 0.063     |


### By `template`

| Run        | template                                                        | N   | Contains | Token F1 | R@1 (0.3) | Top-1 IoU |
| ---------- | --------------------------------------------------------------- | --- | -------- | -------- | --------- | --------- |
| eval_nlq_A | None                                                            | 17  | 0.000    | 0.941    | 0.000     | 0.016     |
| eval_nlq_A | Objects: How many X’s? (quantity question)                      | 386 | 0.000    | 0.881    | 0.067     | 0.055     |
| eval_nlq_A | Objects: In what location did I see object X ?                  | 359 | 0.000    | 0.969    | 0.033     | 0.026     |
| eval_nlq_A | Objects: State of an object                                     | 235 | 0.000    | 0.813    | 0.047     | 0.040     |
| eval_nlq_A | Objects: What X did I Y?                                        | 350 | 0.000    | 0.963    | 0.054     | 0.040     |
| eval_nlq_A | Objects: What X is Y?                                           | 332 | 0.000    | 0.916    | 0.042     | 0.031     |
| eval_nlq_A | Objects: What did I put in X?                                   | 546 | 0.000    | 0.899    | 0.042     | 0.032     |
| eval_nlq_A | Objects: Where is my object X?                                  | 72  | 0.000    | 0.972    | 0.028     | 0.029     |
| eval_nlq_A | Objects: Where is object X before / after event Y?              | 750 | 0.000    | 0.955    | 0.035     | 0.031     |
| eval_nlq_A | Objects: Where is object X?                                     | 552 | 0.000    | 0.891    | 0.027     | 0.026     |
| eval_nlq_A | People: When did I talk to or interact with person with role X? | 22  | 0.000    | 1.000    | 0.000     | 0.003     |
| eval_nlq_A | People: Who did I interact with when I did activity X?          | 115 | 0.000    | 0.965    | 0.017     | 0.018     |
| eval_nlq_A | People: Who did I talk to in location X?                        | 91  | 0.000    | 0.923    | 0.055     | 0.048     |
| eval_nlq_A | Place: Where did I put X?                                       | 725 | 0.000    | 0.928    | 0.028     | 0.024     |
| eval_nlq_B | None                                                            | 17  | 0.000    | 0.941    | 0.000     | 0.030     |
| eval_nlq_B | Objects: How many X’s? (quantity question)                      | 386 | 0.000    | 0.881    | 0.039     | 0.032     |
| eval_nlq_B | Objects: In what location did I see object X ?                  | 359 | 0.000    | 0.969    | 0.039     | 0.027     |
| eval_nlq_B | Objects: State of an object                                     | 235 | 0.000    | 0.813    | 0.013     | 0.017     |
| eval_nlq_B | Objects: What X did I Y?                                        | 350 | 0.000    | 0.963    | 0.031     | 0.027     |
| eval_nlq_B | Objects: What X is Y?                                           | 332 | 0.000    | 0.916    | 0.039     | 0.031     |
| eval_nlq_B | Objects: What did I put in X?                                   | 546 | 0.000    | 0.899    | 0.027     | 0.025     |
| eval_nlq_B | Objects: Where is my object X?                                  | 72  | 0.000    | 0.972    | 0.028     | 0.030     |
| eval_nlq_B | Objects: Where is object X before / after event Y?              | 750 | 0.000    | 0.955    | 0.039     | 0.025     |
| eval_nlq_B | Objects: Where is object X?                                     | 552 | 0.000    | 0.891    | 0.024     | 0.022     |
| eval_nlq_B | People: When did I talk to or interact with person with role X? | 22  | 0.000    | 1.000    | 0.000     | 0.014     |
| eval_nlq_B | People: Who did I interact with when I did activity X?          | 115 | 0.000    | 0.965    | 0.043     | 0.033     |
| eval_nlq_B | People: Who did I talk to in location X?                        | 91  | 0.000    | 0.923    | 0.055     | 0.045     |
| eval_nlq_B | Place: Where did I put X?                                       | 725 | 0.000    | 0.928    | 0.018     | 0.017     |
| eval_nlq_C | None                                                            | 17  | 0.000    | 0.941    | 0.000     | 0.005     |
| eval_nlq_C | Objects: How many X’s? (quantity question)                      | 386 | 0.000    | 0.881    | 0.067     | 0.055     |
| eval_nlq_C | Objects: In what location did I see object X ?                  | 359 | 0.000    | 0.969    | 0.042     | 0.033     |
| eval_nlq_C | Objects: State of an object                                     | 235 | 0.000    | 0.813    | 0.038     | 0.038     |
| eval_nlq_C | Objects: What X did I Y?                                        | 350 | 0.000    | 0.963    | 0.063     | 0.047     |
| eval_nlq_C | Objects: What X is Y?                                           | 332 | 0.000    | 0.916    | 0.042     | 0.032     |
| eval_nlq_C | Objects: What did I put in X?                                   | 546 | 0.000    | 0.899    | 0.035     | 0.030     |
| eval_nlq_C | Objects: Where is my object X?                                  | 72  | 0.000    | 0.972    | 0.069     | 0.055     |
| eval_nlq_C | Objects: Where is object X before / after event Y?              | 750 | 0.000    | 0.955    | 0.047     | 0.036     |
| eval_nlq_C | Objects: Where is object X?                                     | 552 | 0.000    | 0.891    | 0.067     | 0.042     |
| eval_nlq_C | People: When did I talk to or interact with person with role X? | 22  | 0.000    | 1.000    | 0.091     | 0.054     |
| eval_nlq_C | People: Who did I interact with when I did activity X?          | 115 | 0.000    | 0.965    | 0.078     | 0.048     |
| eval_nlq_C | People: Who did I talk to in location X?                        | 91  | 0.000    | 0.923    | 0.055     | 0.051     |
| eval_nlq_C | Place: Where did I put X?                                       | 725 | 0.000    | 0.928    | 0.029     | 0.025     |

