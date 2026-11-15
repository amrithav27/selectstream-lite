# P2 Ablation: Standard Metrics & Breakdowns

## Overall Metrics

| Run               | N    | Exact | Contains | Token F1 | R@1 (0.3) | R@1 (0.5) | R@5 (0.3) | R@5 (0.5) | Top-1 IoU |
| ----------------- | ---- | ----- | -------- | -------- | --------- | --------- | --------- | --------- | --------- |
| W_A_seed43_qa     | 867  | 0.123 | 0.198    | 0.230    | 0.029     | 0.014     | 0.093     | 0.044     | 0.022     |
| W_B_seed43_qa     | 867  | 0.122 | 0.197    | 0.229    | 0.022     | 0.009     | 0.065     | 0.030     | 0.018     |
| eval_qa_A         | 3751 | 0.117 | 0.220    | 0.229    | 0.031     | 0.019     | 0.091     | 0.047     | 0.027     |
| eval_qa_B         | 3751 | 0.117 | 0.220    | 0.229    | 0.024     | 0.008     | 0.070     | 0.027     | 0.021     |
| eval_qa_C         | 3751 | 0.117 | 0.219    | 0.228    | 0.040     | 0.018     | 0.102     | 0.047     | 0.031     |
| eval_qa_untrained | 3751 | 0.000 | 0.043    | 0.028    | 0.031     | 0.018     | 0.094     | 0.047     | 0.026     |
| p3_oracle_frames  | 867  | 0.103 | 0.234    | 0.254    | 0.388     | 0.170     | 0.388     | 0.170     | 0.269     |


## Paired Bootstrap vs Reference `eval_qa_A` (resampling video streams, 95% CI)

| Run               | Metric   | Diff vs eval_qa_A | 95% CI             | p-value |
| ----------------- | -------- | ----------------- | ------------------ | ------- |
| W_A_seed43_qa     | contains | -0.0023           | [-0.0064, +0.0000] | 0.2920  |
| W_A_seed43_qa     | token_f1 | -0.0008           | [-0.0028, +0.0000] | 0.7620  |
| W_A_seed43_qa     | r1_iou03 | +0.0023           | [+0.0000, +0.0063] | 0.2300  |
| W_A_seed43_qa     | r5_iou03 | -0.0023           | [-0.0109, +0.0050] | 0.6300  |
| W_A_seed43_qa     | top1_iou | +0.0012           | [-0.0000, +0.0031] | 0.0580  |
| W_B_seed43_qa     | contains | -0.0035           | [-0.0082, +0.0000] | 0.1060  |
| W_B_seed43_qa     | token_f1 | -0.0019           | [-0.0052, +0.0000] | 0.2600  |
| W_B_seed43_qa     | r1_iou03 | -0.0046           | [-0.0175, +0.0074] | 0.5080  |
| W_B_seed43_qa     | r5_iou03 | -0.0311           | [-0.0644, -0.0075] | 0.0120  |
| W_B_seed43_qa     | top1_iou | -0.0025           | [-0.0091, +0.0034] | 0.3920  |
| eval_qa_B         | contains | +0.0000           | [-0.0016, +0.0014] | 0.9800  |
| eval_qa_B         | token_f1 | -0.0002           | [-0.0013, +0.0009] | 0.8560  |
| eval_qa_B         | r1_iou03 | -0.0069           | [-0.0132, -0.0014] | 0.0100  |
| eval_qa_B         | r5_iou03 | -0.0213           | [-0.0319, -0.0124] | 0.0000  |
| eval_qa_B         | top1_iou | -0.0061           | [-0.0096, -0.0029] | 0.0000  |
| eval_qa_C         | contains | -0.0013           | [-0.0032, +0.0003] | 0.1280  |
| eval_qa_C         | token_f1 | -0.0007           | [-0.0019, +0.0003] | 0.1880  |
| eval_qa_C         | r1_iou03 | +0.0091           | [+0.0016, +0.0164] | 0.0220  |
| eval_qa_C         | r5_iou03 | +0.0112           | [-0.0037, +0.0250] | 0.1400  |
| eval_qa_C         | top1_iou | +0.0045           | [+0.0003, +0.0089] | 0.0420  |
| eval_qa_untrained | contains | -0.1776           | [-0.1954, -0.1598] | 0.0000  |
| eval_qa_untrained | token_f1 | -0.2013           | [-0.2166, -0.1863] | 0.0000  |
| eval_qa_untrained | r1_iou03 | -0.0003           | [-0.0030, +0.0028] | 0.9220  |
| eval_qa_untrained | r5_iou03 | +0.0024           | [-0.0022, +0.0064] | 0.3160  |
| eval_qa_untrained | top1_iou | -0.0012           | [-0.0028, +0.0006] | 0.2020  |
| p3_oracle_frames  | contains | +0.0334           | [+0.0023, +0.0726] | 0.0360  |
| p3_oracle_frames  | token_f1 | +0.0224           | [+0.0038, +0.0438] | 0.0220  |
| p3_oracle_frames  | r1_iou03 | +0.3610           | [+0.3043, +0.4118] | 0.0000  |
| p3_oracle_frames  | r5_iou03 | +0.2918           | [+0.2212, +0.3521] | 0.0000  |
| p3_oracle_frames  | top1_iou | +0.2486           | [+0.2168, +0.2757] | 0.0000  |


## Fine-Grained Breakdowns

### By `delay_bucket`

| Run               | delay_bucket | N    | Contains | Token F1 | R@1 (0.3) | Top-1 IoU |
| ----------------- | ------------ | ---- | -------- | -------- | --------- | --------- |
| W_A_seed43_qa     | <0s          | 8    | 0.250    | 0.237    | 0.000     | 0.000     |
| W_A_seed43_qa     | [0,16)s      | 87   | 0.161    | 0.188    | 0.092     | 0.060     |
| W_A_seed43_qa     | [1024,inf)s  | 96   | 0.167    | 0.189    | 0.000     | 0.003     |
| W_A_seed43_qa     | [16,64)s     | 144  | 0.181    | 0.217    | 0.021     | 0.021     |
| W_A_seed43_qa     | [256,1024)s  | 220  | 0.232    | 0.256    | 0.005     | 0.005     |
| W_A_seed43_qa     | [64,256)s    | 312  | 0.202    | 0.243    | 0.042     | 0.029     |
| W_B_seed43_qa     | <0s          | 8    | 0.250    | 0.237    | 0.125     | 0.067     |
| W_B_seed43_qa     | [0,16)s      | 87   | 0.161    | 0.188    | 0.069     | 0.043     |
| W_B_seed43_qa     | [1024,inf)s  | 96   | 0.167    | 0.189    | 0.000     | 0.003     |
| W_B_seed43_qa     | [16,64)s     | 144  | 0.181    | 0.217    | 0.021     | 0.017     |
| W_B_seed43_qa     | [256,1024)s  | 220  | 0.232    | 0.256    | 0.014     | 0.013     |
| W_B_seed43_qa     | [64,256)s    | 312  | 0.199    | 0.240    | 0.019     | 0.018     |
| eval_qa_A         | <0s          | 50   | 0.220    | 0.213    | 0.100     | 0.069     |
| eval_qa_A         | [0,16)s      | 370  | 0.249    | 0.245    | 0.070     | 0.054     |
| eval_qa_A         | [1024,inf)s  | 310  | 0.248    | 0.240    | 0.010     | 0.007     |
| eval_qa_A         | [16,64)s     | 635  | 0.208    | 0.229    | 0.024     | 0.025     |
| eval_qa_A         | [256,1024)s  | 1012 | 0.222    | 0.232    | 0.017     | 0.016     |
| eval_qa_A         | [64,256)s    | 1374 | 0.210    | 0.221    | 0.037     | 0.031     |
| eval_qa_B         | <0s          | 50   | 0.220    | 0.213    | 0.060     | 0.041     |
| eval_qa_B         | [0,16)s      | 370  | 0.251    | 0.248    | 0.049     | 0.038     |
| eval_qa_B         | [1024,inf)s  | 310  | 0.248    | 0.240    | 0.006     | 0.006     |
| eval_qa_B         | [16,64)s     | 635  | 0.211    | 0.230    | 0.024     | 0.019     |
| eval_qa_B         | [256,1024)s  | 1012 | 0.222    | 0.232    | 0.017     | 0.017     |
| eval_qa_B         | [64,256)s    | 1374 | 0.208    | 0.220    | 0.026     | 0.022     |
| eval_qa_C         | <0s          | 50   | 0.220    | 0.213    | 0.060     | 0.044     |
| eval_qa_C         | [0,16)s      | 370  | 0.249    | 0.246    | 0.068     | 0.054     |
| eval_qa_C         | [1024,inf)s  | 310  | 0.248    | 0.240    | 0.013     | 0.009     |
| eval_qa_C         | [16,64)s     | 635  | 0.203    | 0.225    | 0.046     | 0.034     |
| eval_qa_C         | [256,1024)s  | 1012 | 0.221    | 0.231    | 0.023     | 0.018     |
| eval_qa_C         | [64,256)s    | 1374 | 0.210    | 0.221    | 0.049     | 0.038     |
| eval_qa_untrained | <0s          | 50   | 0.060    | 0.035    | 0.080     | 0.064     |
| eval_qa_untrained | [0,16)s      | 370  | 0.051    | 0.039    | 0.073     | 0.054     |
| eval_qa_untrained | [1024,inf)s  | 310  | 0.045    | 0.021    | 0.006     | 0.004     |
| eval_qa_untrained | [16,64)s     | 635  | 0.052    | 0.029    | 0.027     | 0.025     |
| eval_qa_untrained | [256,1024)s  | 1012 | 0.038    | 0.024    | 0.017     | 0.015     |
| eval_qa_untrained | [64,256)s    | 1374 | 0.039    | 0.029    | 0.036     | 0.029     |
| p3_oracle_frames  | <0s          | 8    | 0.125    | 0.254    | 0.125     | 0.140     |
| p3_oracle_frames  | [0,16)s      | 87   | 0.218    | 0.226    | 0.368     | 0.266     |
| p3_oracle_frames  | [1024,inf)s  | 96   | 0.188    | 0.220    | 0.406     | 0.296     |
| p3_oracle_frames  | [16,64)s     | 144  | 0.222    | 0.231    | 0.389     | 0.257     |
| p3_oracle_frames  | [256,1024)s  | 220  | 0.286    | 0.301    | 0.405     | 0.271     |
| p3_oracle_frames  | [64,256)s    | 312  | 0.224    | 0.249    | 0.381     | 0.269     |


### By `participant_disjoint`

| Run               | participant_disjoint | N    | Contains | Token F1 | R@1 (0.3) | Top-1 IoU |
| ----------------- | -------------------- | ---- | -------- | -------- | --------- | --------- |
| W_A_seed43_qa     | disjoint (unseen)    | 47   | 0.277    | 0.211    | 0.000     | 0.000     |
| W_A_seed43_qa     | seen                 | 820  | 0.194    | 0.232    | 0.030     | 0.023     |
| W_B_seed43_qa     | disjoint (unseen)    | 47   | 0.277    | 0.211    | 0.000     | 0.001     |
| W_B_seed43_qa     | seen                 | 820  | 0.193    | 0.230    | 0.023     | 0.019     |
| eval_qa_A         | disjoint (unseen)    | 185  | 0.286    | 0.239    | 0.027     | 0.017     |
| eval_qa_A         | seen                 | 3566 | 0.217    | 0.229    | 0.031     | 0.027     |
| eval_qa_B         | disjoint (unseen)    | 185  | 0.281    | 0.239    | 0.022     | 0.015     |
| eval_qa_B         | seen                 | 3566 | 0.217    | 0.229    | 0.024     | 0.021     |
| eval_qa_C         | disjoint (unseen)    | 185  | 0.276    | 0.228    | 0.043     | 0.033     |
| eval_qa_C         | seen                 | 3566 | 0.216    | 0.229    | 0.040     | 0.031     |
| eval_qa_untrained | disjoint (unseen)    | 185  | 0.049    | 0.039    | 0.022     | 0.017     |
| eval_qa_untrained | seen                 | 3566 | 0.042    | 0.027    | 0.031     | 0.026     |
| p3_oracle_frames  | disjoint (unseen)    | 47   | 0.340    | 0.318    | 0.511     | 0.352     |
| p3_oracle_frames  | seen                 | 820  | 0.228    | 0.250    | 0.380     | 0.264     |


### By `linked`

| Run               | linked   | N    | Contains | Token F1 | R@1 (0.3) | Top-1 IoU |
| ----------------- | -------- | ---- | -------- | -------- | --------- | --------- |
| W_A_seed43_qa     | linked   | 828  | 0.197    | 0.230    | 0.030     | 0.023     |
| W_A_seed43_qa     | unlinked | 39   | 0.231    | 0.237    | 0.000     | 0.000     |
| W_B_seed43_qa     | linked   | 828  | 0.196    | 0.229    | 0.023     | 0.019     |
| W_B_seed43_qa     | unlinked | 39   | 0.231    | 0.237    | 0.000     | 0.000     |
| eval_qa_A         | linked   | 3396 | 0.220    | 0.230    | 0.034     | 0.030     |
| eval_qa_A         | unlinked | 355  | 0.225    | 0.224    | 0.000     | 0.000     |
| eval_qa_B         | linked   | 3396 | 0.219    | 0.230    | 0.027     | 0.023     |
| eval_qa_B         | unlinked | 355  | 0.228    | 0.224    | 0.000     | 0.000     |
| eval_qa_C         | linked   | 3396 | 0.218    | 0.229    | 0.044     | 0.034     |
| eval_qa_C         | unlinked | 355  | 0.228    | 0.224    | 0.000     | 0.000     |
| eval_qa_untrained | linked   | 3396 | 0.042    | 0.028    | 0.034     | 0.028     |
| eval_qa_untrained | unlinked | 355  | 0.045    | 0.026    | 0.000     | 0.000     |
| p3_oracle_frames  | linked   | 828  | 0.237    | 0.259    | 0.406     | 0.282     |
| p3_oracle_frames  | unlinked | 39   | 0.179    | 0.143    | 0.000     | 0.002     |


### By `evidence_location`

| Run               | evidence_location | N    | Contains | Token F1 | R@1 (0.3) | Top-1 IoU |
| ----------------- | ----------------- | ---- | -------- | -------- | --------- | --------- |
| W_A_seed43_qa     | committed         | 796  | 0.206    | 0.239    | 0.028     | 0.021     |
| W_A_seed43_qa     | open              | 51   | 0.098    | 0.130    | 0.020     | 0.026     |
| W_A_seed43_qa     | split             | 20   | 0.150    | 0.158    | 0.100     | 0.043     |
| W_B_seed43_qa     | committed         | 774  | 0.199    | 0.232    | 0.022     | 0.018     |
| W_B_seed43_qa     | open              | 69   | 0.159    | 0.200    | 0.000     | 0.006     |
| W_B_seed43_qa     | split             | 24   | 0.250    | 0.225    | 0.083     | 0.036     |
| eval_qa_A         | committed         | 3433 | 0.219    | 0.228    | 0.030     | 0.026     |
| eval_qa_A         | open              | 235  | 0.226    | 0.230    | 0.034     | 0.032     |
| eval_qa_A         | split             | 83   | 0.253    | 0.286    | 0.072     | 0.057     |
| eval_qa_B         | committed         | 3388 | 0.220    | 0.227    | 0.024     | 0.020     |
| eval_qa_B         | open              | 271  | 0.214    | 0.249    | 0.011     | 0.014     |
| eval_qa_B         | split             | 92   | 0.261    | 0.230    | 0.054     | 0.049     |
| eval_qa_C         | committed         | 3488 | 0.217    | 0.227    | 0.039     | 0.030     |
| eval_qa_C         | open              | 110  | 0.255    | 0.248    | 0.027     | 0.022     |
| eval_qa_C         | split             | 153  | 0.242    | 0.259    | 0.072     | 0.061     |
| eval_qa_untrained | committed         | 3433 | 0.040    | 0.026    | 0.029     | 0.024     |
| eval_qa_untrained | open              | 235  | 0.064    | 0.040    | 0.034     | 0.030     |
| eval_qa_untrained | split             | 83   | 0.096    | 0.061    | 0.084     | 0.069     |
| p3_oracle_frames  | baseline_frames   | 867  | 0.234    | 0.254    | 0.388     | 0.269     |


### By `template`

| Run               | template                                                        | N   | Contains | Token F1 | R@1 (0.3) | Top-1 IoU |
| ----------------- | --------------------------------------------------------------- | --- | -------- | -------- | --------- | --------- |
| W_A_seed43_qa     | None                                                            | 40  | 0.250    | 0.251    | 0.000     | 0.000     |
| W_A_seed43_qa     | Objects: How many X’s? (quantity question)                      | 73  | 0.315    | 0.299    | 0.014     | 0.017     |
| W_A_seed43_qa     | Objects: In what location did I see object X ?                  | 64  | 0.062    | 0.174    | 0.031     | 0.021     |
| W_A_seed43_qa     | Objects: State of an object                                     | 28  | 0.571    | 0.545    | 0.036     | 0.018     |
| W_A_seed43_qa     | Objects: What X did I Y?                                        | 70  | 0.214    | 0.168    | 0.029     | 0.022     |
| W_A_seed43_qa     | Objects: What X is Y?                                           | 63  | 0.270    | 0.233    | 0.032     | 0.017     |
| W_A_seed43_qa     | Objects: What did I put in X?                                   | 85  | 0.141    | 0.139    | 0.035     | 0.024     |
| W_A_seed43_qa     | Objects: Where is my object X?                                  | 20  | 0.200    | 0.298    | 0.000     | 0.002     |
| W_A_seed43_qa     | Objects: Where is object X before / after event Y?              | 150 | 0.147    | 0.237    | 0.027     | 0.028     |
| W_A_seed43_qa     | Objects: Where is object X?                                     | 105 | 0.181    | 0.245    | 0.029     | 0.024     |
| W_A_seed43_qa     | People: Who did I interact with when I did activity X?          | 14  | 0.429    | 0.257    | 0.000     | 0.001     |
| W_A_seed43_qa     | People: Who did I talk to in location X?                        | 9   | 0.667    | 0.243    | 0.111     | 0.086     |
| W_A_seed43_qa     | Place: Where did I put X?                                       | 146 | 0.123    | 0.207    | 0.041     | 0.025     |
| W_B_seed43_qa     | None                                                            | 40  | 0.250    | 0.251    | 0.000     | 0.000     |
| W_B_seed43_qa     | Objects: How many X’s? (quantity question)                      | 73  | 0.315    | 0.299    | 0.027     | 0.024     |
| W_B_seed43_qa     | Objects: In what location did I see object X ?                  | 64  | 0.062    | 0.174    | 0.016     | 0.012     |
| W_B_seed43_qa     | Objects: State of an object                                     | 28  | 0.571    | 0.545    | 0.036     | 0.028     |
| W_B_seed43_qa     | Objects: What X did I Y?                                        | 70  | 0.214    | 0.168    | 0.000     | 0.009     |
| W_B_seed43_qa     | Objects: What X is Y?                                           | 63  | 0.270    | 0.233    | 0.016     | 0.014     |
| W_B_seed43_qa     | Objects: What did I put in X?                                   | 85  | 0.129    | 0.127    | 0.024     | 0.017     |
| W_B_seed43_qa     | Objects: Where is my object X?                                  | 20  | 0.200    | 0.298    | 0.000     | 0.007     |
| W_B_seed43_qa     | Objects: Where is object X before / after event Y?              | 150 | 0.147    | 0.237    | 0.040     | 0.027     |
| W_B_seed43_qa     | Objects: Where is object X?                                     | 105 | 0.181    | 0.245    | 0.010     | 0.013     |
| W_B_seed43_qa     | People: Who did I interact with when I did activity X?          | 14  | 0.429    | 0.257    | 0.143     | 0.069     |
| W_B_seed43_qa     | People: Who did I talk to in location X?                        | 9   | 0.667    | 0.243    | 0.111     | 0.098     |
| W_B_seed43_qa     | Place: Where did I put X?                                       | 146 | 0.123    | 0.207    | 0.014     | 0.013     |
| eval_qa_A         | None                                                            | 368 | 0.234    | 0.231    | 0.000     | 0.001     |
| eval_qa_A         | Objects: How many X’s? (quantity question)                      | 283 | 0.261    | 0.253    | 0.064     | 0.054     |
| eval_qa_A         | Objects: In what location did I see object X ?                  | 264 | 0.144    | 0.217    | 0.038     | 0.025     |
| eval_qa_A         | Objects: State of an object                                     | 154 | 0.610    | 0.602    | 0.026     | 0.034     |
| eval_qa_A         | Objects: What X did I Y?                                        | 280 | 0.254    | 0.208    | 0.054     | 0.038     |
| eval_qa_A         | Objects: What X is Y?                                           | 248 | 0.335    | 0.301    | 0.036     | 0.025     |
| eval_qa_A         | Objects: What did I put in X?                                   | 428 | 0.133    | 0.122    | 0.035     | 0.032     |
| eval_qa_A         | Objects: Where is my object X?                                  | 61  | 0.197    | 0.245    | 0.000     | 0.009     |
| eval_qa_A         | Objects: Where is object X before / after event Y?              | 587 | 0.145    | 0.202    | 0.029     | 0.027     |
| eval_qa_A         | Objects: Where is object X?                                     | 408 | 0.184    | 0.250    | 0.025     | 0.025     |
| eval_qa_A         | People: When did I talk to or interact with person with role X? | 2   | 0.500    | 0.417    | 0.000     | 0.000     |
| eval_qa_A         | People: Who did I interact with when I did activity X?          | 63  | 0.651    | 0.251    | 0.016     | 0.018     |
| eval_qa_A         | People: Who did I talk to in location X?                        | 52  | 0.654    | 0.231    | 0.038     | 0.034     |
| eval_qa_A         | Place: Where did I put X?                                       | 553 | 0.136    | 0.187    | 0.029     | 0.023     |
| eval_qa_B         | None                                                            | 368 | 0.236    | 0.232    | 0.000     | 0.001     |
| eval_qa_B         | Objects: How many X’s? (quantity question)                      | 283 | 0.258    | 0.249    | 0.042     | 0.031     |
| eval_qa_B         | Objects: In what location did I see object X ?                  | 264 | 0.144    | 0.217    | 0.027     | 0.020     |
| eval_qa_B         | Objects: State of an object                                     | 154 | 0.610    | 0.602    | 0.013     | 0.016     |
| eval_qa_B         | Objects: What X did I Y?                                        | 280 | 0.257    | 0.210    | 0.029     | 0.025     |
| eval_qa_B         | Objects: What X is Y?                                           | 248 | 0.339    | 0.305    | 0.024     | 0.027     |
| eval_qa_B         | Objects: What did I put in X?                                   | 428 | 0.129    | 0.119    | 0.033     | 0.030     |
| eval_qa_B         | Objects: Where is my object X?                                  | 61  | 0.197    | 0.245    | 0.000     | 0.006     |
| eval_qa_B         | Objects: Where is object X before / after event Y?              | 587 | 0.145    | 0.202    | 0.032     | 0.022     |
| eval_qa_B         | Objects: Where is object X?                                     | 408 | 0.184    | 0.250    | 0.020     | 0.018     |
| eval_qa_B         | People: When did I talk to or interact with person with role X? | 2   | 0.500    | 0.417    | 0.000     | 0.000     |
| eval_qa_B         | People: Who did I interact with when I did activity X?          | 63  | 0.667    | 0.253    | 0.032     | 0.021     |
| eval_qa_B         | People: Who did I talk to in location X?                        | 52  | 0.635    | 0.231    | 0.038     | 0.028     |
| eval_qa_B         | Place: Where did I put X?                                       | 553 | 0.136    | 0.187    | 0.020     | 0.019     |
| eval_qa_C         | None                                                            | 368 | 0.236    | 0.232    | 0.000     | 0.000     |
| eval_qa_C         | Objects: How many X’s? (quantity question)                      | 283 | 0.265    | 0.256    | 0.064     | 0.051     |
| eval_qa_C         | Objects: In what location did I see object X ?                  | 264 | 0.140    | 0.213    | 0.030     | 0.025     |
| eval_qa_C         | Objects: State of an object                                     | 154 | 0.604    | 0.596    | 0.052     | 0.039     |
| eval_qa_C         | Objects: What X did I Y?                                        | 280 | 0.250    | 0.206    | 0.061     | 0.045     |
| eval_qa_C         | Objects: What X is Y?                                           | 248 | 0.335    | 0.301    | 0.036     | 0.028     |
| eval_qa_C         | Objects: What did I put in X?                                   | 428 | 0.131    | 0.121    | 0.033     | 0.032     |
| eval_qa_C         | Objects: Where is my object X?                                  | 61  | 0.197    | 0.245    | 0.049     | 0.037     |
| eval_qa_C         | Objects: Where is object X before / after event Y?              | 587 | 0.145    | 0.202    | 0.043     | 0.029     |
| eval_qa_C         | Objects: Where is object X?                                     | 408 | 0.181    | 0.248    | 0.059     | 0.039     |
| eval_qa_C         | People: When did I talk to or interact with person with role X? | 2   | 0.500    | 0.417    | 0.000     | 0.000     |
| eval_qa_C         | People: Who did I interact with when I did activity X?          | 63  | 0.635    | 0.257    | 0.111     | 0.065     |
| eval_qa_C         | People: Who did I talk to in location X?                        | 52  | 0.635    | 0.224    | 0.038     | 0.040     |
| eval_qa_C         | Place: Where did I put X?                                       | 553 | 0.136    | 0.187    | 0.029     | 0.027     |
| eval_qa_untrained | None                                                            | 368 | 0.043    | 0.026    | 0.000     | 0.001     |
| eval_qa_untrained | Objects: How many X’s? (quantity question)                      | 283 | 0.046    | 0.021    | 0.060     | 0.048     |
| eval_qa_untrained | Objects: In what location did I see object X ?                  | 264 | 0.034    | 0.020    | 0.042     | 0.027     |
| eval_qa_untrained | Objects: State of an object                                     | 154 | 0.078    | 0.012    | 0.032     | 0.031     |
| eval_qa_untrained | Objects: What X did I Y?                                        | 280 | 0.036    | 0.026    | 0.054     | 0.037     |
| eval_qa_untrained | Objects: What X is Y?                                           | 248 | 0.081    | 0.029    | 0.032     | 0.026     |
| eval_qa_untrained | Objects: What did I put in X?                                   | 428 | 0.023    | 0.014    | 0.033     | 0.029     |
| eval_qa_untrained | Objects: Where is my object X?                                  | 61  | 0.049    | 0.051    | 0.016     | 0.014     |
| eval_qa_untrained | Objects: Where is object X before / after event Y?              | 587 | 0.046    | 0.029    | 0.031     | 0.027     |
| eval_qa_untrained | Objects: Where is object X?                                     | 408 | 0.044    | 0.048    | 0.022     | 0.022     |
| eval_qa_untrained | People: When did I talk to or interact with person with role X? | 2   | 0.000    | 0.000    | 0.000     | 0.000     |
| eval_qa_untrained | People: Who did I interact with when I did activity X?          | 63  | 0.000    | 0.019    | 0.016     | 0.014     |
| eval_qa_untrained | People: Who did I talk to in location X?                        | 52  | 0.000    | 0.004    | 0.038     | 0.038     |
| eval_qa_untrained | Place: Where did I put X?                                       | 553 | 0.040    | 0.037    | 0.027     | 0.022     |
| p3_oracle_frames  | None                                                            | 40  | 0.175    | 0.140    | 0.000     | 0.004     |
| p3_oracle_frames  | Objects: How many X’s? (quantity question)                      | 73  | 0.315    | 0.253    | 0.329     | 0.240     |
| p3_oracle_frames  | Objects: In what location did I see object X ?                  | 64  | 0.078    | 0.166    | 0.359     | 0.255     |
| p3_oracle_frames  | Objects: State of an object                                     | 28  | 0.500    | 0.389    | 0.571     | 0.302     |
| p3_oracle_frames  | Objects: What X did I Y?                                        | 70  | 0.271    | 0.212    | 0.343     | 0.248     |
| p3_oracle_frames  | Objects: What X is Y?                                           | 63  | 0.492    | 0.391    | 0.444     | 0.301     |
| p3_oracle_frames  | Objects: What did I put in X?                                   | 85  | 0.306    | 0.268    | 0.318     | 0.239     |
| p3_oracle_frames  | Objects: Where is my object X?                                  | 20  | 0.200    | 0.241    | 0.500     | 0.319     |
| p3_oracle_frames  | Objects: Where is object X before / after event Y?              | 150 | 0.207    | 0.277    | 0.460     | 0.311     |
| p3_oracle_frames  | Objects: Where is object X?                                     | 105 | 0.181    | 0.261    | 0.362     | 0.269     |
| p3_oracle_frames  | People: Who did I interact with when I did activity X?          | 14  | 0.071    | 0.238    | 0.214     | 0.223     |
| p3_oracle_frames  | People: Who did I talk to in location X?                        | 9   | 0.111    | 0.265    | 0.222     | 0.213     |
| p3_oracle_frames  | Place: Where did I put X?                                       | 146 | 0.151    | 0.222    | 0.493     | 0.328     |

