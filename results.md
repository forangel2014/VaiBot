|sni         | seen task (90%) |            |unseen task (10%) |         |
|-----------|-----------|-------------|-----------| ---------|
|**method** (tested on the 5 unseen samples of each subtask)    | deduction    | induction    | deduction    | induction |
| prompting * | 8.98      | 8.02      |  8.88     |    0.00  |
| vanilla SFT | 32.94   | 40.11       |  31.90    |    16.67  |
| TAGI      |           | -           |           |    -     |
| ItD       | -         |             |    -      |          |
| NesyFlow-in-domain | 33.26 | 85.56  |    21.11   |   44.44   |
| NesyFlow-in-domain - w/o xy |    |    |           |          |
| NesyFlow-in-domain - w/o k | 32.94  | 84.49  |   4.44       | 22.22        |
| NesyFlow-in-domain - finetune w/o encoder |    |    |           |          |
| NesyFlow-pretrain * | 30.37  | 36.36   |   32.22       | 50.00        |
| NesyFlow-pretrain - w/o xy * | 28.77  | 0.53  |   27.78       | 0.00        |
| NesyFlow-pretrain - w/o k * | 18.29  | 36.90  |   10.00       | 44.44        |
| NesyFlow-pretrain - finetune w/o encoder * | 28.98  | 2.14  |   26.67       | 0.00        |

|list functions         | seen task (90%) |            |unseen task (10%) |         |
|-----------|-----------|-------------|-----------| ---------|
|**method** (tested on the 5 unseen samples of each subtask)    | deduction    | induction    | deduction    | induction |
| prompting * |        |        |        |       |
| vanilla SFT |     |        |       |       |
| TAGI      |           | -           |           |    -     |
| ItD       | -         |             |    -      |          |
| NesyFlow-in-domain | 44.09 | 62.67  |    44.00   |   16.00   |
| NesyFlow-pretrain * | 5.33  | 0.44   |   13.60       | 4.00        |


### ps:
- Methods marked with * were not trained on seen tasks
- seen task: sample-level generalization
- unseen task: task-level generalization
- deduction: given $k, x$, infer $y$
- induction: given multiple $x, y$ pairs, infer $k$