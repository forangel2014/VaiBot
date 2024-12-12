|sni         | seen task (90%) |            |unseen task (10%) |         |
|-----------|-----------|-------------|-----------| ---------|
|**method** (tested on the 5 unseen samples of each subtask)    | deduction    | induction    | deduction    | induction |
| prompting * | 8.98      | 8.02      |  8.88     |    0.00  |
| vanilla SFT | 32.94   | 33.16       |  31.90    |    0.00  |
| TAGI      |           | -           |           |    -     |
| ItD       | -         |             |    -      |          |
| NesyFlow-in-domain | 33.26 | 85.56  |    21.11   |   44.44   |
| NesyFlow-pretrain * | 3.77  | 30.00   |   5.60       | 44.44        |
| NesyFlow-pretrain (llama-2-7b => Yi-Coder-9B) | -  | 19.79   |   -       | 11.11       |

### ps:
- Methods marked with * were not trained on seen tasks
- seen task: sample-level generalization
- unseen task: task-level generalization
- deduction: given $k, x$, infer $y$
- induction: given multiple $x, y$ pairs, infer $k$