|sni         | seen task (90%) |            |unseen task (10%) |         |
|-----------|-----------|-------------|-----------| ---------|
|**method** (tested on the 50% unseen samples of each subtask)    | deduction    | induction    | deduction    | induction |
| prompting * | 1.28      | 9.09      |  1.17     |    0.00  |
| vanilla SFT | 25.69   | 37.43       |  24.59    |    0.00  |
| TAGI      |           | -           |           |    -     |
| ItD       | -         |             |    -      |          |
| NesyFlow-in-domain | 26.95 | 93.58  |    5.56   |   33.33   |
| NesyFlow-pretrain * | 3.77  | 30.00   |   5.60       | 44.44        |
| NesyFlow-pretrain (llama-2-7b => Yi-Coder-9B) | -  | 19.79   |   -       | 11.11       |

### ps:
- Methods marked with * were not trained on seen tasks
- seen task: sample-level generalization
- unseen task: task-level generalization
- deduction: given $k, x$, infer $y$
- induction: given multiple $x, y$ pairs, infer $k$