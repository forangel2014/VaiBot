|sni         | seen task (90%) |            |unseen task (10%) |         |
|-----------|-----------|-------------|-----------| ---------|
|           | deduction    | induction    | deduction    | induction |
| prompting * | 1.28      | 9.09      |  1.17     |    0.00  |
| vanilla SFT | 25.69   | 37.43       |  24.59    |    0.00  |
| huanxuan  |           | -           |           |    -     |
| itd       | -         |             |    -      |          |
| NesyFlow-in-domain | 26.95 | 93.58  |    5.56   |   33.33   |
| NesyFlow-pretrain * | 3.77  | 30.00   |   5.60       | $\approx$ 30.00         |

\* Methods marked with * were not trained on seen tasks

\* deduction: given $k, x$, infer $y$

\* induction: given multiple $x, y$ pairs, infer $k$