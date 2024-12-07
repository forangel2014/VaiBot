|sni         | seen task (90%) |            |unseen task (10%) |         |
|-----------|-----------|-------------|-----------| ---------|
|           | neural    | symbolic    | neural    | symbolic |
| prompting * | 1.28      | 9.09      |  1.17     |    0.00  |
| vanilla SFT | 25.69   | 37.43       |           |    0.00  |
| huanxuan  |           | -           |           |          |
| itd       | -         |             |           |          |
| NesyFlow-in-domain | 26.95 | 93.58  |    5.56   |   33.33   |
| NesyFlow-pretrain * | 3.77  | 30.00   |   $\approx$ 3.77       | $\approx$ 30.00         |

\* Methods marked with * were not trained on seen tasks

\* neural: given $k, x$, infer $y$

\* symbolic: given multiple $x, y$ pairs, infer $k$