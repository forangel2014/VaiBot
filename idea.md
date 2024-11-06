### 出发点
+ 建立参数知识到文本知识的双向映射
    - 具体而言，给定一个Base LM
    - 选取其一部分参数（LoRA，共$N$个实数）作为参数空间（i.e. $\mathbb{\Theta} = \mathbb{R}^{N}$）
    - 设定最大文本长度$T$，作为文本空间（i.e. $\mathbb{\Phi} = \bigcup_{t=1}^T \mathbb{V}^{t}$）
    - 我们希望学习到两个映射
        - $f: \mathbb{\Theta} \rightarrow \mathbb{\Phi}$，即对于模型学习到的任务知识带来的参数增量$\Delta \theta$，$f$能将其映射成自然语言$\varphi$（归纳，or Neural to Symbolic）
        - $g: \mathbb{\Phi} \rightarrow \mathbb{\Theta}$，即对于外部给定的自然语言描述的任务知识$\varphi$，$g$能将其映射成模型参数增量$\Delta \theta$（应用，or Symbolic to Neural）

### 数据
+ 为了训练两个映射，我们需要（任务知识$\varphi$，参数增量$\Delta \theta$）的对。为了获得$\Delta \theta$，我们还需要对应任务的实例化输入-输出对($x$, $y$)
+ 由于预计$f$和$g$在小规模的训练下缺少泛化性，我们可以先将场景局限在比较toy的任务上。例如，潜在的任务知识是“小明是一个中国人”，则可以构造的训练数据有（小明会说什么语言，中文），（小明所在国家的首都是，北京）等等
+ 这些数据可以通过给定任务知识，加上ICL从现有LLM中合成获得

### 网络
+ 两个单独的超网络
    - 分别训练，互为输入输出
+ 标准化流（优势在哪里？可以计算具体的概率值？）
    - $g = f^{-1}$，只需训练一个网络，共享参数
    - 自回归流
        - 24年1月的paper Transformer Neural Autoregressive Flow表明可以用现成的transformer作为流模型的主干网络，只需在最外层套一个可逆的变换
        - 我们可以共享Base LM的参数，用于同时作为流模型的参数
        - 虽然文本分布可以很好的用自回归建模，但参数空间大小应是固定的，而自回归流要求输入与输出尺寸一致？(这意味着输入token数与输出参数个数必须一样？)
        - 似乎不合理，参数增量的联合分布不能用自回归建模，应该建模为$\mathcal{N}(\mu, \Sigma)$更为合理
        - 为了使参数增量的联合分布更符合多元高斯，可以尽可能分散参数的选取？
        - 预训练？先预训练流网络将文本分布映射到$\mathcal{N}(0, 1)$的能力

        - 把任务finetuning $p(y_i | x_i, theta)$和VAE放在一起训，这样会导致损失一部分任务性能，但是相当于用任务性能换了可解释性？（与kappa相互映射的能力）
        - 文本空间拼接，参数空间相加：启发式损失
        - subtask优化的参数空间可能是多峰的，不是简单的高斯分布

        - tsne验证假设：是否encoder生成的z和真实任务微调出的z分布一致？每个subtask多微调几轮，获得多个z

        - 训练：task loss微调没有用验证集，导致参数是过拟合的分布；但测试时用了验证集，导致分布不一致
        - decoder能力不足？尝试加大soft token数至5
        - 参数空间越接近，不一定代表LM分布越接近？

        - 扰动训练，对每个训练样本的Latent + encoder尺度的高斯噪声，促使其收敛于encoder的分布里
        - flow将latent 映射至param space
        - 初始分布：条件高斯
        - 目标分布：delta 所有局部最优解空间参数
        
        -straight through?

        $\min \mathbb{E} -\log N(f_{w}^{-1}(\theta) | \mu(k), \Sigma(k))$
        where
        $\theta = \argmin \mathbb{E} -\log p(y|x;\theta)$

        $\frac{\partial N}{\partial w} = g(\theta)$

        -采样latent
        -forward,变化到param
        -计算params梯度
        -进行一步params梯度更新，得到假想的更优params
        -对假想的更优params backward
        -计算对数似然，优化
        -流模型正确的训练方法：从param space中，根据EBM建模的分布采样（MCMC?），然后backward到latent space中，计算最大似然
        -pretrain abc
        -加强正则，要求delta params稀疏