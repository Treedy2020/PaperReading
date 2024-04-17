context-target pairs: 
## Conditional language modeling objective
- $\mathcal{Z}=\{(x_{i}, y{i})\}_{i=1,...,N}$
$$
\max_{\Phi}\sum_{x, y\in\mathcal{Z}}\sum_{t=1}^{\left\vert y \right\vert}\log{P(y_{t} \vert x, y_{<t} )}
$$

## SVD(Sigular Value Decomposition) 
Why does SVD could be used for the subspace similarity between different matrix.

## Frobenius Norm
What is FN. 


## LoRA 中的R是什么，它因该怎么选取？
根据搜索结果，LoRA中的r指的是低秩自适应矩阵的秩(rank)，它控制了LoRA引入的可训练参数的数量。r值的选择需要在模型复杂性、自适应能力和过拟合/欠拟合风险之间进行权衡。

1. LoRA的核心思想是用两个低秩矩阵A和B来近似表示微调过程中的参数更新量ΔW，其中A和B的秩为r，远小于原始矩阵的秩。这样可以大幅减少微调参数量。

2. r值越大，LoRA引入的可训练参数就越多，模型的表达能力和自适应能力就越强，但也更容易过拟合，尤其在小样本微调场景下。

3. r值越小，LoRA引入的参数就越少，模型表达能力受限，可能出现欠拟合，但更适合小样本微调，不容易过拟合。

4. 实验表明，r值在4-8之间往往能取得较好的平衡，再高的r值提升有限[8][12]。不过这个结论主要针对单一下游任务，对于指令微调这种多任务学习场景，可能需要更大的r值如16或32。

5. 一些经验法则是令r=d/k，其中d是原始矩阵的维度，k在64-512之间。或者设置α=2r，其中α控制LoRA权重的缩放[10]。

6. 最佳的r值取决于任务的复杂度和多样性。任务越复杂、数据集覆盖的指令越广泛，可能需要越大的r值。最终还是需要在特定任务上实验探索。

综上，LoRA中的r是个关键的超参数，控制了参数量和模型能力。一般建议r在4-32之间探索，在更复杂的任务上可以尝试更大的r值，但也要警惕过拟合风险。同时也可以参考一些经验法则来设置r的初始值。

Citations:
[1] https://www.thepaper.cn/newsDetail_forward_23250236
[2] https://blog.csdn.net/qq_43800752/article/details/130976670
[3] http://mingchao.wang/ShYWOOwr/
[4] https://juejin.cn/post/7314121672886599731
[5] https://www.51cto.com/article/756030.html
[6] https://hub.baai.ac.cn/view/33321
[7] https://www.lovesjava.com/d/3912.html
[8] https://www.cnblogs.com/gogoSandy/p/17363983.html
[9] http://www.360doc.com/content/23/1205/06/46368139_1106338508.shtml
[10] https://blog.csdn.net/qq_40714949/article/details/131988734
[11] https://github.com/mymusise/ChatGLM-Tuning/issues/59
[12] https://cloud.tencent.com/developer/article/2276508
[13] https://www.jiqizhixin.com/articles/2023-12-04-14
[14] https://blog.csdn.net/qq_41838627/article/details/133231964
[15] https://blog.cloudflare.com/id-id/fine-tuned-inference-with-loras-zh-cn