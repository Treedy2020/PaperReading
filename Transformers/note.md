## 目前国内的常见的大模型类型
根据以上搜索结果,目前主流的开源大模型体系包括:

1. LLaMA - Meta发布的大语言模型系列,有70亿、130亿、330亿和650亿参数规模的模型。LLaMA-13B在性能上可以超过GPT-3[9]。

2. Alpaca、Vicuna等在LLaMA基础上进行指令微调的模型[9][14]。

3. BLOOM - 由BigScience发布的176B参数多语言开源模型[9]。

4. GLM - 清华大学发布的双语(中英)预训练语言模型系列,最大规模达1300亿参数[9][13]。

5. ChatGLM - 清华大学KEG实验室发布的支持中英双语的对话语言模型,基于 General Language Model (GLM) 架构[13]。

6. MOSS - 复旦大学发布的支持中英双语的开源对话语言模型,最大规模160B参数[14]。

7. 悟道 - 智谱AI发布的双语多模态预训练模型,包括图文模型CogView、BriVL和语言模型GLM等[9]。

8. 盘古α - 华为诺亚方舟实验室发布的大规模中英文预训练语言模型,最大规模达200B参数[13]。

9. 通义千问 - 阿里达摩院发布的中文大规模语言模型,基于自研的混合专家-MoE技术[14]。

10. CPM系列 - 鹏城实验室发布的中英文预训练语言模型,如CPM-1、CPM-2、EVA等[14]。

此外还有一些如Megatron-LM、GPT-J、OPT、T5、FLAN-T5等由科技公司和研究机构开源的大模型。这些模型大多基于Transformer架构,通过在大规模语料上预训练,可以应用于对话、问答、摘要、写作等多种自然语言任务。开源社区也基于这些模型进行了大量的微调和应用拓展。

Citations:
[1] https://cloud.tencent.com/developer/article/2217896
[2] https://www.51cto.com/article/769820.html
[3] https://top.aibase.com/tool/yolo-nas-pose
[4] https://cloud.tencent.com/developer/article/2288178
[5] https://www.infoq.cn/article/9dlxasjhzypasptsz2gr
[6] https://openi.org.cn/html/2023/dongtai_0928/693.html
[7] https://cloud.tencent.com/developer/article/1855265
[8] https://www.jiqizhixin.com/articles/2023-05-19-4
[9] https://juejin.cn/post/7247089411803562040
[10] https://marketplace.huaweicloud.com/article/1-76780adb568e37a132deedac238fd089
[11] http://www.deeprlhub.com/d/703-2016-2022idea
[12] https://developer.aliyun.com/article/1283220
[13] https://github.com/HqWu-HITCS/Awesome-Chinese-LLM
[14] https://my.oschina.net/oscpyaqxylk/blog/8727824
[15] https://juejin.cn/post/7109424435534233630
[16] https://blog.csdn.net/allenhsu6/article/details/115622313
[17] https://developer.aliyun.com/article/1293994
[18] https://www.zhihu.com/question/49230922
[19] https://developer.aliyun.com/article/900193
[20] https://cloud.tencent.com/developer/article/1955325

## 常见集成学习方法回顾
根据以上搜索结果,我总结了常见的集成学习方法及其基本思想如下:

1. Bagging (Bootstrap Aggregating)
- 基本思想:从原始数据集中采用有放回抽样的方式构造多个训练子集,分别训练多个基学习器,然后将这些学习器的预测结果进行简单平均或加权平均作为最终预测。
- 代表算法:随机森林(Random Forest)
- 特点:可以并行训练,减少模型的方差,避免过拟合。

2. Boosting
- 基本思想:通过迭代训练一系列弱学习器,每次根据上一轮弱学习器的表现调整训练样本的权重,使得先前分类错误的样本在后续受到更多关注。最后将各个弱学习器的结果加权组合作为最终预测。
- 代表算法:AdaBoost, Gradient Boosting
- 特点:可以串行训练,减少模型的偏差,将弱学习器提升为强学习器。

3. Stacking
- 基本思想:将多个不同类型的学习算法组合,分为两层。第一层由多个基学习器组成,其输出结果作为第二层的输入特征,第二层的学习器根据这些特征做出最终预测。
- 特点:结合了多个异质模型,需要避免过拟合,训练过程相对复杂。

4. Voting
- 基本思想:对多个学习器的预测结果进行投票,得票最多的类别作为最终预测。可分为绝对多数投票、相对多数投票、加权投票等。
- 特点:直接利用多个模型的预测结果,实现简单。

5. Blending
- 基本思想:将数据划分为训练集和验证集,用训练集训练多个基学习器,然后用验证集去训练一个组合器,基于验证集上基学习器的预测结果,得到组合策略。
- 特点:引入验证集避免过拟合,但会减少训练数据。

总的来说,集成学习通过构建并结合多个学习器,利用单个学习器的优点,克服其缺点,从而获得比单个学习器显著优越的性能。不同的集成策略在学习器的构建方式、结合方式等方面各有特点。在实践中,要根据具体问题和数据特点,权衡集成的模型复杂度、训练开销和性能提升,选择合适的集成学习方法。

Citations:
[1] https://www.jiqizhixin.com/articles/2019-05-15-15
[2] https://www.cnblogs.com/ZihanZhang/p/16351469.html
[3] https://www.jiqizhixin.com/articles/2018-12-28-11
[4] https://blog.csdn.net/zwqjoy/article/details/80431496
[5] https://leovan.me/cn/2018/12/ensemble-learning/
[6] https://blog.csdn.net/weixin_39948381/article/details/106442535
[7] https://fuxi.163.com/database/694
[8] https://www.jiqizhixin.com/graph/technologies/29722de0-8501-4b01-9b73-189141b9eefd
[9] https://wfwei.github.io/posts/ensemble-methods/
[10] http://sklearn123.com/ThirdSection/1.11.Ensemble%20methods/
[11] https://shmc.tongji.edu.cn/88/6f/c32676a297071/page.htm
[12] https://developer.baidu.com/article/details/3141963


## 推导反向传播
反向传播算法（Backpropagation）是训练神经网络中最核心的算法之一，它通过计算损失函数相对于网络参数的梯度来更新网络的权重，以此最小化损失函数。反向传播算法的推导基于链式法则，以下是其基本步骤和推导过程：

1. **前向传播**：首先，网络进行前向传播，计算出每一层的输出值。对于一个简单的三层神经网络，输入层到隐藏层的计算可以表示为 $$z^{[1]} = W^{[1]}x + b^{[1]}$$，其中 $$x$$ 是输入，$$W^{[1]}$$ 和 $$b^{[1]}$$ 分别是权重和偏置，$$z^{[1]}$$ 是加权输入。激活函数应用于 $$z^{[1]}$$ 得到隐藏层的输出 $$a^{[1]} = \sigma(z^{[1]})$$，其中 $$\sigma$$ 是激活函数[4]。

2. **计算损失**：在网络的最后一层，计算预测输出与实际标签之间的损失。损失函数 $$L$$ 可以是均方误差、交叉熵等。

3. **反向传播误差**：反向传播的核心是计算损失函数相对于每个参数的梯度。这个过程从输出层开始，逆向通过网络进行。

    - 对于输出层，计算损失函数 $$L$$ 相对于激活函数输出的梯度 $$\frac{\partial L}{\partial a^{[L]}}$$。然后，根据链式法则，计算损失函数相对于加权输入的梯度 $$\frac{\partial L}{\partial z^{[L]}} = \frac{\partial L}{\partial a^{[L]}} \cdot \frac{\partial a^{[L]}}{\partial z^{[L]}}$$，其中 $$\frac{\partial a^{[L]}}{\partial z^{[L]}}$$ 是激活函数的导数[4][7]。

    - 接下来，计算损失函数相对于权重和偏置的梯度 $$\frac{\partial L}{\partial W^{[L]}}$$ 和 $$\frac{\partial L}{\partial b^{[L]}}$$。这些梯度可以用来更新权重和偏置：$$W^{[L]} = W^{[L]} - \alpha \frac{\partial L}{\partial W^{[L]}}$$，$$b^{[L]} = b^{[L]} - \alpha \frac{\partial L}{\partial b^{[L]}}$$，其中 $$\alpha$$ 是学习率[4][7]。

4. **反向传播到隐藏层**：对于隐藏层，需要计算损失函数相对于隐藏层输出的梯度，然后重复上述过程。对于第 $$l$$ 层，误差项 $$\delta^{[l]}$$ 可以表示为下一层的误差项 $$\delta^{[l+1]}$$ 乘以当前层权重的转置，再乘以当前层激活函数的导数：$$\delta^{[l]} = ((W^{[l+1]})^T \delta^{[l+1]}) \cdot \sigma'(z^{[l]})$$[4][7]。

5. **权重更新**：最后，使用计算出的梯度更新每一层的权重和偏置。

反向传播算法的关键在于利用链式法则逐层计算梯度，并更新权重以最小化损失函数。这个过程需要对每一层进行迭代，直到达到预定的迭代次数或损失函数收敛到一个小值。

Citations:
[1] https://harrymei.github.io/2018/12/14/%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%AE%97%E6%B3%95%E6%8E%A8%E5%AF%BC/
[2] https://github.com/INTERMT/BP-Algorithm
[3] https://blog.csdn.net/ft_sunshine/article/details/90221691
[4] https://martinlwx.github.io/zh-cn/backpropagation-tutorial/
[5] https://halfrost.com/neural_networks_learning/
[6] https://soptq.me/2020/10/11/bp-algorithm/
[7] https://www.cnblogs.com/jsfantasy/p/12177275.html
[8] https://github.com/EthanYuan/TensorFlow-Zero-to-N/blob/master/TensorFlow%E4%BB%8E0%E5%88%B0N/TensorFlow%E4%BB%8E0%E5%88%B01/10-NN%E5%9F%BA%E6%9C%AC%E5%8A%9F%EF%BC%9A%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%9A%84%E6%8E%A8%E5%AF%BC.md