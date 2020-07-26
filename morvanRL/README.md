# 强化学习 Reinforcement Learning (莫烦 Python 教程)

## 1 什么是 强化学习 (Reinforcement Learning)

### 1.1 对比监督学习

![https://morvanzhou.github.io/static/results/ML-intro/RL3.png](https://morvanzhou.github.io/static/results/ML-intro/RL3.png)

- 监督学习, 是已经有了数据和数据对应的正确标签
- 强化学习一开始并没有数据和标签。他要通过一次次在环境中的尝试, 获取这些数据和标签, 然后再学习通过哪些数据能够对应哪些标签, 通过学习到的这些规律, 尽可能地选择带来高分的行为 (比如这里的开心脸). 这也就证明了在强化学习中, 分数标签就是他的老师, 他和监督学习中的老师也差不多.

### 1.2 RL算法家族

![https://morvanzhou.github.io/static/results/ML-intro/RL4.png](https://morvanzhou.github.io/static/results/ML-intro/RL4.png)

- 比如有通过行为的价值来选取特定行为的方法, 包括使用表格学习的 Q-learning, Sarsa, 使用神经网络学习的 Deep Q Network...
- 还有直接输出行为的 policy gradients
- 又或者了解所处的环境, 想象出一个虚拟的环境并从虚拟的环境中学习 等等.

## 2 强化学习方法汇总

### 2.1 Model-free 和 Model-based 

- 我们可以将所有强化学习的方法分为理不理解所处环境,如果我们不尝试去理解环境, 环境给了我们什么就是什么. 我们就把这种方法叫做`model-free`, 这里的 model 就是用模型来表示环境, 那理解了环境也就是学会了用一个模型来代表环境, 所以这种就是` model-based `方法. 
- `Model-free` 的方法有很多, 像 [Q learning](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/4-03-q-learning/), [Sarsa](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/4-04-sarsa/), [Policy Gradients](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/4-07-PG/) 都是从环境中得到反馈然后从中学习. 而 `model-based` RL 只是多了一道程序, 为真实世界建模, 也可以说他们都是 `model-free `的强化学习, 只是 `model-based` 多出了一个虚拟环境, 
- `Model-free` 中, 机器人只能按部就班, 一步一步等待真实世界的反馈, 再根据反馈采取下一步行动. 而 `model-based`, 他能通过想象来预判断接下来将要发生的所有情况. 然后选择这些想象情况中最好的那种. 并依据这种情况来采取下一步的策略

### 2.2 基于概率 和 基于价值 

![https://morvanzhou.github.io/static/results/ML-intro/RLmtd2.png](https://morvanzhou.github.io/static/results/ML-intro/RLmtd2.png)

- **基于概率是强化学习中最直接的一种, 他能通过感官分析所处的环境, 直接输出下一步要采取的各种动作的概率, 然后根据概率采取行动, 所以每种动作都有可能被选中, 只是可能性不同**. 
  - 对于选取连续的动作, 基于价值的方法是无能为力的. 我们却能用一个概率分布在连续动作中选取特定动作, 这也是基于概率的方法的优点之一
- **而基于价值的方法输出则是所有动作的价值, 我们会根据最高价值来选着动作**, 相比基于概率的方法, 基于价值的决策部分更为铁定, 毫不留情, 就选价值最高的, 而基于概率的, 即使某个动作的概率最高, 但是还是不一定会选到他.
- 比如在基于概率这边, 有 [Policy Gradients](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/4-07-PG/), 在基于价值这边有 [Q learning](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/4-03-q-learning/), [Sarsa](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/4-04-sarsa/) 等. 而且我们还能结合这两类方法的优势之处, 创造更牛逼的一种方法, 叫做 [Actor-Critic](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/4-08-AC/), actor 会基于概率做出动作, 而 critic 会对做出的动作给出动作的价值, 这样就在原有的 policy gradients 上加速了学习过程.

### 2.3 回合更新 和 单步更新 

![RLmtd3](https://morvanzhou.github.io/static/results/ML-intro/RLmtd3.png)

- 强化学习还能用另外一种方式分类, 回合更新和单步更新, 想象强化学习就是在玩游戏, 游戏回合有开始和结束.
- **回合更新**指的是游戏开始后, 我们要等待游戏结束, 然后再总结这一回合中的所有转折点, 再更新我们的行为准则. 
- **单步更新**则是在游戏进行中每一步都在更新, 不用等待游戏的结束, 这样我们就能边玩边学习了
- Monte-carlo learning 和基础版的 policy gradients 等都是回合更新制, Qlearning, Sarsa, 升级版的 policy gradients 等都是单步更新制. 因为单步更新更有效率, 所以现在大多方法都是基于单步更新. 比如有的强化学习问题并不属于回合问题.

### 2.4 在线学习 和 离线学习 

![RLmtd4](https://morvanzhou.github.io/static/results/ML-intro/RLmtd4.png)

- 所谓在线学习, 就是指我必须本人在场, 并且一定是本人边玩边学习,
- 而离线学习是你可以选择自己玩, 也可以选择看着别人玩, 通过看别人玩来学习别人的行为准则, 离线学习 同样是从过往的经验中学习, 但是这些过往的经历没必要是自己的经历, 任何人的经历都能被学习. 
- 最典型的在线学习就是 Sarsa 了, 还有一种优化 Sarsa 的算法, 叫做 Sarsa lambda, 最典型的离线学习就是 Q learning, 后来人也根据离线学习的属性, 开发了更强大的算法, 比如让计算机学会玩电动的 Deep-Q-Network.



## 3 Q-learning

- 学出`Q表(Q-table)`

## 参考资料

