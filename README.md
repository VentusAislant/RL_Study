# Reinforcement Learning  学习

## 1. 强化学习基本概念

强化学习可以被认为是智能体 (Agent) 在环境中**试错式学习最优策略** (Policy) 的过程，其核心是马尔科夫决策过程 (Markov Decision Process)

### 1.1 马尔科夫决策过程 (MDP)
- MDP 是 RL 的数学基础，由一个五元组构成:
$$
    MDP = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)
$$

- 符号含义如下:

    | 符号 | 含义 | 
    | ----| --------|
    | $\mathcal{S}$| 状态空间（state space）|
    | $\mathcal{A}$| 动作空间（action space）|
    | $\mathcal{P}(s\prime \mid s, a)$| 在状态 $s$ 下采取动作 $a$ 转移到下一个状态 $s\prime$ 的概率|
    | $\mathcal{R}(s,a)$| 奖励函数: 在状态 s 采取动作 a 后获得的期望奖励|
    | $\gamma \in [0,1]$| 折扣因子: 权衡当前奖励和未来奖励的重要性|

- 马尔科夫性质: MDP满足马尔科夫性质 (Markov Property), 即 *状态的转移只与当前状态 $s_t$ 和 当前动作 $a_t$ 有关, 与过去状态和动作无关*。形式化表达为:
$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1},...,s_0,a_0)=P(s_{t+1}|s_t,a_t)
$$
​	*这意味着当前状态 $s_t$ 已经包含了过去所有历史信息的充分统计量*

- 强化学习的交互流程 

    - 环境处于状态 $s_t$
    - Agent 根据策略 $\pi(a_t\mid s_t)$ 选择一个动作 $a_t$
    - 环境转移到下一个状态 $s_{t+1} \sim \mathcal{P}(s_{t+1} \mid s_t, a_t)$
    - Agent 接收一个奖励 $r_t=\mathcal{R}(s_t, a_t)$

- 强化学习的目标:
  
    - 学习一个策略 $\pi^*(a \mid s)$, 最大化从任意状态 $s$ 出发的期望累积回报
    - 累积回报定义为: 
    $$
    G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}
    $$

    - *$r_{t+k}$: 第 $t+k$ 步的即时奖励*
    - *$\gamma$: 折扣因子，控制未来奖励的重要性，越接近1越注重长期*

- 例子: CartPole 中的 MDP

    | MDP 组件 | 实例 |
    | ---- | --------|
    |状态空间 $\mathcal{S}$|小车位置、速度、杆速度、角速度 (连续空间)|
    |动作空间 $\mathcal{A}$|推左 / 推右 (离散空间)|
    |状态转移 $\mathcal{P}$|由物理动力学决定|
    |奖励函数 $\mathcal{R}$|每步+1,直到杆倒下|
    |折扣因子 $\gamma$|通常设为 0.99|



## 2. 强化学习实践

- 一些强化学习的资料:

  - [OpenAI Gymnasium](https://gymnasium.farama.org/introduction/basic_usage/)
  - [OpenAI: Spinning Up in Deep RL! ](https://spinningup.openai.com/en/latest/)

  - [Ray RLlib](https://docs.ray.io/en/latest/index.html)
  - [cleanrl](https://github.com/vwxyzjn/cleanrl)

### 2.1 Gymnasium

- 环境安装:

  - 创建环境

    ```sh
    conda create -n rl python=3.10 -y
    conda activate rl
    ```

  - 安装相关的包

    ```
    # 安装 torch 相关
    pip install torch torchvision torchaudio pycharm
    
    # 安装 Gymnasium
    pip install gymnasium
    pip install gymnasium[all]
    ```

  - 如果使用的是conda管理则会有问题

    ```
    libGL error: MESA-LOADER: failed to open iris: /home/ventus/Apps/dev/anaconda/envs/rl/bin/../lib/libstdc++.so.6: version GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1) (search paths /usr/lib/x86_64-linux-gnu/dri, suffix _dri)
    libGL error: failed to load driver: iris
    libGL error: MESA-LOADER: failed to open swrast: /home/ventus/Apps/dev/anaconda/envs/rl/bin/../lib/libstdc++.so.6: version GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1) (search paths /usr/lib/x86_64-linux-gnu/dri, suffix _dri)
    libGL error: failed to load driver: swrast
    X Error of failed request:  BadValue (integer parameter out of range for operation)
      Major opcode of failed request:  149 (GLX)
      Minor opcode of failed request:  3 (X_GLXCreateContext)
      Value in failed request:  0x0
      Serial number of failed request:  163
      Current serial number in output stream:  164
    ```

  - 解决方案:

    - 使用系统的 `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` 而不是 conda 安装的

    - 操作流程

      ```sh
      # 找到 anaconda 对应环境中的 libstdc++.so.6
      # 这里是自己环境的 lib 目录
      cd ~/Apps/dev/anaconda/envs/rl/lib
      
      # 里面有三个关键文件
      ls -la libstdc++.so*
      
      # libstdc++.so -> libstdc++.so.6.0.29
      # libstdc++.so.6 -> libstdc++.so.6.0.29
      # libstdc++.so.6.0.29
      
      # 将系统的 libstdc++.so.6 copy到当前的 libstdc++.so.6.x.xx 即可
      cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./libstdc++.so.6.0.29
      ```


#### 2.1.1 Basic Usage

- 什么是强化学习?
  - 强化学习就像是通过试错进行教学: 一个智能体通过尝试动作，接受反馈 (奖励)，并逐渐改进行为来学习。
  - 可以把它想象成用零食训练宠物、通过练习学骑自行车，或通过反复游玩掌握一款电子游戏。
- 为什么使用 Gymnasium?
  - 无论你是想训练智能体玩游戏、控制机器人，还是优化交易策略，Gymnasium 都为你提供了构建和测试想法的工具
  - Gymnasium 的核心是一套用于所有单智能体强化学习环境的 API，并实现了常见的环境，如: CartPOle, Pendulum, MountainCar, Mujoco, Atari等
  -  Gymnasium 提供了四个关键函数: `make()`, `Env.reset()`, `Env.step()`, `Env.render()`
  - Gymnasium 的核心是 `Env`, 这是一个高级的 Python 类，表示强化学习理论中的马尔科夫决策过程 (MDP), *注意: 这不是对MDP的完全还原，缺少一些组成部分*
    - 这个类是的用户能够开始一轮新的交互、执行动作并可视化智能体的当前状态
  - 除了 `Env` 之外，Gymnasium 还提供了 `Wrapper`， 用于增强或修改环境，特别是对智能体的观察值、奖励以及所采取的动作

