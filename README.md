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
    pip install torch torchvision torchaudio pygame
    
    # 安装 Gymnasium
    pip install gymnasium
    
    # 安装所有环境
    sudo apt install swig 
    pip install gymnasium[all]
    ```
  
  - 如果使用的是conda管理则可能有问题 (如没遇到相关报错则忽略)
  
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

- 初始化 Gymnasium 环境

  - 在 Gymnaisum 中初始化环境非常简单，可以通过 `make()` 函数完成:

    ```python
    import gymnasium as gym
    
    # 创建一个 CartPloe游戏环境
    env = gym.make('CartPole-v1')
    ```

  - 这个 `CartPole` 环境的任务是: 在一辆移动的小车上保持一根杆子的平衡

    - 简单但不平凡
    - 训练速度快
    - 有明确的成功/失败标准

  - 这个函数会返回一个 `Env` 对象，供用户进行交互

  - 如果想查看所有可创建的环境，可以使用 `pprint_registry()`

  - 此外 `make()` 函数还提供了许多其他参数，用于为环境指定关键词、添加或移除 `wrappers` 等

- 理解 Agent-Environment 循环

  - 在强化学习中，经典的 "智能体-环境"循环，如下图所示，代表了学习在RL中如何发生

    <img src="./.typora_pic/AE_loop.png" alt="img" style="zoom: 15%;" />

    - Observation: 智能体观察当前情境 (就像在看游戏画面)
    - Action: 智能体根据观察到的内容选择一个动作 (就像按下一个按钮)
    - Reward: 环境根据动作反馈一个新的情境和奖励 (游戏状态发生变化，分数更新)
    - 重复此过程，直到当前回合结束

- 第一个 RL 程序

  - 以一个简单的 CartPole 游戏来理解强化学习的基础

    ```python
    # Run `pip install "gymnasium[classic-control]"` for this example.
    import gymnasium as gym
    
    # Create our training environment - a cart with a pole that needs balancing
    env = gym.make("CartPole-v1", render_mode="human")
    
    # Reset environment to start a new episode
    observation, info = env.reset()
    # observation: what the agent can "see" - cart position, velocity, pole angle, etc.
    # info: extra debugging information (usually not needed for basic learning)
    
    print(f"Starting observation: {observation}")
    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
    # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    
    episode_over = False
    total_reward = 0
    
    while not episode_over:
        
        # Choose an action: 0 = push cart left, 1 = push cart right
        action = env.action_space.sample()  # Random action for now - real agents will be smarter!
    
        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)
    
        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)
    
        total_reward += reward
        episode_over = terminated or truncated
    
    print(f"Episode finished! Total reward: {total_reward}")
    env.close()
    ```

- 详细解释

  - 首先我们会使用 `Env.make()` 函数创建一个环境，可以带一个可选的 `render_mode` 参数， 这个参数用于指定环境应如何可视化
    - `render_mode=human`: 用户会看到一个可视化窗口
    - `render_mode=rgb_array`: 用户会获得图像数组
    - `render_mode=None`: 无可视化运行，对于训练来说是最快的
  - 在初始化环境之后，我们使用 `Env.reset()` 来重置环境，从而获得第一个观测值以及额外信息。这就类似开始一个新游戏或新一轮。可以在 `reset()` 中使用 `seed` 或 `options` 参数来用特定的随机种子或选项初始化环境
  - 因为我们希望持续进行 `Agent-Environment` 循环，直到环境终止, 所以通过定义变量 `epidode_over` 来控制 while 循环
  - 接下来智能体通过 `Env.step()` 在环境中执行一个动作, 本例中使用了 `env.action_space.sample()` 进行随机决策来更新环境
    - 这个动作可以被想象成移动一个机器人，按下游戏控制器上的按钮，或者作出一个交易决策
    - 结果是智能体从更新后的环境中接收到一个新的观测值，以及因执行该动作而得到的奖励
      - 这个奖励可能因为好的动作而是正值 (比如成功平衡杆子)
      - 或者也可能因为坏的动作而是负值 (比如让杆子倒下)
    - 这样一次动作-观测的交换成为一个时间步 (timestep)
  - 然而，在若干时间步之后，环境可能会结束，这被成为终止状态。例如机器人追欢，或者任务完成，又或者我们希望在固定的时间步之后停止
    - 在 Gymnasium 中，如果由于任务完成或失败导致环境终止，这会通过 `step()` 返回 `terminated=True`
    - 如果我们希望环境在固定的时间步之后结束，环境会发出一个 `truncated=True` 
    - 如果 `terminated` 或 `truncated` 中任意一个为 True, 我们就结束这一轮
    - 在大多数情况下，我们将使用 `env.reset()` 重新开始环境，以启动新的一轮

- Action Space 和 Observation Space

  - 每个环境 `Env` 都通过 `action_space` 和 `observation_space` 属性指定有效动作和观测的格式。

  - 这对于了解环境所希望的输入和输出非常有帮助，因为所有有效的动作和观测都应包含在他们各自的空间中

  - 理解这些空间对构建智能体至关重要:

    - 动作空间 Action Space: 你的智能体能做什么? (离散选择、连续值等)
    - 观测空间 Observation Space: 你的智能体能看到什么? (图像、数字、结构化数据)

  - 重要的是 `Env.action_space` 和 `Env.observation_space` 是 `Space` 的实例，这是一个高级的 Python 类，提供了关键函数: `Space.contains()` 和 `Space.sample()`

  - Gymnasium 支持多种类型的空间:

    - `Box`: 描述一个具有上下界的任意 n 维形状的空间 (例如连续控制或图像像素)
    - `Discrete`: 描述一个离散空间，其可能值为 {0, 1, ..., n-1} (例如按钮按压或菜单选择)
    - `MultiBinary`: 描述一个具有任意 n 维形状的二进制空间 (例如多个开关)
    - `MultiDiscrete`: 由一系列具有不同元素动作数量的离散动作空间组成。
    - `Text`: 描述具有最小和最大长度的字符串空间
    - `Dict`: 描述一个更简单空间组成的字典
    - `Tuple`: 描述一个简单空间的元组
    - `Graph`: 描述一个带有互联节点的和边的数学图
    - `Sequence`: 描述一个变长的简单元素序列

  - 实例:

    ```python
    import gymnasium as gym
    
    # 离散动作空间（按钮按压）
    env = gym.make("CartPole-v1")
    print(f"Action space: {env.action_space}")  # Discrete(2) - 向左或向右
    print(f"Sample action: {env.action_space.sample()}")  # 0 或 1
    
    # Box 观测空间（连续值）
    print(f"Observation space: {env.observation_space}")  # 包含 4 个值的 Box
    # Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
    print(f"Sample observation: {env.observation_space.sample()}")  # 随机有效观测值 0 / 1
    ```

- 修改环境

  - `Wrappers` 包装器是一个方便的方式，可以在不直接修改底层代码的情况下更改现有环境
    - 可以将包装器想象成改变你与环境交互方式的过滤器或修改器
    - 使用包装器可以避免样本代码，并使你的环境更加模块化
    - 包装器还可以进行链式组合，以叠加它们的效果
  - 大多数 `gymnasium.make()` 创建的环境，默认已经通过以下包装器进行了包装:
    - `TimeLimit`: 在最大步数后终止回合
    - `OrderEnforcing`: 确保 reset 和 step 调用顺序正确
    - `PassiveChecker`: 验证你对环境的使用是否正确

  - 如果你想要包装一个环境，首先要初始化一个基础环境，然后将其可选参数一起传递给包装器的构造函数

    ```python
    import gymnasium as gym
    from gymnasium.wrappers import FlattenObservation
    
    # 从一个复杂的观测空间开始
    env = gym.make("CarRacing-v3")
    print(env.observation_space.shape)
    # (96, 96, 3)  # 96x96 的 RGB 图像
    
    # 使用包装器将观测扁平化为一个一维数组
    wrapped_env = FlattenObservation(env)
    print(wrapped_env.observation_space.shape)
    # (27648,)  # 所有像素合并为一个数组
    
    # 这样更便于使用一些期望一维输入的算法
    ```

  - 一些新手常用的包装器包括:

    - `TimeLimit`: 如果超出了最大时间步数，就发出一个 `truncated` 信号，防止无限回合
    - `ClipAction`: 将传递给 `step` 的动作裁剪到有效动作空间的范围内
    - `RescaleAction`: 将动作重新所放到另一个范围 (适用于输出范围为 [-1, 1] 的算法，但环境期望的是 [0, 10]等)
    - `TimeAwareObeservation`: 向观测中添加当前时间步的信息，有助于学习

  - 可以在 [wrappers](https://gymnasium.farama.org/api/wrappers/) 中查看 Gymnasium 中所有已实现的包装器

  - 如果你现在有一个被包装的环境，但是想访问最底层的原始环境，可以使用 `unwrapped` 属性，如果该环境已经是基础环境，那么 `unwrapped` 仅返回自身

- 新手常见问题

  - 智能体表现为随机行为: 当你使用 `env.action_space.sample()` 时，这是预期的，在之后的强化学习训练过程中会替换这个策略
  - 回合立即结束: 检查你是否正确的在回合之间执行了 reset

#### 2.1.2 Training an Agent

- 引言
  - 当我们谈论训练一个强化学习 (RL) 智能体时，我们是在通过经验教它作出好的决策
  - 与监督学习不同，在监督学习中我们会展示正确答案的示例，而强化学习的智能体是通过尝试不同的动作并观察结果来学习的，就像是骑自行车，你尝试不同的动作，摔倒几次，逐渐学会什么是有效的
  - 强化学习的最终目标是学习一个 **策略 (Policy)**， 一种告诉智能体在每种情况下应如何采取何种动作以最大化长期回报的策略

- 直观理解 Q-Learning
  - 在本教程中将使用 `Q-learning` 来解决 Blackjack (黑杰克) 环境，首先我们需要理解 Q-learning 是如何工作的
  - Q-learning 会构建一张巨大的 "秘籍表", 称为 Q表 (Q-table), 告诉智能体在每种情况下采取每个动作的好坏
    - 行 (Rows): 智能体可能遇到的不同情况 (状态)
    - 列 (Columns): 智能体可以采取的不同动作
    - 值 (Values): 在该状态下采取该动作的好坏 (预期未来奖励)
  - 以 Blackjack 为例
    - 状态 (States): 你的手牌总值, 庄家明牌, 你是否有可用的 A
    - 动作 (Actions): 要牌 (Hit) 或停牌 (Stand)
    - Q值 (Q-values): 在每个状态下执行每个动作的期望奖励
  - 学习过程
    - 尝试一个动作，观察结果 (获得奖励和新状态)
    - 更新秘籍表: 这个动作比预期好/差，就更新对应的 Q 值
    - 通过反复尝试与更新估计来逐步改进策略
    - 在探索与利用之间寻找平衡: 
      - 探索 (exploration): 尝试新动作以获取更多信息
      - 利用 (exploitation): 选择你已知效果最好的动作
  - 为什么有效？
    - 随着时间推移，好的动作会拥有更高的 Q-value, 差的动作则拥有更低的 Q-alue，最终智能体学会选择预期奖励最高的动作，从而实现最有策略

- 了解环境: BlackJack
  - BlackJack是最受欢迎的赌场纸牌游戏之一，同时也是强化学习的绝佳入门环境，其包括一下特点：
    - 规则清晰：尽量让你的手牌总点数接近21,但不能超过，同时要比庄家高
    - 观察简单：包括你的手牌点数，庄家明牌，以及你是否拥有一个可用的 A
    - 动作离散：0=停牌 (stand) 1=要牌 (Hit)
    - 即使反馈：每一局游戏结束后立即得知输赢或平局
  - 当前环境使用了 *无限副牌 (抽牌后放回)*, 因此无法通过记牌策略取胜，智能体只能通过试错方式学习最优策略
  - 环境细节
    - 观察空间 (Observation): 一个元组, 形式为 `(player_sum, dealer_card, usable_ace)`
      - `player_sum`: 玩家当前手牌的总点数, (范围: 4 - 21)
      - `dealer_card`: 庄家的明牌点数, (范围:1 - 10)
      - `usable_ace`: 玩家是否有一个可用的 A, (布尔值, True/False)
    - 动作空间 (Action): 
      - `0`: 停牌 (Stand)
      - `1`: 要牌 (Hit)
    - 奖励 (Rewards):
      - `+1`: 赢
      - `-1`: 输
      - `0`: 平局
    - 回合结束条件 (Episode ends):
      - 玩家选择停牌 (Stand)
      - 玩家爆牌 (Bust: 点数超过21)

- 执行动作

  - 在通过 `env.reset()` 获得初始观察值之后，我们使用 `env.step(action)` 来与环境进行交互，这个函数接受一个动作并返回五个重要的值

    ```python
    observation, reward, terminated, truncated, info = env.step(action)
    ```

    - `observation`: 执行该动作后，智能体看到的环境状态
    - `reward`: 该动作带来的即时反馈
    - `terminated`: 表示该回合是否自然结束
    - `truncated`: 表示该回合是否因时间限制等外部原因被中止
    - `info`: 附加调试信息

  - 关键点: `reward` 仅告诉我们**当前动作的即时好坏**，但智能体需要学习的是**长期结果的好坏** 

  - Q-Learning的核心就是通过估算**总的未来奖励 (累计回报),** 而不是仅仅依赖即时奖励, 从而作出更优的决策

- 构建一个 Q-Learning Agent

  - 构建一个智能体，需要实现以下几个关键功能: 

    - 选择动作 (探索 / 利用)
    - 从经验中学习 (更新 Q 值)
    - 管理探索程度 （随着时间推移降低随机性）

  - Exploration vs. Exploitation

    - 探索和利用是强化学习的一个核心问题
      - 探索 (Exploration): 尝试新动作, 以了解环境可能的反馈
      - 利用 (Exploitation): 使用已有的知识选择当前来看最好的动作，以获得最大回报
    - $\epsilon$-贪婪策略: 我们通常使用 $\epsilon$-贪婪策略 来在探索和利用之间做权衡
      - 以概率 $\epsilon$: 随机选择一个动作 (探索)
      - 以概率 $1-\epsilon$: 选择当前Q值最高的动作 (利用)
    -  实践建议: 从较高的 $\epsilon$ 开始 (鼓励多探索), 随着学习过程的推进逐渐降低 $\epsilon$ 值 (逐步转向利用)

  - BlackjackAgent 的代码实现

    ```python
    import gymnasium as gym
    from collections import defaultdict
    import numpy as np  
    
    class BlackjackAgent:
        def __init__(
            self, 
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
        ):
            """ Initialize the Blackjack agent.
            
            Args:
                env (gym.Env): The Blackjack environment.
                learning_rate (float): How quickly to update Q-values (0-1)
                initial_epsilon (float): Starting exploration rate (usually 1.0)
                epsilon_decay (float): How much to reduce epsilon each episode
                final_epsilon (float): Minimum exploration rate (usually 0.1)
                discount_factor (float): How much to value future rewards (0-1)
            """
            self.env = env
            
            # Q_Table: maps(state, action) -> Q-value
            # defaultdict automatically creates a new entry with 0.0 if not found
            # print(env.action_space.n)  # 2
            self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
    
            self.lr = learning_rate
            self.discount_factor = discount_factor
    
            # exploration parameters
            self.epsilon = initial_epsilon
            self.epsilon_decay = epsilon_decay
            self.final_epsilon = final_epsilon
    
            # Trace learning progress
            self.training_error = []
    
        def get_action(self, obs: tuple[int, int, bool]) -> int:
            """ Choose an action based on epsilon-greedy policy.
            
            Args:
                obs (tuple): Current observation (player_sum, dealer_card, usable_ace)
            
            Returns:
                int: Action to take (0 = stand, 1 = hit)
            """
    
            # expoloration
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()
            else:
                # 如果不存在 Q-value for this state, defaultdict 会默认调用 lambda 返回一个零向量
                return int(np.argmax(self.q_values[obs]))
            
        def update(
            self, 
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool]
        ):
            """ Update Q-values based on the action taken and the reward received.
            
            Args:
                obs (tuple): Current observation (player_sum, dealer_card, usable_ace)
                action (int): Action taken (0 = stand, 1 = hit)
                reward (float): Reward received after taking the action
                terminated (bool): Whether the episode has ended
                next_obs (tuple): Next observation after taking the action
            """
    
            # 计算未来最大收益
            future_q_value = (not terminated) * np.max(self.q_values[next_obs])
    
            finual_q_value = reward + self.discount_factor * future_q_value
    
            # how wrong was our current estimate?
            temporal_difference = finual_q_value - self.q_values[obs][action]
    
            # Update our estimate in the direction of the error
            # Learning rate controls how big steps we take
            self.q_values[obs][action] = (
                self.q_values[obs][action] + self.lr * temporal_difference
            )
    
             # Track learning progress (useful for debugging)
            self.training_error.append(temporal_difference)
    
        def decay_epsilon(self):
            """Reduce exploration rate after each episode."""
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    ```

  - 理解 Q-Learnng

    - 目标：找到一个最优策略，使得智能体 在每个状态下选择动作都能获得最大的长期累计奖励 (即最大Q-Value)

    - 背后的数学思想是：贝尔曼方程 (Bellman Equation), 公式形式如下:
      $$
      Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\ max_{a \prime}Q(s\prime,a\prime)-Q(s,a)]
      $$

      - $Q(s,a)$: 表示在状态 s 下采取动作 a 得到的 Q值
      - $\alpha$: 表示学习率，表示更新Q值的速率
      - $r$: 表示当前动作获得奖励
      - $\gamma$: 表示折扣因子，代表未来最大奖励的权重，权重越大越考虑长期累计奖励

    - 代码详解

      ```python
      # 当前估计：Q(状态, 动作)
      current_q = self.q_values[obs][action]
      
      # 实际经历到的结果：即时奖励 + 折扣后的未来最大奖励
      target = reward + self.discount_factor * max(self.q_values[next_obs])
      
      # 我们的误差有多大？
      error = target - current_q
      
      # 更新估计值：朝目标值靠近一点
      new_q = current_q + learning_rate * error
      ```

    - 关键含义是：一个状态-动作对的价值应该等于当前获得的奖励+未来最优行为的折扣价值

    - 一开始我们的未来最优行为的Q值是零初始化的，但是一开始模型倾向于探索而非利用，逐渐探索到比较好的最又行为，从而推动优化

- 训练 Agent

  - 训练流程:

    1. 重置环境：开始新的一轮游戏
    2. 玩完整的一轮游戏 (epsiode): 选择动作，并从每一步中学习
    3. 更新探索率 (逐渐减少 epsilon)
    4. 重复很多轮游戏来学习好的策略

  - 代码

    ```python
    def train(
        lr=0.01,
        n_episodes=100_000,
        start_epsilon=1.0,
        final_epsilon=0.1,
    ):
        """ 训练 Agent
        Args:
            lr (float): How fast to learn (higher = faster but less stable)
            n_episodes (int): Number of hands to practice
            start_epsilon (float): Start with 100% random actions
            final_epsilon (float): Always keep some exploration
        """
        from tqdm import tqdm
        
        epsilon_decay = start_epsilon / (n_episodes/2)
        
        env = gym.make("Blackjack-v1", sab=False)
        env=gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
        
        agent = BlackjackAgent(
            env=env,
            learning_rate=lr,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=0.99
        )
    
        for episode in tqdm(range(n_episodes), desc="Training"):
            # 开始新的一轮游戏
            obs, _ = env.reset()
            done = False
            
            # 完成完整的一轮
            while not done:
                # Agent选择一个动作 (一开始比较随机，之后逐渐智能)
                action = agent.get_action(obs)
                
                # 采取动作后，的观察结果，奖励等
                next_obs, reward, terminated, truncated, _ = env.step(action)
                
                # agent 从返回值中学习
                agent.update(obs, action, reward, terminated, next_obs)
    			
                # 进行下一步
                done = terminated or truncated
                obs = next_obs
            
            # 减少探索率
            agent.decay_epsilon()
    
    ```

  - 训练过程中会发生什么？

    - 早期阶段 (第0-10,000局)
      - 智能体的行为大多数是随机的 (因为 $\epsilon$ 很高)
      - 胜率约为 43%
      - Q值非常不准
    - 中期阶段 (第10,000-50,000局)
      - 智能体开始发现有效策略
      - 胜率提升至45%-48%
      - 随着估计值变得更准确，学习误差开始下降
    - 后期阶段 (第50,000局之后)
      - 智能体逐渐收敛到近似最优策略
      - 胜率趋于稳定到49%左右，这是该环境下的理论上限
      - Q值逐渐稳定，学习误差变得非常小

- 分析训练结果

  - 可视化训练过程

    ```python
    def visualize(agent):
        """可视化训练过程"""
        def get_moving_avgs(arr, window, convolution_mode='valid'):
            """计算移动平均值来平滑噪音数据"""
            # np.array(arr).flatten() 将 arr 转为 numpy 数组并展平
            # np.ones(window)  生成一个长度为 window 的数组，值全是1
            # np.convolve(..., mode=convolution_mode) 对前面的两个数组做加权平均，权重都为1,相当于滑动求和
            return np.convolve(
                np.array(arr).flatten(),
                np.ones(window),
                mode=convolution_mode
            ) / window
    
        from matplotlib import pyplot as plt
    
        # 定义滑动窗口长度为 500，用来平滑图像曲线
        rolling_length = 500
    
        # 创建3列子图，分别用于绘制奖励、回合长度、训练误差
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    
        # 奖励相关可视化
        axs[0].set_title("Episode rewards")
        reward_moving_average = get_moving_avgs(
            agent.env.return_queue,
            rolling_length,
            "valid"
        )
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[0].set_ylabel("Average Reward")
        axs[0].set_xlabel("Episode")
    
        # 回合数相关可视化
        axs[1].set_title("Episode lengths")
        length_moving_average = get_moving_avgs(
            agent.env.length_queue,
            rolling_length,
            "valid"
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[1].set_ylabel("Average Reward")
        axs[1].set_xlabel("Episode")
    
        # 训练误差
        axs[2].set_title("Training Error")
        training_error_moving_average = get_moving_avgs(
            agent.training_error,
            rolling_length,
            "same"
        )
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
        axs[2].set_ylabel("Temporal Difference Error")
        axs[2].set_xlabel("Step")
    
        plt.tight_layout()
        plt.show()
    ```

  - 可视化解释

    - 奖励图 (Reward Plot): 奖励图应该显示出从负值逐渐接近0，Blackjack是一款本身就不利于玩家的游戏，即使是完美策略，也会因为赌场优势而略微亏损
    - 每局动作数 (Episode length): 回合长度应该稳定在每局2-3个动作之间，如果非常短说明agent过早选择stand，如果非常长，说明agent过度使用hit
    - 训练误差 (Training Error / TD Error): 
    - 训练误差应当随时间逐渐减小，说明 agent 对环境的预测变得更准确，在训练早期出现较大波动是正常的，因为 agent 遇到很多新情况

- 训练常见问题及解决

  - Agent 没有进步 (Reward 一直没变化)
    - 症状: 平均奖励保持不变; TD误差一直很大; 学了很多轮但表现没有任何提升
    - 可能原因: 学习率太高或太低; 奖励设计不合理; Q表没更新, 因为逻辑写错了
    - 解决方法: 尝试学习率在 0.001 - 0.1 之间; 确保 Blackjack 的奖励是合理的
    - 打印调试信息确认 Q表真的在变化
  - 训练过程不稳定
    - 症状: 奖励忽高忽低; 表现波动剧烈, 难以收敛
    - 可能原因: 学习率太高; 探索不足, agent陷入局部策略
    - 解决方法: 减少学习率, 设置最低探索概率 $finlal\_\epsilon \ge 0.05$; 多训练一些回合让策略更稳定
  - Agent 被卡在错误策略中
    - 症状: 一开始表现提升, 但后期停滞; 1最终策略表现明显不优
    - 可能原因: 探索太少, 太早exploitation; 学习率太低, 更新缓慢
    - 解决方法: 降低 epsilon 衰减速度; 初始时设置更高的学习率; 使用不同的探索方法, 如乐观初始化Q表, 把Q值设的很高
  - 学习太慢
    - 症状: 能学但是学的很慢; 奖励缓慢上升或者TD误差下降非常缓慢
    - 可能原因: 学习率台地; 探索太多(比如epsilon太大，一直在试探)
    - 解决方法: 稍微调高学习率, 更快地降低epsilon, 聚焦训练困难的状态(加权训练, 修改奖励函数)

- 测试 Agent

  - 代码:

    ```python
    def test(agent, num_episodes=1000):
        """测试 Agent的表现"""
        total_rewards = []
        env = agent.env
    
        # 不让模型探索
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0
    
        for episode in tqdm(range(num_episodes), desc="Test"):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
        
        # 恢复agent的探索率
        agent.epsilon = old_epsilon
    
        win_rate = np.mean(np.array(total_rewards)>0)
        average_reward = np.mean(total_rewards)
    
        print(f"Test Results over {num_episodes} episodes:")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Average Reward: {average_reward:.3f}")
        print(f"Standard Deviation: {np.std(total_rewards):.3f}")
        
        
        """
        =================
        Agent 0-shot
        =================
        Test: 100%|█████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 6711.57it/s]
        Test Results over 1000 episodes:
        Win Rate: 37.0%
        Average Reward: -0.211
        Standard Deviation: 0.952
        Training: 100%|███████████████████████████████████████████| 1000000/1000000 [02:03<00:00, 8126.45it/s]
        =======================
        Agent full-trained
        =======================
        Test: 100%|█████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 7755.09it/s]
        Test Results over 1000 episodes:
        Win Rate: 43.0%
        Average Reward: -0.039
        Standard Deviation: 0.947
        """
    ```

  - 一个好的 Blackjack Agent 应该表现为 :

    - 胜率: 42 - 45%
    - 平均奖励: -0.02 - 0.01
    - 标准差: 越小越好 (<0.5较为理想)

- 下一步:

  - 上面已经通过Q-Learning完成了Blackjack Agent的训练，接下来可以挑战复杂或不同类型的环境
    - `CartPole-v1`: 经典平衡问题, 适合理解连续控制
    - `MountainCar`: 目标是让小车爬上山, 需要有耐心的积累动量
    - `LunarLander-v2`: 控制飞船平稳着陆, 更有挑战性, 适合离散动作策略
  - 调整超参
    - 强化学习对超参非常敏感，可以尝试不同的学习率, 不同的探索策略
  - 尝试其他算法: SARSA, Expected SARSA, Monte Carlo, Double Q-Learning
  - 使用函数逼近方法，如果状态空间太大, Q-table不再适用, 可以用神经网络代替表格
  - 自定义环境: 设计自己的强化学习问题

- 
