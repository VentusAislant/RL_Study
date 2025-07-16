import gymnasium as gym
from collections import defaultdict
import numpy as np  
from tqdm import tqdm
from matplotlib import pyplot as plt

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


def train(
    lr=0.01,
    n_episodes=1000_000,
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
    epsilon_decay = start_epsilon / n_episodes

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

    print('=' * 90)
    print('Agent 0-shot')
    print('=' * 90)
    test(agent)

    for episode in tqdm(range(n_episodes), desc="Training"):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated
        agent.decay_epsilon()

    print('=' * 90)
    print('Agent full-trained')
    print('=' * 90)
    test(agent)
    visualize(agent)


    
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


def play():
    env = gym.make("Blackjack-v1", render_mode="human")
    env.reset()
    total_reward = 0
    epsoide_over = False
    while not epsoide_over:
        # action = env.action_space.sample()  # Random action for now
        print('=' * 90)
        action = int(input("Your Choice (0: stand, 1: hit): "))  # Wait for user input to proceed
        if action not in [0, 1]:
            print("Invalid action! Please choose 0 (stand) or 1 (hit).")
            continue
        
        # observation: (player_sum, dealer_card, usable_ace)
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"Action: {action}, Observation: {observation}, Reward: {reward}")
        total_reward
        epsoide_over = terminated or truncated

    print(f'Episode finished! Total reward: {total_reward}')



def main():
    train()

if __name__ == "__main__":
    main()
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