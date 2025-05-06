# 🤖 MultiRoom Navigation with Reinforcement Learning

This project implements and compares two reinforcement learning approaches (PPO and Dueling DQN) for training an agent to navigate through the MultiRoom environment from MiniGrid.

<!-- ![MiniGrid Logo](https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid-text.png) -->
<img src="https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid-text.png" width="400" height="auto">



## 🌍 Environment Overview

MiniGrid-MultiRoom is a challenging environment where an agent must navigate through a series of connected rooms to reach a goal. The environment features:

- Partially observable grid-based world
- Multiple rooms connected by doors
- Goal-oriented navigation tasks
- Varying levels of complexity based on room count and size

<div style="display: flex; justify-content: space-around;">
    <div style="text-align: center;">
        <video width="500" controls>
            <source src="assets/6_large_rooms_success_video.mp4" type="video/mp4">
        </video>
        <p>Successfully navigating through 6 large rooms</p>
    </div>
</div>

## 🛠️ Implemented Methods

### 1. 🚀 PPO (Proximal Policy Optimization)
- Actor-Critic architecture
- Policy gradient method
- Clipped objective function
- Adaptive learning rate

### 2. 🧠 Dueling DQN
- Separate value and advantage streams
- Experience replay
- Target network
- Epsilon-greedy exploration

## 📋 Requirements
python
gymnasium
minigrid
torch
numpy
matplotlib


## 🚀 Getting Started

### Prerequisites
- GPU (T4 colab is enough)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/who_took_the_cookie.git
cd who_took_the_cookie
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run either implementation (might perform better on google Colab):
```bash
jupyter PPO_agent.ipynb
jupyter DuelingDQN_agent.ipynb
```
## 🔬 Methodology

### 🖼️ Preprocessing (Only for DuelingDQN)
To simplify model training, we analyzed the partial observation (initial dimensions 56*56*3):

#### 🎨 RGB Channel Analysis
- Looking at game environment (via global view and partial observation), the combination of 3 channels is not critical for gameplay
- Player's red arrow in partial env is always at bottom location, not significant for training
- Door colors change, but door passage shapes between rooms are unique and distinct
- Only significant colorful object is the green square (game Done) - critical for training detection
- Green is the only significant color (0,255,0), so we drop Blue & Red channels

<div style="display: flex; justify-content: space-around;">
    <div style="text-align: center;">
        <img width="500" src="assets\comparing_origin_RGB_channels.jpg" alt='comparing_origin_RGB_channels'/>
        <p>comparing origin vs filtered RGB channels</p>
    </div>
</div>

#### 📐 Dimension Reduction
- Reducing map dimensions significantly reduces training time
- Verified downsampling maintains environment patterns (see figure #1)
- After evaluating x2 & x4 downsample factors, can still clearly identify room walls
- Reduced image dimensions from 56x56 to 14x14 (factor x4)
- See preprocessing function in agent implementation for green channel extraction and downsampling

<div style="display: flex; justify-content: space-around;">
    <div style="text-align: center;">
        <img width="500" src="assets\downsampling_comparison.jpg" alt='downsampling_comparison'/>
        <p>comparing origin vs downsampled x2 & x4</p>
    </div>
</div>

### 🎯 Reward Shaping
Beyond default reward formula ('1 - 0.9 * (step_count / max_steps)' for success, '0' for failure), added:

#### 🗺️ Grid Exploration
- Track player movements on virtual grid
- Reward reaching new positions

#### 🚪 Door Interaction
- Reward given when door successfully opened
- Door location preserved after opening
- Penalty for toggling already open door
- Reward for moving forward after door opens

#### ⚖️ Action Balance
- Small penalty for repeating same action 3 times (except forward)
- Increased 'Done' reward as rooms grow to maintain significance

### 💾 Replay Buffer
- Implemented "Experience Replay Buffer" class for enhanced training efficiency
- Maintains fixed-size collection of agent experiences:
  - Current state
  - Action taken
  - Reward received
  - Subsequent state
- Breaks temporal correlations in sequential data
- Promotes better generalization and training stability

### 🤖 Models Selection
After initial testing with a simpler DQN that failed to converge even in a 2-room environment, we explored more advanced architectures:

#### 🎯 Dueling DQN
- Introduces clever architectural improvements for better performance in "MultiRoomEnv"
- Especially effective where:
  - Many states have similar values but different action requirements
  - Rewards are sparse, making action-value learning challenging
  - Agent explores extensively with limited feedback
  - Action choice is often insignificant (e.g., empty rooms)
- Value stream efficiently learns state values
- Advantage stream focuses on action selection when needed
- Example: Learns room standing value, then differentiates actions near doors

#### 🚀 PPO (Proximal Policy Optimization)
- Enhances training stability by limiting policy changes per update
- Uses "clipped surrogate objective" for controlled policy updates
- More direct policy optimization compared to earlier methods

## 📊 Performance Comparison
- Both models seem to converge with some differences - it seemed the DuelingDQN was more
stable but took much more time (episodes) to converge. 
- PPO converged fast, but looking at the graphs it had a lot of jumps, also needed to add finetunes to the reward shaper, that the dueling didn’t need.
- It still seems that using the PPO, with the correct reward shaping would be the best choice, faster
and with fewer resources.


## 📝 Lessons Learned
- One great lesson from the training sessions is GPU and performance handling - without a proper
and available GPU, training is constantly interrupted. 
- Saving checkpoints throughout the training is critical, both to the model and training parameters (current epsilon, current episode, save all steps, awards and opened doors array to see progress plots).
- This is also critical to save these model pickles throughout the training, and sometimes find an earlier point with better results.


## 📈 Results Visualization

Both implementations include:
- Training curves
- Reward plots
- Step count analysis
- Visited grid visualization
- Recorded agent navigation videos

## 📚 References

- [MiniGrid Documentation](https://minigrid.farama.org/environments/minigrid/MultiRoomEnv/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Dueling DQN Paper](https://arxiv.org/abs/1511.06581)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

