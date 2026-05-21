# REINFORCEMENT LEARNING
# DS -> ML -> RL -> Q-Learning & Policy Gradient

## Where It Fits
```
Machine Learning
│
├── Supervised Learning
│   ├── Regression
│   │   ├── Linear Regression
│   │   ├── Polynomial Regression
│   │   └── Ridge / Lasso
│   │
│   └── Classification
│       ├── Logistic Regression
│       ├── Decision Trees
│       ├── Random Forest
│       ├── SVM
│       └── k-NN
│
├── Unsupervised Learning
│   ├── Clustering
│   │   ├── k-means
│   │   ├── Hierarchical Clustering
│   │   └── DBSCAN
│   │
│   ├── Dimensionality Reduction
│   │   ├── PCA
│   │   └── t-SNE
│   │
│   └── Anomaly Detection
│       └── Isolation Forest
│
├── Reinforcement Learning       <--- YOU ARE HERE
│   ├── Q-Learning
│   └── Policy Gradient
│
└── Ensemble Methods
    ├── Bagging (Random Forest)
    └── Boosting (XGBoost, LightGBM)
```

---

> Unlike Supervised Learning (learn from labeled examples) or Unsupervised Learning (find hidden structure), Reinforcement Learning learns by **trial and error** — an agent takes actions, receives rewards or penalties, and figures out the best strategy over time.

---

## The 3 Core Concepts

### 1. The Setup
> Before understanding the algorithms, you need the four building blocks every RL problem shares.

- **Agent** — The learner or decision-maker. The thing that takes actions (a robot, a game character, a trading bot).
- **Environment** — Everything the agent interacts with (a game board, a factory floor, a financial market).
- **State (S)** — A snapshot of the current situation the agent observes (current board position, sensor readings, stock prices).
- **Action (A)** — What the agent can do from a given state (move left, increase speed, buy/sell).

---

### 2. Rewards
> The signal the agent uses to learn what's good and what's bad.

- **Reward (R)** — A number the environment gives back after each action. Positive = good, negative = bad.
- **Cumulative Reward** — The agent's goal isn't to maximize the next reward — it's to maximize the *total* reward over time.
- **Discount Factor (γ)** — A number between 0 and 1 that controls how much the agent values future rewards vs. immediate ones. γ close to 1 = patient (plans ahead). γ close to 0 = shortsighted (grabs rewards now).

```
Total Reward = R₁ + γR₂ + γ²R₃ + γ³R₄ + ...
```

---

### 3. Policy
> The agent's strategy — a mapping from states to actions.

- **Policy (π)** — The rule the agent follows to decide what to do in any given state.
- **Optimal Policy (π*)** — The policy that maximizes cumulative reward. Finding this is the entire goal of RL.

| | Supervised Learning | Reinforcement Learning |
|---|---|---|
| **Learns from** | Labeled examples | Trial and error |
| **Feedback** | Correct answer given upfront | Reward after each action |
| **Goal** | Predict Y from X | Maximize cumulative reward |
| **When labels exist** | Yes | No |

---

## RL Algorithm Types

### Q-Learning
> Learns the value of taking a specific action in a specific state, building a lookup table of what's worth doing where.

**When to use:** The environment has a manageable number of states and actions that can be stored in a table (games, grid worlds, simple robots).

**The core idea:** Q-Learning builds a **Q-table** — a grid where every row is a state and every column is an action. Each cell holds a Q-value: *how good is it to take this action from this state?*

```
Q(State, Action) = Expected cumulative reward from here
```

**Example:** A robot learning to navigate a 4-room grid to reach a goal (★) while avoiding a trap (✗).

```
┌───┬───┬───┐
│ S │   │   │
├───┼───┼───┤
│   │ ✗ │   │
├───┼───┼───┤
│   │   │ ★ │
└───┴───┴───┘
```

The robot starts at S and can move Up, Down, Left, Right. After many episodes of trial and error, it builds this Q-table:

| State      | Up    | Down  | Left  | Right |
|------------|-------|-------|-------|-------|
| S (0,0)    | -0.1  | +0.3  | -0.1  | +0.5  |
| (0,1)      | -0.1  | -0.8  | +0.2  | +0.6  |
| (1,0)      | +0.2  | +0.4  | -0.1  | -0.1  |
| ...        | ...   | ...   | ...   | ...   |

The robot hits the trap (✗), gets a big negative reward (-1). It reaches the goal (★), gets a big positive reward (+1). Over thousands of attempts, the Q-values converge — the highest value in each row tells the robot the best move from that state.

**The Q-Learning update rule:**

```
Q(S, A) ← Q(S, A) + α × [R + γ × max Q(S', A') − Q(S, A)]
                           ↑ reward    ↑ best future value
```

- **α (learning rate)** — How fast to update. Too high = unstable. Too low = slow to learn.
- **γ (discount factor)** — How much to value future rewards.
- **max Q(S', A')** — The best Q-value available in the next state.

> **Limitation:** If the state space is huge (like a video game with millions of possible screens), a Q-table becomes impossible to store. This is where Deep Q-Networks (DQN) come in — replacing the table with a neural network.

---

### Policy Gradient
> Instead of learning the value of actions, directly learns the policy itself — the probability of taking each action in each state.

**When to use:** The action space is continuous (steering angles, joint torques, throttle levels) or too large for a Q-table. Used heavily in robotics, game-playing AI, and language model fine-tuning.

**The core idea:** The policy is a function (usually a neural network) that takes a state and outputs a probability for each action. Training nudges the policy to make high-reward actions more likely and low-reward actions less likely.

```
π(A | S) = probability of taking action A in state S
```

**Example:** Training a self-driving car to stay in its lane.

| Timestep | State (drift from center) | Action (steering) | Reward |
|----------|--------------------------|-------------------|--------|
| 1        | 0.1m right               | Steer left 2°     | +1     |
| 2        | 0.0m (centered)          | Steer straight    | +1     |
| 3        | 0.3m right               | Steer left 8°     | +1     |
| 4        | 0.5m right               | Steer right 3°    | −1     |

After timestep 4, the policy gets penalized. The gradient update makes "steer right when drifting right" less probable in the future. After thousands of miles of simulation, the policy learns smooth, safe steering.

**The Policy Gradient update rule:**

```
θ ← θ + α × ∇θ log π(A|S) × R
         ↑ increase probability    ↑ scale by reward
```

If the reward R is positive → push the policy to make that action more likely.
If R is negative → push it to make that action less likely.

> **Key difference from Q-Learning:** Q-Learning asks *"what's the value of this action?"* Policy Gradient asks *"how should I adjust my probabilities?"* Policy Gradient handles continuous actions naturally; Q-Learning struggles with them.

---

### Q-Learning vs. Policy Gradient — Key Differences

| | Q-Learning | Policy Gradient |
|---|---|---|
| **Learns** | Value of each (state, action) pair | Probability of each action directly |
| **Output** | Q-table or Q-network | Policy function (probabilities) |
| **Action space** | Discrete (finite actions) | Discrete or continuous |
| **Best for** | Games, grid problems, simple robots | Robotics, driving, complex control |
| **Sample efficiency** | Higher | Lower (needs more experience) |
| **Example** | Chess, Pac-Man, grid navigation | Self-driving cars, robot arms, ChatGPT RLHF |

---

## The 3 Stages of an RL Training Loop

### 1. Exploration vs. Exploitation
> The agent must balance trying new things (exploration) vs. using what it already knows (exploitation).

- **Exploration** — Taking random or novel actions to discover new rewards.
- **Exploitation** — Taking the action currently believed to be best.
- **ε-greedy** — The most common strategy: with probability ε, explore randomly; otherwise, exploit the best known action. ε typically starts high (lots of exploration) and decays over time.

```
Early training:  ε = 0.9  → explore 90% of the time
Late training:   ε = 0.05 → exploit 95% of the time
```

---

### 2. Training
> The agent interacts with the environment, collects experience, and updates its Q-table or policy.

- **Episode** — One complete run from start to finish (one game, one maze attempt, one driving session).
- **Step** — A single action taken within an episode.
- **Experience Replay** — Storing past (State, Action, Reward, Next State) tuples and replaying them randomly during training to break correlation and improve stability.

| Episode | Steps Taken | Total Reward | Outcome  |
|---------|-------------|--------------|----------|
| 1       | 47          | −12          | Hit trap |
| 10      | 31          | −4           | Hit trap |
| 50      | 18          | +8           | Reached goal |
| 200     | 9           | +14          | Reached goal (optimal path) |

---

### 3. Evaluation
> Metrics used after training to measure how well the agent learned.

- **Cumulative Reward** — Total reward collected per episode. Should trend upward over training.
- **Convergence** — When the cumulative reward stabilizes and stops improving significantly — the agent has found a good policy.
- **Average Steps to Goal** — For navigation tasks, fewer steps = better policy.
- **Win Rate / Success Rate** — For games or binary-outcome tasks (did the robot reach the goal?).

---

## Full Worked Example (Q-Learning)

**Setup:** A robot in a 3-state hallway must reach the exit (State C) from the start (State A). It can move Forward or Backward.

```
[State A] ←→ [State B] ←→ [State C ★]
```

**Rewards:**
- Reaching State C: +10
- Any other move: −1 (time penalty)

**Initial Q-table (all zeros):**

| State | Forward | Backward |
|-------|---------|----------|
| A     | 0       | 0        |
| B     | 0       | 0        |
| C     | 0       | 0        |

**Training parameters:** α = 0.5, γ = 0.9

---

**Episode 1, Step 1:** Agent is in State A. Explores randomly → moves Forward to State B. Reward = −1.

```
Q(A, Forward) ← 0 + 0.5 × [−1 + 0.9 × max(0, 0) − 0]
              ← 0 + 0.5 × [−1]
              ← −0.5
```

**Episode 1, Step 2:** Agent is in State B. Explores → moves Forward to State C. Reward = +10.

```
Q(B, Forward) ← 0 + 0.5 × [+10 + 0.9 × max(0, 0) − 0]
              ← 0 + 0.5 × [10]
              ← +5.0
```

**After many episodes, Q-table converges:**

| State | Forward | Backward |
|-------|---------|----------|
| A     | +7.3    | −1.2     |
| B     | +9.1    | −0.8     |
| C     | 0       | 0        |

The highest value in each row now points toward the exit. The optimal policy: A → Forward → B → Forward → C.

---

> **Key takeaway:** RL doesn't need labeled data — it learns purely from the consequences of its own actions. The Q-table (or policy network) is the model; rewards are the teacher.
