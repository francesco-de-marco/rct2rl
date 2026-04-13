# Reinforcement Learning su RollerCoaster Tycoon (OpenRCT2)

![Project Status](https://img.shields.io/badge/Status-Experimental-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Ray RLlib](https://img.shields.io/badge/Ray-RLlib-blueviolet)

This project presents an experimental implementation and analysis of a **Deep Reinforcement Learning** approach applied to the management simulation video game **OpenRCT2**. The agent uses the **PPO (Proximal Policy Optimisation)** algorithm with a multimodal neural network to manage amusement parks, optimising the layout and financial management to achieve specific visitor and rating targets.

## 🎯 Objectives

*   **Validation of the PPO algorithm** in complex and stochastic environments such as RCT.
*   **Grid-based spatial management**: The agent ‘sees’ the park via heatmaps (excitement, intensity, height) processed by an Encoder-Decoder CNN.
*   **Action Masking**: Integration of logical constraints to prevent invalid actions (e.g. building in water or off-map).
*   **Replication on Consumer Hardware**: Adaptation of the original hyperparameters for training on a single GPU (RTX 3050 Ti) with limited resources.

## 🧠 System Architecture

The system decouples the agent from the game environment by using **OpenRCT2** in headless mode and communicating via ZeroMQ sockets.

*   `pathrl.py`: **Entry Point**. Initialises Ray and starts the PPO training loop.
*   `gen_envs/rct.py`: **Gymnasium** wrapper that translates game states into tensors for the neural network.
*   `bridge.py`: Handles **ZeroMQ** communication with the OpenRCT2 C++ process.
*   `visionnet2d.py`: **CNN Encoder-Decoder** network that processes the 87x87 map of the park.
*   `rl_model.py`: Implements the policy network and **Action Masking**.

## 🚀 Installation

### Prerequisites
*   Python 3.10
*   OpenRCT2 installed and configured (including the original RCT2 asset files).
*   Linux (tested on Ubuntu).

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/francesco-de-marco/rct2rl.git
   cd rct2rl
   ```

2. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Check the paths in `paths.py`:
   Ensure that `RCT_EXECUTABLE` points to your OpenRCT2 binary.

> **Note on Replication**: For full details on the base codebase and the original replication procedure, please refer to the [README of the original research](https://github.com/campbelljc/rctrl) cited in the acknowledgements.


## 💻 Usage

Training can be started in different modes depending on the complexity of the scenario (see environment hierarchy in `rct.py`).

Basic command:
```bash
python pathrl.py <mode>
```

Where `<mode>` indicates the difficulty level:
*   `1`: **RCTEnv** (Free simulation, continuous training with no Game Over).
*   `2`: **MeetObjectiveRCTEnv** (Adds win/loss conditions based on guest targets).
*   `3`: **ResearchMeetObjectiveRCTEnv** (Adds the ride search mechanic).
*   `4`: **DaysResearchMeetObjectiveRCTEnv** (Realistic day-by-day simulation).

## 📊 Experimental Results

The training process was optimised compared to the original configuration by implementing three key changes to ensure it could run on consumer hardware:

1.  **Hyperparameter Tuning**: Reducing the *training batch size* (from 512 to 256) and setting the number of workers to 2 to operate within the VRAM limits of the GPU (RTX 3050 Ti).
2.  **Resource Stability**: Explicit constraints on Ray’s object store memory to prevent OOM (*Out Of Memory*) crashes and swap space saturation.
3.  **Hybrid Reward Function**: Modification of the reward function to combine the original dense *shaping* with sparse terminal signals (win/loss bonuses), accelerating convergence over shorter time horizons.
4.  **CBAM Attention**: Integration of the *Convolutional Block Attention Module* (CBAM) into the visual network to improve the extraction of critical spatial features.
5.  **Transfer Learning**: Implementation of *Fine-Tuning* pipelines that enabled the model to be adapted to new scenarios (e.g. *Crazy Castle*).

## 👤 Author

**Francesco De Marco**
Machine & Deep Learning Project - Academic Year 2025/2026

---
*Credit: Based on the original research and the code base `https://github.com/campbelljc/rctrl`.*
