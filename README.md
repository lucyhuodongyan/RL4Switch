# Reinforcement Learning for Switch Scheduling

This repository investigates the application of reinforcement learning (RL) techniques for optimizing scheduling decisions in the switch scheduling problem. It explores the use of Proximal Policy Optimization (PPO) and Atomic Action (an action decomposition technique) with PPO to handle environments with large action spaces. The project specifically focuses on improving scheduling policies within a input-queued switch network.

## Project Structure

The repository is organized into two main folders:

### 1. **`PPO/`**: Contains all code and experiment results related to the implementation of Proximal Policy Optimization (PPO).

- **`switchNetwork.py`**: Defines the structure of the switch network, including the system dynamics (including traffic arrival) and the queuing network.
- **`switchValue.py`**: Implements the value function using PyTorch, which is used to estimate the expected future rewards during training.
- **`switchPolicy.py`**: Implements the policy used for decision-making in the network. This includes the PyTorch model responsible for policy learning.
- **`switchUtils.py`**: Contains utility functions that support the network and policy operations.
- **`switchTrain.py`**: Contains the main training loop for policy optimization. It interacts with the environment, collects data, and updates the policy.
- **`test_code.py`**: The main script to update key parameters such as network size, policy settings, and hyperparameters. Running this script launches the program.
- **log-files Folder**: Contains log files from the PPO experiments, detailing performance metrics, convergence behavior, and other outputs.

### 2. **`AAPPO/`**: Contains all code and experiment results related to Atomic Action Proximal Policy Optimization (AAPPO).
- **`network_testNN.py`**: Defines the neural network architecture and testing functions for atomic actions within the switch network.
- **`value_testNN.py`**: Implements the value network testing functions for atomic actions, which evaluate the expected return of various states and actions.
- **`policy_testNN.py`**: Implements policy evaluation using neural networks for atomic actions in the switch network.
- **`utils.py`**: Contains utility functions that assist with training, data handling, and processing.
- **`train_testNN_parallel.py`**: Contains the main training loop for atomic action policy optimization. It interacts with the environment, collects data, and updates the policy.
- **`test_code.py`**: The main script to update key parameters such as network size, policy settings, and hyperparameters. Running this script launches the program.
- **`compiled_pruning.py`**: Implements a pruning mechanism to reduce the complexity of the neural network used in policy learning. It compiles results based on predefined criteria for pruning.
- **log-files Folder**: Contains log files from AAPPO experiments, tracking performance, pruning results, and policy evaluation metrics.


### Documentation
- **`INFORMS slides.pdf`**:  A slide deck that details the switch scheduling problem, reinforcement learning techniques (PPO and Atomic Action PPO), and neural network pruning strategies. This presentation was prepared for the INFORMS2022 annual conference and is included in this repository.
- **Technical Report**: A technical report further elaborating on the research behind the code in this repository will be uploaded shortly. Stay tuned!


## Usage

### Running PPO Experiments

To run experiments related to PPO, navigate to the `PPO/` folder, update the necessary parameters, and launch the program:

```bash
cd PPO
python test_code.py
```

### Running AAPPO Experiments

To run experiments for Atomic Action PPO, navigate to the `AAPPO/` folder, update the necessary parameters, and launch the program:

```bash
cd AAPPO
python test_code.py
```

### Pruning Neural Networks for Atomic Actions

To prune the policy networks of AAPPO, use the following script:

- **`compiled_pruning.py`**: Executes model pruning and compile the results.

### Key Parameters in `test_code.py`

Some key parameters you can configure in `test_code.py` include:

- **Network Size (`N`)**: Adjust the switch network size to simulate different configurations.
- **Policy Import (`policy_import`)**: Toggle whether to import pre-trained policies.
- **Number of Policy Iterations**: Control the number of iterations for the policy optimization.

## Dependencies

- Python 3.8+
- PyTorch
- `ray`
- `numpy`
- `scipy`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

We welcome contributions to this project! Please open an issue to discuss any changes or improvements.

## Acknowledgments

This project was inspired by ongoing research in queuing network control, reinforcement learning, and neural network optimization, with a particular focus on using atomic action decomposition to address operational challenges involving large action spaces.
