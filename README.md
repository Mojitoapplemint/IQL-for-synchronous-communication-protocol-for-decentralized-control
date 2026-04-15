
# Q-Learning for Synthesizing Synchronous Communication Protocols for Decentralized Supervisory  Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📌 Abstracts
The optimality of synchronous communication protocols in decentralized discrete-event control has traditionally been analyzed through syntactic measures, such as the number of communication events or logical minimality. However, these metrics provide no insight into how these protocols perform during actual system operation. As such, analyzing communication empirically provides more practical metrics, such as communication frequency, decision quality, and operational efficiency. We use Multi-Agent Reinforcement Learning to analyze empirically such communication protocols in decentralized decision-making architectures.

## 🎓 Citation
This work is currently accepted by the Workshop on Discrete Event Systems(WODES) 2026. The formal citation will be updated soon.

## 🛠️ Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Mojitoapplemint/IQL-for-synchronous-communication-protocol-for-decentralized-control.git
    cd IQL-for-synchronous-communication-protocol-for-decentralized-control
    ```

2.  **Create a virtual environment:**

    ```bash
    # Using conda
    conda env create -f environment.yml
    conda activate research-env
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 📊 Usage

Explain how to run the main experiments or generate the results presented in your thesis.

1. For each example folder, go to data_collector.py. This file will conduct 1000 independent training sessions to collect the data
2. Once data collection is done, go to stats.py to analyze the data
3. For some examples, there is an additional protocol_analysis.py file, which produces the actual communication protocol induced from the algorithm
4. If you want to run the algorithm only once, then run q_train.py


## 🤝 Acknowledgments
We acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), and the Atlantic Association for Research in the Mathematical Sciences (AARMS).
