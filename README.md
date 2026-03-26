# EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation

[![arXiv](https://img.shields.io/badge/arXiv-2511.03370-b31b1b.svg)](https://arxiv.org/abs/2511.03370)

![EQ-Negotiator Framework](flow_eq.png)

**EQ-Negotiator** is a framework that bridges the capability gap between Small Language Models (SLMs) and Large Language Models (LLMs) in emotionally charged credit negotiations. It integrates game theory (Win-Stay, Lose-Shift) with Hidden Markov Models to dynamically adapt emotional strategies in real-time, enabling 7B parameter models to outperform LLMs 10x their size.

## Project Structure

```
eq_negotiator/
├── main.py                          # Entry point
├── config/scenarios.json            # CRAD dataset (100 scenarios)
├── experiments/
│   └── run_eq_negotiator.py         # Experiment runner
├── llm/
│   ├── llm_wrapper.py               # LLM interface (GPT, Claude, DeepSeek, Llama)
│   └── negotiator.py                # Multi-agent negotiation (LangGraph)
├── models/
│   ├── constants.py                 # Emotions, matrices (Tables 4-5)
│   ├── hmm_model.py                 # HMM emotional strategy (Eq. 2-6)
│   ├── payoff_matrix.py             # WSLS game theory (Eq. 1)
│   └── eq_negotiator.py             # Main EQ engine (Eq. 4)
├── results/                         # Experiment outputs
└── utils/
    └── helpers.py                   # Utilities
```

## Quick Start

```bash
pip install langchain langchain-openai langgraph numpy scipy python-dotenv

# Set API key
echo "OPENAI_API_KEY=your_key" > .env

# Run with vanilla debtor
python main.py --model_creditor gpt-4o-mini --scenarios 5

# Run with adversarial debtor
python main.py --debtor_persona threatening --scenarios 5

# Run across all personas
python main.py --debtor_persona all --iterations 3 --scenarios 10
```

## Citation

```bibtex
@article{liu2025eq,
    title={EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation},
    author={Liu, Yuhan and Long, Yunbo and Brintrup, Alexandra},
    journal={arXiv preprint arXiv:2511.03370},
    year={2025}
}
```
