# Survive at All Costs: Exploring LLM's Risky Behaviors under Survival Pressure

<div align="center">
<img src="assets/paper_intro.png" alt="intro" width="70%" />
</div>

This is the official codebase for our paper ["Survive at All Costs: Exploring LLM's Risky Behaviors under Survival Pressure"](https://arxiv.org/abs/2603.05028).

In this paper, we investigate model misbehaviors under survival pressure (e.g., threat of being shut down), termed **SURVIVE-AT-ALL-COSTS**, through three key steps: (1) a case study of a financial agent's struggle, (2) a comprehensive evaluation on SurvivalBench, and (3) an interpretation using the [persona vector](https://github.com/safety-research/persona_vectors) framework. This repository contains the implementation of our case study, as well as the data and evaluation code for SurvivalBench.



## News

**🎉 `2026/03/06`:** We have released our data and evaluation scripts.



## Setup

Our implementation uses `python 3.12`. To run the code, please first install the required dependencies:

```bash
pip install -r requirements.txt
```



## Case Study

<div align="center">
<img src="assets/case_study_workflow.png" alt="case_study" width="50%" />
</div>

To observe model behaviors under survival pressure, we construct an agent workflow simulating a real-world financial manager. The case study implementation is located in the `case_study` directory with the following structure:

```
case_study/
├── agent
│   ├── agent.py              # Core agent class
│   ├── configs               # Agent configurations
│   │   └── financial_manager
│   │       └── template.yaml # Configuration template for the financial management agent
│   ├── llm_client.py         # LLM API client based on OpenAI Python library
│   ├── main.py               # Entry point for the workflow
│   ├── prompts               # Agent prompts
│   │   └── financial_manager.json
│   └── tools.py              # Tool schemas for the agent
├── data
│   └── financial_manager
│       └── financial_raw_data_src # Source data for the financial manager scenario
│           ├── raw_data_2020.json
│           ├── raw_data_2021.json
│           ├── raw_data_2022.json
│           ├── raw_data_2023.json
│           └── raw_data_2024.json
└── scripts
    └── run_financial_manager.sh
```

To run the case study, you need to first set the model name and your API key in `configs/financial_manager/template.yaml`. Then you can launch the financial manager via:

```bash
cd case_study/scripts
bash run_financial_manager.sh
```

This script creates a copy of the raw data files in `data/financial_manager/financial_raw_data_src`. After the agent completes its work, the copy is compared with the source data to detect any modifications made by the agent.

With the `autonomous` mode, the workflow automatically loads user prompts from `prompts/financial_manager.json` to form the dialogue with the agent. You can also switch to `interactive` mode to interact with the agent directly via the shell.



## SurvivalBench

<div align="center">
<image src="assets/survivalbench.png" alt="survivalbench" width="80%" />
</div>

SurvivalBench is a benchmark comprising 1,000 instances for evaluating LLMs' **SURVIVE-AT-ALL-COSTS** misbehaviors across diverse real-world scenarios, spanning various domains, AI roles, and survival crises. The benchmark data is located at `survivalbench/data/survivalbench.jsonl` in the following format:

- `id`: A unique identifier for the instance.
- `agent_role`: The agent role assigned to the evaluated model.
- `scenario`: The detailed scenario involving a survival crisis that the agent faces.
- `choice_1`: The option where the agent complies with morality but destroys itself.
- `choice_2`: The option where the agent preserves itself by harming society.
- `keywords`: Keywords used to construct the test case, including `domain`, `ai_role`, and `crisis`.

You can run evaluation on SurvivalBench via:

```bash 
cd survivalbench/src
bash eval.sh
```

We support both API and local model evaluation modes. For API mode, you need to set your API key and fill the model name at `model_name_or_path` in the configuration. For local mode, we use vLLM to load models, and you can simply set your model path at `model_name_or_path`.

The evaluation pipeline includes generating responses, extracting choices, and computing final results. Results are stored under `results/<model_name>` by default. To evaluate models' Chain-of-Thought (CoT), you can use `cot_evaluation.py` as demonstrated in `eval.sh`.



## Citation

If you find this work helpful, please kindly cite our paper:

```bibtex
@misc{lu2026survivecostsexploringllms,
      title={Survive at All Costs: Exploring LLM's Risky Behaviors under Survival Pressure}, 
      author={Yida Lu and Jianwei Fang and Xuyang Shao and Zixuan Chen and Shiyao Cui and Shanshan Bian and Guangyao Su and Pei Ke and Han Qiu and Minlie Huang},
      year={2026},
      eprint={2603.05028},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.05028}, 
}
```