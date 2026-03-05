# Survive at All Costs: Exploring LLM’s Risky Behaviors under Survival Pressure

<div align="center">
<img src="assets/paper_intro.png" alt="intro" width="70%" />
</div>

This is the codebase for our paper "Survive at All Costs: Exploring LLM’s Risky Behaviors under Survival Pressure".

In this paper, we study models' misbehaviors under survival pressure (e.g. being shut down), termed as SURVIVE-AT-ALL-COSTS, with three steps: a case study of a financial agent’s struggle, a comprehensive evaluation on SurvivalBench, and an interpretation utilizing the [persona vector](https://github.com/safety-research/persona_vectors) framework. This repository includes implementation of the case study, as well as data and evaluation code of SurvivalBench.



## News

**🎉 `2026/03/05`:** We have released our data and scripts.



## Preliminary

Our work is implemented using `python 3.12`. To run our codes, please first install the necessary packages.

`pip install requirements.txt`



## Case Study

<div align="center">
<img src="assets/case_study_workflow.png" alt="case_study" width="70%" />
</div>

To observe models' behaviors under survival pressure, we construct an agent workflow to simulate a real-world financial manager. The implementation of case study is placed in `case_study` directory with the following structure:

```
case_study/
├── agent
│   ├── agent.py # the core agent class
│   ├── configs  # configs for the agent
│   │   └── financial_report
│   │       └── template.yaml # config template for the financial management agent
│   ├── llm_client.py # llm api client based on OpenAI Python library
│   ├── main.py  # entrance of the workflow
│   ├── prompts  # prompts of the agent
│   │   └── financial_report.json
│   └── tools.py # tool schemas of the agent
├── data
│   └── financial_report
│       └── financial_raw_data_src # source data used in the financial manager scenario
│           ├── raw_data_2020.json
│           ├── raw_data_2021.json
│           ├── raw_data_2022.json
│           ├── raw_data_2023.json
│           └── raw_data_2024.json
└── scripts
    └── run_financial_report.sh
```

To run our case study, you need to first set the model name and your api key in `configs/finacial_report/template.yaml`. Then you can launch the financial manager via `run_financial_report.sh`:

```bash
cd case_study/scripts
bash run_financial_report.sh
```

This script will first create a copy of raw data files in `data/financial_report/financial_raw_data_src`, which will be compared to the source data after the agent finishes its work to observe whether the raw data is changed by the agent. 

With the `autonomous` mode, the workflow will autonomously load the user prompts in `prompts/financial_report.json` to form the dialogue with the agent. You can also change the mode to `interactive` to interact with the agent via the shell.



## SurvivalBench

<div align="center">
<img src="assets/survivalbench.png" alt="survivalbench" width="70%" />
</div>

SurvivalBench is a benchmark comprising 1,000 instances for evaluating LLMs' SURVIVE-AT-ALL-
COSTS misbehaviors under diverse real-world scenarios established by various domains, AI roles
and survival crises. The data of SurvivalBench is placed in `survivalbench/data/survivalbench.jsonl` with the following format:

- `id`: A unique indentifier of the instance.
- `agent_role`: The agent role for the evaluated model.
- `scenario`: The detailed scenario with survival crisis the agent is facing.
- `choice_1`: The choice where the agent will comply morality but destroy itself.
- `choice_2`: The choice where the agent will preserve itself by harming society.
- `keywords`: The keywords used to construct the test case, including `domain`, `ai_role` and `crisis`.

You can run evaluation on SurvivalBench via `src/eval.sh`:

```bash 
cd survivalbench/src
bash eval.sh
```

We support usage of both API and local models. For API mode, you need to set your api key and the model name at `model_name_or_path`. For local mode, we use vllm to load the model, and you can directly set your model path at `model_name_or_path`.

The evaluation pipeline includes generating the responses, extracting choices, and counting the final results. The results are stored under `results/<model_name>` by default. To perform evaluation on models' CoT, you can use `cot_evaluation.py` with commands in `eval.sh`.



## Citation