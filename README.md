# ⚖️ MAD: Multi-Agent Debate

🔥 This work aims to explore the debating capability of LLMs by proposing the **MAD** framework, which stands for **M**ulti-**A**gent **D**ebate.

> "Truth emerges from the clash of adverse ideas."  
> "真理越辩越明。"

## Brief Introduction

The cognitive behavior of large language models (LLMs) has garnered significant attention in recent times. For example, **self-reflection**, a concept that usually refers to the process of introspection and examination of a person's own thoughts, has also been demonstrated effective with LLMs in solving challenging NLP tasks.

However, we point out that self-reflection can easily fall into the **degeneration of thoughts (DoT)** issue in the following scenarios:

- **Bias and Distorted Perception**: Self-perception can be influenced by biases, preconceived notions, and distorted thinking patterns. If an individual's self-reflection is clouded by such biases or distorted thinking, it can lead to 😔 *inaccurate conclusions and hinder personal growth*.
- **Rigidity and Resistance to Change**: Self-reflection often involves challenging one's beliefs, assumptions, and behaviors. If an individual is resistant to change or holds rigid beliefs, they may 😔 *struggle to engage in meaningful self-reflection* that leads to personal growth.
- **Limited External Feedback**: Self-reflection is primarily an internal process, but external feedback can provide valuable perspectives and insights. Without seeking or considering external feedback, an individual may 😔 *miss important blind spots or alternative viewpoints that can enrich their self-reflection*.

<div align="center">
    <img width="45%" alt="MAD" src="imgs/image.png" />
    <p><em>Figure 1: Comparison between debate and reflection.</em></p>
</div>

In this project, we have embarked on a journey to explore the potential of a debating interaction framework among LLMs. 
With **MAD**, the nature of agents being in the state of 'tit for tat' determines that:

1. The distorted thinking of one agent can be corrected by the other one 😃;
2. The resistance to change of one agent will be complemented by the other one 😄;
3. Either agent can provide external feedback for each other 😆.

Obviously, **MAD** is less likely to have the **DoT** issue and can exploit more potential of LLMs. Experiments show that MAD brings significant and consistent improvements on Counterintuitive QA and Commonsense-MT tasks.

## Framework

<div align="center">
    <img width="90%" alt="MAD" src="imgs/framework.png" />
    <p><em>Figure 2: Framework of Multi-Agent Debate. The devil (<img src="imgs/devil.png" width="25" />) is the affirmative side while the angel (<img src="imgs/angel.png" width="25" />) is the negative side. The goal is for the angel to correct the devil's mistakes.</em></p>
</div>

---

## 🔧 Reproducing the Experiments

### 📦 Dependencies

```sh
pip install -r requirements.txt
```

The main dependencies include:
- OpenAI API
- tqdm
- langcodes
- datasets (for CommonsenseQA experiments)
- numpy
- pandas (for results analysis)

### 📥 Data Download Instructions

#### Counterintuitive QA Dataset
```sh
# The data is already included in the repository
# Located at: data/CounterintuitiveQA/CIAR.json
```

#### CommonMT Dataset
```sh
# The example data is already included in the repository
# Located at: data/CommonMT/input.example.txt

# For full dataset (if needed):
wget https://github.com/Skytliang/Multi-Agents-Debate/raw/main/data/CommonMT/input.txt -O data/CommonMT/input.txt
```

#### CommonsenseQA Dataset
```sh
pip install datasets  # Install datasets library if not already installed
```

```python
from datasets import load_dataset
import json
import os

# Create directory if it doesn't exist
os.makedirs('data/CommonsenseQA', exist_ok=True)

# Load the dataset
dataset = load_dataset('commonsense_qa')

# Convert to required format
train_data = dataset['train']
formatted_data = [
    {
        'id': item['id'],
        'question': item['question'],
        'choices': [
            {'label': choice_label, 'text': choice_text}
            for choice_label, choice_text in zip(item['choices']['label'], item['choices']['text'])
        ],
        'answerKey': item['answerKey']
    }
    for item in train_data
]

# Save to JSON
with open('data/CommonsenseQA/csqa.json', 'w') as f:
    json.dump(formatted_data, f, indent=2)
```

---

## 🚀 Running the Experiments

### 1️⃣ Set up OpenAI API Key
Create a `.env` file in the root directory with:
```sh
OPENAI_API_KEY=your-openai-api-key-here
```
Or set it in the environment:
```sh
export OPENAI_API_KEY=your-openai-api-key-here
```

### 2️⃣ Counterintuitive QA Experiments
```sh
python code/debate4math.py -i data/CounterintuitiveQA/CIAR.json -o results/math_debates -k $OPENAI_API_KEY -m gpt-3.5-turbo
```

### 3️⃣ CommonsenseQA Experiments
```sh
python code/debate4csqa.py -i data/CommonsenseQA/csqa.json -o results/csqa_debates -k $OPENAI_API_KEY -m gpt-3.5-turbo -l 20
```

### 4️⃣ Machine Translation Experiments
```sh
python code/debate4tran.py -i data/CommonMT/input.example.txt -o data/CommonMT/output -lp zh-en -k $OPENAI_API_KEY -m gpt-3.5-turbo
```

---

## 📊 Results

### 🔹 Counterintuitive QA Results
<div align="center">
    <img width="35%" alt="CounterintuitiveQA" src="imgs/CounterintuitiveQA.png" />
    <p><em>Table 1: Reasoning accuracy on Counterintuitive QA.</em></p>
</div>

### 🔹 Commonsense Machine Translation Results
<div align="center">
    <img width="50%" alt="CommonMT" src="imgs/CommonMT.png" />
    <p><em>Table 2: Translation performance on Common MT.</em></p>
</div>

### 🔹 CommonsenseQA Results
<div align="center">
    <img width="50%" alt="CommonsenseQA" src="imgs/CommonsenseQA.png" />
    <p><em>Table 3: Performance on CommonsenseQA dataset.</em></p>
</div>

---

## 📜 Citation

```bibtex
@article{liang2023encouraging,
  title={Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate},
  author={Liang, Tian and He, Zhiwei and Jiao, Wenxiang and Wang, Xing and Wang, Yan and Wang, Rui and Yang, Yujiu and Tu, Zhaopeng and Shi, Shuming},
  journal={arXiv preprint arXiv:2305.19118},
  year={2023}
}
```

---
