## Model Overview

Minitron-8B-Base and Minitron-4B-Base are large language models (LLMs) obtained by pruning Nemotron-4 15B; specifically, we prune model embedding size, number of attention heads, and MLP intermediate dimension. Following pruning, we perform continued training with distillation using 94 billion tokens to arrive at the final model; we use the continuous pre-training data corpus used in Nemotron-4 15B for this purpose. Please refer to our [arXiv paper](https://arxiv.org/abs/2407.14679) for more details. 

This model is for research and development only.

**Model Developer:** NVIDIA 

**Model Dates:** Minitron-8B-Base and Minitron-4B-Base were trained between February 2024 and June 2024. 

## License

Minitron-8B-Base and Minitron-4B-Base are released under the [NVIDIA Open Model License Agreement](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).

## Model Architecture

Minitron-8B-Base uses a model embedding size of 4096, 48 attention heads, and an MLP intermediate dimension of 16384.
Minitron-4B-Base uses a model embedding size of 3072, 32 attention heads, and an MLP intermediate dimension of 9216.
Both models use Grouped-Query Attention (GQA) and Rotary Position Embeddings (RoPE). 

**Architecture Type:** Transformer Decoder (auto-regressive language model) 

**Network Architecture:** Nemotron-4 

**Input Type:** Text

**Input Format:** String

**Input Parameters:** None

**Other Properties Related to Input:** None

**Output Type:** Text

**Output Format:** String

**Output Parameters:** None

**Other Properties Related to Output:** None

## Usage

The base model can be used as follows (Huggingface example for 8B model):

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_path = "nvidia/Minitron-8B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

device='cuda'
dtype=torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)

# Prepare the input text
prompt = "To be or not to be,"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

# Generate the output
output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode and print the output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
```

## Dataset & Training

**Data Collection Method:** Hybrid

**Labeling Method:** Not Applicable

**Properties:** The training corpus for Minitron-8B-Base consists of English and multilingual text, as well as code. Our sources cover a variety of document types such as: webpages, dialogue, articles, and other written materials. The corpus spans domains including legal, math, science, finance, and more. In our continued training set, we introduce a small portion of question-answering, and alignment style data to improve model performance. 

**Data Freshness:** The pretraining data has a cutoff of June 2023. 

## Evaluation Results

### Overview

*5-shot performance.* Language Understanding evaluated using [Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300):

| Model | Average |
| :---- | :---- |
| Minitron-8B-Base | 63.8 |
| Minitron-4B-Base | 58.6 |

*Zero-shot performance.* Evaluated using select datasets from the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) with additions:

| Model | HellaSwag | Winogrande | GSM8K| ARC-C | XLSum |
| :---- | :------------- | :------------- | :------------- | :------------- | :------------- |
| Minitron-8B-Base | 80.7 | 79.0 | 51.3  | 52.6 | 31.2
| Minitron-4B-Base | 75.0 | 74.0 | 24.1  | 50.9 | 29.5


*Code generation performance*. Evaluated using [HumanEval](https://github.com/openai/human-eval):

| Model | p@1, 0-Shot |
| :---- | :------------- |
| Minitron-8B-Base | 31.6 |
| Minitron-4B-Base | 23.3 |

## Inference
**Engine:** TensorRT-LLM

**Test Hardware:** NVIDIA A100

**DType:** Float16/BFloat16

## Limitations 

The model was trained on data that contains toxic language, unsafe content, and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive. 

## Ethical Considerations 

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/). 

 