# Minitron

<p align="center">
<img src="https://www.sauravm.com/assets/img/minitron.png"  width="256">
</p>
<p align="center">
        🤗 <a href="https://huggingface.co/collections/nvidia/minitron-669ac727dc9c86e6ab7f0f3e">Hugging Face Models</a>&nbsp&nbsp | &nbsp&nbsp 📄 <a href="">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📜 <a href="">Blog</a> &nbsp | &nbsp 💬 <a href="">Demo</a>
</p>


## Introduction

Minitron is a family of small language models (SLMs) obtained by pruning NVIDIA's [Nemotron-4 15B](https://arxiv.org/abs/2402.16819) model. We prune model embedding size, attention heads, and MLP intermediate dimension, following which, we perform continued training with distillation to arrive at the final models.

Deriving the Minitron 8B and 4B models from the base 15B model using our approach requires up to **40x fewer training tokens** per model compared to training from scratch; this results in **compute cost savings of 1.8x** for training the full model family (15B, 8B, and 4B). Minitron models exhibit up to a 16% improvement in MMLU scores compared to training from scratch, perform comparably to other community models such as Mistral 7B, Gemma 7B and Llama-3 8B, and outperform state-of-the-art compression techniques from the literature. Please refer to our [arXiv paper]() for more details.

Minitron models are for research and development only.

## Minitron Model Performance

<p align="center">
  <img src="images/minitron.png" alt="Minitron accuracy" width="512"/>
  <p align="center">Minitron accuracy (MMLU) vs. other baseline models. Compression results in significant reduction of training costs for additional models(40x) while producing better results. Please refer to our <a href="">paper</a> for the full set of results.</p>
</p>

## Model Card
Please see [MODEL_CARD.md](MODEL_CARD.md).

## Quickstart

### Hugging Face

The [PR](https://github.com/huggingface/transformers/pull/31699) to support our models in Hugging Face in under review and expected to be merged soon. This [branch](https://github.com/suiyoubi/transformers/tree/aot/nemotron-support) can be used meanwhile to use our Minitron models.

```
git clone git@github.com:suiyoubi/transformers.git
cd transformers
git checkout 63d9cb0
pip install .
```
The following code provides an example of how to load the Minitron-8B model and use it to perform text generation.

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

### TRT-LLM

The following steps provide an example of how to load the Minitron-8B model in the `.nemo` checkpoint format. You can download the corresponding `.nemo` checkpoints here: [Minitron-8B-Base](https://huggingface.co/nvidia/Minitron-8B-Base/tree/main/nemo) and [Minitron-4B-Base](https://huggingface.co/nvidia/Minitron-4B-Base/tree/main/nemo).

1. Export TensorRT-LLM checkpoint.

    First launch the NeMo container `nvcr.io/nvidia/nemo:24.05` with the `.nemo` model checkpoint and [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) folder mounted:

    ```
    git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
    docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --init -it -v <TensorRT-Model-Optimizer_directory>:/workspace/TensorRT-Model-Optimizer -v <minitron_model_directory>:/workspace/minitron --rm nvcr.io/nvidia/nemo:24.05 bash
    ```

    Inside the container, run the following commands to export the TensorRT-LLM checkpoint:
    ```
    export GPT_MODEL_FILE=<minitron_nemo_file_directory>
    pip install "nvidia-modelopt[torch]" -U
    cd TensorRT-Model-Optimizer/llm_ptq/
    scripts/nemo_example.sh --type gptnext --model $GPT_MODEL_FILE --quant bf16 --tp 1 --task "build"
    ```

    You will see something like:

    ```
    Model config exported to: <TensorRT-LLM_checkpoint_directory>. Total time used ** s.
    ```

    which means the TensorRT-LLM checkpoint has been exported successfully.

2. Export TensorRT engine.

    Use docker to build and run [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) following these [instructions](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html):
    
    ```
    # TensorRT-LLM uses git-lfs, which needs to be installed in advance.
    apt-get update && apt-get -y install git git-lfs
    git lfs install

    git clone https://github.com/NVIDIA/TensorRT-LLM.git
    cd TensorRT-LLM
    git submodule update --init --recursive
    git lfs pull
    make -C docker release_build
    ```
    
    Now copy the exported TensorRT-LLM checkpoint to directory of TensorRT-LLM and launch the docker container:
    
    ```
    cp -r <TensorRT-LLM_checkpoint_directory> <TensorRT-LLM_directory>
    cd <TensorRT-LLM_directory>
    make -C docker release_run
    ```

    Inside the docker container, build TensorRT engine:

    ```
    trtllm-build --checkpoint_dir /code/tensorrt_llm/<TensorRT-LLM_directory> --gpt_attention_plugin bfloat16 --gemm_plugin bfloat16 --output_dir <trt_engine_directory>
    ```

    Run inference with the built TensorRT engine to summarize articles from the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset:

    ```
    python3 example/summarize.py --test_trt_llm --no_add_special_tokens --engine_dir <trt_engine_directory> --vocab_file <TensorRT-LLM_checkpoint_directory>/tokenizer.model
    ```

### Fine-tuning with LMFlow
[LMFlow](https://github.com/OptimalScale/LMFlow) is a complete pipeline for fine-tuning large language models. The following steps provide an example of how to fine-tune the ``Minitron-8B-Base`` models using LMFlow with the `alpaca` dataset.

1. Install LMFlow

    ```
    git clone https://github.com/OptimalScale/LMFlow.git
    cd LMFlow
    bash install.sh
    ```
  
2. Prepare the dataset
  
      Download the [alpaca](https://huggingface.co/datasets/wikitext) dataset and preprocess it using the following command.
  
      ```bash
      cd data && ./download.sh alpaca && cd -
      ```

3. Fine-tune the model
  
      Fine-tune the Minitron-8B model on the Wikitext-103 dataset using the following command.
    
      ```bash
      bash ./scripts/run_finetune.sh \
        --model_name_or_path nvidia/Minitron-8B-Base \
        --dataset_path data/alpaca/train_conversation \
        --output_model_path output_models/finetuned_minitron
      ```

With LMFlow, you can also fine-tune the model on your custom dataset. The only thing you need to do is transform your dataset into the [LMFlow data format](https://optimalscale.github.io/LMFlow/examples/DATASETS.html).
In addition to full-finetuniing, you can also fine-tune minitron efficiently with [LoRA](https://github.com/OptimalScale/LMFlow?tab=readme-ov-file#lora), [LISA](https://github.com/OptimalScale/LMFlow?tab=readme-ov-file#lisa), [Flash Attention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md), and other acceleration techniques.
  

## License

Minitron is released under the [NVIDIA Open Model License Agreement](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).


## Acknowledgments

We would like to thank Ameya Sunil Mahabaleshwarkar, Hayley Ross, Brandon Rowlett, Oluwatobi
Olabiyi, Ao Tang, and Yoshi Suhara for help with producing the instruction-tuned versions of
MINITRON; additionally, James Shen for TRT-LLM support, and Sanjeev Satheesh, Oleksii Kuchaiev,
Shengyang Sun, Jiaqi Zeng, Zhilin Wang, Yi Dong, Zihan Liu, Rajarshi Roy, Wei Ping, and Makesh
Narsimhan Sreedhar for help with datasets; Ao Tang for HF support. We’d also like to gratefully acknowledge the insightful
discussion and feedback from Chenhan Yu and Daniel Korzekwa.

## Citation

If you find our work helpful, please consider citing our paper:
```
@article{minitron2024,
      title={Compact Language Models via Pruning and Knowledge Distillation}, 
      author={Saurav Muralidharan and Sharath Turuvekere Sreenivas and Raviraj Joshi and Marcin Chochowski and Mostofa Patwary and Mohammad Shoeybi and Bryan Catanzaro and Jan Kautz and Pavlo Molchanov},
      journal={arXiv preprint arXiv:XXX},
      year={2024}
}
```
