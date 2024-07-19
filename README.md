# Minitron

<p align="center">
<img src="https://www.sauravm.com/assets/img/minitron.png"  width="256">
</p>
<p align="center">
        🤗 <a href="https://huggingface.co/nvidia">Hugging Face Models</a>&nbsp&nbsp | &nbsp&nbsp 📄 <a href="">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📜 <a href="">Blog</a> &nbsp | &nbsp 💬 <a href="">Demo</a>
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
The following code provides an example of how to load the Minitron-8B model and use it to perform text generation.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_path = "nvidia/minitron-8b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Prepare the input text
prompt = "To be or not to be,"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate the output
output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode and print the output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
```

### TRT-LLM

The following steps provide an example of how to load the Minitron-8B model in the `.nemo` checkpoint format.

1. Export TensorRT-LLM checkpoint

    First launch the NeMo container `nvcr.io/nvidia/nemo:24.05` with the `.nemo` model checkpoint and `TensorRT-Model-Optimizer` folder mounted.

    ```
    git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
    docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --init -it -v <TensorRT-Model-Optimizer_directory>:/workspace/TensorRT-Model-Optimizer -v <minitron_model_directory>:/workspace/minitron --rm nvcr.io/nvidia/nemo:24.05 bash
    ```

    Inside the container, run the following commands to export TensorRT-LLM checkpoint.
    ```
    export GPT_MODEL_FILE=<minitron_nemo_file_directory>
    pip install "nvidia-modelopt[torch]" -U
    cd llm_ptq/
    scripts/nemo_example.sh --type gptnext --model $GPT_MODEL_FILE --quant bf16 --tp 1
    ```

2. Export TensorRT engine

    Use docker to build and run TensorRT-LLM following this [instructions](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html).
    
    ```
    # TensorRT-LLM uses git-lfs, which needs to be installed in advance.
    apt-get update && apt-get -y install git git-lfs
    git lfs install

    git clone https://github.com/NVIDIA/TensorRT-LLM.git
    cd TensorRT-LLM
    git submodule update --init --recursive
    git lfs pull
    ```
    
    Now copy the exported TensorRT-LLM checkpoint to directory of TensorRT-LLM and launch the docker container
    
    ```
    cp -r <TensorRT-LLM_checkpoint_directory> <TensorRT-LLM_directory>
    cd <TensorRT-LLM_directory>
    make -C docker release_build
    ```

    Inside the docker container, build TensorRT engine.

    ```
    trtllm-build --checkpoint_dir /code/tensorrt_llm/<TensorRT-LLM_directory> --gpt_attention_plugin bfloat16 --gemm_plugin bfloat16 --output_dir <trt_engine_directory>
    ```

    Run inference with the built TensorRT engine to summarize articles from the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

    ```
    python3 example/summarize.py --test_trt_llm --no_add_special_tokens --engine_dir <trt_engine_directory> --vocab_file <TensorRT-LLM_checkpoint_directory>/tokenizer.model
    ```



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
