# REMARK-LLM

**[USENIX Scurity'24] Remark-llm: A robust and efficient watermarking framework for generative large language models**

[Paper](https://arxiv.org/abs/2310.12362)

#### Environment

**Important: Plz don't install huggingface transformer**

```bash
pip install -r requirements.txt
```

#### Experiment

Train on the hc3 dataset

```bash
$ bash bash/run.sh train
```

Inference on the chatgpt dataset

```bash
$ bash bash/run.sh val
```

#### Citation

If you found our code/paper helpful, please kindly cite:

```latex
@inproceedings{zhang2024remark,
  title={$\{$REMARK-LLM$\}$: A Robust and Efficient Watermarking Framework for Generative Large Language Models},
  author={Zhang, Ruisi and Hussain, Shehzeen Samarah and Neekhara, Paarth and Koushanfar, Farinaz},
  booktitle={33rd USENIX Security Symposium (USENIX Security 24)},
  pages={1813--1830},
  year={2024}
}
```
