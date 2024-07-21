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
@article{zhang2023remark,
  title={Remark-llm: A robust and efficient watermarking framework for generative large language models},
  author={Zhang, Ruisi and Hussain, Shehzeen Samarah and Neekhara, Paarth and Koushanfar, Farinaz},
  journal={arXiv preprint arXiv:2310.12362},
  year={2023}
}
```