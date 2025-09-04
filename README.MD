# T5-to-Qwen

![Chwon](trans_pt_18e5_TM.png)
_Our iconic mascot, Chwon_
_(result of just 500 steps x 64 batch size of training from base Qwen3 0.6B Text Embedder, 1e-5->18e-5 warmup)_

A handy script for distillation training Qwen3 on T5-xxl output with an accompanying inference script for the Chroma image generation model. [Credit to lodestone-rock for the inference code](https://github.com/lodestone-rock/flow/tree/master), which was largely just taken and bundled together in one script for simplicity.

After experimentation and deciding on my likely final training settings, the default settings in the script are the recommended settings that you should use for training. Note that when using the transformer->mlp projection, you should always use the T5-xxl mask for inference, since the transformer projection will match to the longer T5-xxl sequence. (If using dropout, you will likely project beyond the T5-xxl sequence, so test outputs with varying levels of additional attention).

### Quick installation
---
```
pip install torch --extra-index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```
Currently, the script is only tested on Linux, so I don't know if there are any incompatibilities with Windows.

Provided default settings will run on 24GB VRAM with cached T5-xxl embeddings and the 0.6B Qwen3 model. If you have 16GB of VRAM, try reducing the batch size to 32, and for 12GB, a batch size of 16 should work.

You're recommended to use a 3.11 Python venv, as that's what I've been using. 

Configuration options are provided at the top of both scripts. You can pass a --t5 flag to the inference script to use T5-xxl instead.

### Credit
---
- [lodestone-rock](https://huggingface.co/lodestones) for the inference script and the Chroma image generation model
- [Qwen](https://huggingface.co/Qwen) for the Qwen large language models
- [Google](https://huggingface.co/google) for the original T5-xxl model
- [DeepSeek](https://huggingface.co/deepseek-ai) for their R1 0528 model, which I used to generate some code used here
- [unsloth](https://huggingface.co/unsloth) for the VRAM-efficient Qwen3 training

And especially, thanks to those continuing to release open-source code and open-weight models with permissive licensing for the community!
