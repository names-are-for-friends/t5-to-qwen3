# T5-to-Qwen

![Qwen](1000+11000_teacher_5taw_15e5proj.png)

---

A handy script for distillation training Qwen3 on T5-xxl output with an accompanying inference script for the Chroma image generation model. [Credit to lodestone-rock for the inference code](https://github.com/lodestone-rock/flow/tree/master), which was largely just taken and bundled together in one script for simplicity.

Work-in-progress so don't expect everything to necessarily work at the moment. 

We use a direct text-matching method to pair tokens, and a method for matching split tokens (multi-to-one or multi-to-multi) to ensure high coverage while maintaining consistently patterned matches.

Layers should be bolted on. It's recommended to use T5's output RMSNorm at the end, and you can automatically extract its weights from the T5-xxl model via the script. A T5-style t5_encoder option is included, but it's better to use the standard transformer_encoder layer type. Usually it's best to train one transformer at a time.

### Quick installation
---
```
pip install torch --extra-index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```
Currently, the script is only tested on Linux, so I don't know if there are any incompatibilities with Windows.

You're recommended to use a 3.11 Python venv, as that's what I've been using. 

Configuration options are provided at the top of both scripts. You can pass a --t5 flag to the inference script to use T5-xxl instead.

### VRAM Recommendation
---

24GB will have absolutely no problem running the script with a batch size of 32, unless you decide to pile up huge transformers (ie hidden_dim = 4096), as a lot of memory is allocated for these in training mode.

16GB is viable, but you will probably want to reduce the batch size to 16, and/or exclusively use small transformer layers (ie. hidden_dim = 1024).

12GB should work fine too if you are very conservative with batch size and, again, exclusively use small transformer layers. I wouldn't recommend going much lower than this, though.

### Credit
---
- [lodestone-rock](https://huggingface.co/lodestones) for the inference script and the Chroma image generation model
- [Qwen](https://huggingface.co/Qwen) for the Qwen large language models
- [Google](https://huggingface.co/google) for the original T5-xxl model
- [DeepSeek](https://huggingface.co/deepseek-ai) for their R1 0528 model, which I used to generate some code used here
- [unsloth](https://huggingface.co/unsloth) for the VRAM-efficient Qwen3 training

And especially, thanks to those continuing to release open-source code and open-weight models with permissive licensing for the community!
