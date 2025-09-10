# T5-to-Qwen

![Qwen](1000+11000_teacher_5taw_15e5proj.png)

---

A handy script for distillation training Qwen3 on T5-xxl output with an accompanying inference script for the Chroma image generation model. [Credit to lodestone-rock for the inference code](https://github.com/lodestone-rock/flow/tree/master), which was largely just taken and bundled together in one script for simplicity.

Work-in-progress. Current approach: use token alignment (matching tokens in a sequential window by text content) and per-token losses for 500~1000 steps (64 batch size, adjust accordingly), then use 1:1 positional per-token alignment once the 1:1 projected-student-to-teacher-output relationship is patterned enough (token alignment won't work once the transformer shifts the embedding sequence out of alignment with the token sequence, after all). Results so far (5000 steps in) are promising, with prompt comprehension reliably increasing and clear text present in the output. Initial outputs will have some noise, but by 5000 steps noise seems to be gone, so we seem to have defeated the complete collapse that previously happened with solely 1:1 alignment.

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
