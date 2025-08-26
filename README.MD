# T5-to-Qwen

A handy script for distillation training Qwen3 on T5-xxl output with an accompanying inference script for the Chroma image generation model. [Credit to lodestone-rock for the inference code](https://github.com/lodestone-rock/flow/tree/master), which was largely just taken and bundled together in one script for simplicity.

In order to address the challenge of sequential, feature size and general 'form' mismatch, a small projection layer is trained. The arrangement by default when using the 0.6B parameter model is:

linear_in (1024) -> transformer (1024) -> upscale (1024-4096) -> activation (4096) -> linear_out (4096)

Simply, we take the transformer's ability to project sequentially, and the dimensional upscaling & granular refinement of the linear upscale and activation layers respectively, to approximate the T5-xxl output.

Of course, we also use the T5-xxl attention mask exclusively for both loss calculation and inference, since we are projecting to match the T5-xxl sequence.

### Quick installation
---
'''
pip install torch --extra-index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
'''

You're recommended to use a 3.11 Python venv, as that's what I've been using. Configuration options are provided at the top of both scripts. You can pass a --t5 flag to the inference script to use T5-xxl instead.

### Credit
---
- [lodestone-rock](https://huggingface.co/lodestones) for the inference script and the Chroma image generation model
- [Qwen](https://huggingface.co/Qwen) for the Qwen large language models
- [Google](https://huggingface.co/google) for the original T5-xxl model
- [DeepSeek](https://huggingface.co/deepseek-ai) for their R1 0528 model, which I used to generate some code used here
- [unsloth](https://huggingface.co/unsloth) for the VRAM-efficient Qwen3 training
And especially, thanks to those continuing to release open-source code and open-weight models with permissive licensing for the community!
