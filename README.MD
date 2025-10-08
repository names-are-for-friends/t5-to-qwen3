# T5-to-Qwen

![Qwen](1000+11000_teacher_5taw_15e5proj.png)

---

A handy script for distillation training Qwen3 on T5-xxl output with an accompanying inference script for the Chroma image generation model. [Credit to lodestone-rock for the inference code](https://github.com/lodestone-rock/flow/tree/master), which was largely just taken and bundled together in one script for simplicity.

Several training approaches are supported. Using token alignment (via the input text per token) for the loss calculation, we can fit the T5-xxl token's semantic meaning to the Qwen tokens in the (typically shorter) Qwen sequence; this approach fits closer to the target than using mean pooling loss, but is not perfect. We can also interpolate the sequence length to match the T5-xxl embedding, using a combination of positional token loss for the Qwen-mask-attended sequence and mean pooling loss for the solely T5-mask-attended extended space.

The script allows you to specify which layers to bolt on to the model. A simple configuration would be to use a Transformer->MLP->Linear chain that takes the 1024 embedding dim and outputs to 4096, which you would train using alignment loss. Layer types other than "linear" are bundled with an input linear layer, so this type is mainly for output. For expansion of the sequence, use the "interpolation" type with both token and sequence losses and their associated flags enabled to limit visibility to the appropriate tokens.

### Quick installation
---
```
pip install torch --extra-index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```
Currently, the script is only tested on Linux, so I don't know if there are any incompatibilities with Windows.

Provided default settings will run on 24GB VRAM with cached T5-xxl embeddings and the 0.6B Qwen3 model. 16GB should work, possibly with adjusted batch_size. Maybe 12GB too.

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
