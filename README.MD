# T5-to-Qwen

![Qwen](1000+11000_teacher_5taw_15e5proj.png)

---

A handy script for distillation training Qwen3 on T5-xxl output with an accompanying inference script for the Chroma image generation model. [Credit to lodestone-rock for the inference code](https://github.com/lodestone-rock/flow/tree/master), which was largely just taken and bundled together in one script for simplicity.

Work-in-progress so don't expect everything to work at the moment. I'm still learning, so it's been hard-going, but I understand much more about the process now than I did at the beginning!

In order to match tokens between the differing tokenizers, we first map the words and their corresponding absolute start and end positions, then iterate through the Qwen in-word tokens, looking at the same normalised word-interior position in the T5 word for a matching token via token text. This accomplishes roughly 80-85% matching tokens, so the tokenizers aren't TOO different (actually, based on experimentation, even 30-40% is not too sparse for decent alignment, but this is better). Then, we calculate mean pooling loss with the remaining unmatched in-word tokens, and as a final fallback we match remaining unmatched tokens with approximate position via sequence length differential, looking in a window around the position, and accepting only matches over a cosine similarity threshold. Final token coverage is almost 100%. Experimentation with weighting needs to follow, and you may have better results by reducing the word match weights, since mean pooling losses might dilute the direct token matching losses.

Due to the small size of the 0.6B parameter Qwen3 Text Embedder model, you will need to bolt on additional transformer layers. I'm still experimenting to find the most efficient possible approach - hidden_dim = 1024, dim_feedforward = 4096 is likely the lowest you would want to go, with an MLP at 2048 or 4096 for refinement and a final linear layer to 4096. Transformer size exponentially increases with hidden_dim, but you might also want to try hidden_dim = 2048, dim_feedforward = 8192 for faster alignment. Again, I still haven't experimented sufficiently with this.

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
