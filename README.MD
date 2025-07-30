These scripts were hacked together to provide support for using the Qwen3 Text Embedder model to produce text embeddings for inference with the Chroma image generation model, and for distillation training Qwen3 using T5-xxl as the teacher in order to produce text embeddings that more closely match those of the T5-xxl model. This was only tested on Arch Linux, using an Nvidia GPU, and the below instructions are written for this environment only.

---
Latest Update:
---
30/07/2025
---

- Adjusted the loss function. We now get much better results: for the first time I was able to get fully coherent images and stronger prompt comprehension from just a further hour training on the model we previously trained.

- Requires creating a new dataset cache. Projection layer is unchanged though.

---
Installation:
---

Before proceeding, ensure you are using Python 3.12. Personally, I use pyenv to make a 3.12 venv:

  pacman -S pyenv

  pyenv init

Follow the instructions given, then:

  pyenv install 3.12

  pyenv global 3.12

In the desired folder for the venv:

  python -m venv venv

  pyenv global system

  source venv/bin/activate.fish

Now, install pytorch:

  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

Then, install the requirements:

  pip install -r requirements.txt

You'll also want to install flash attention:

  pip install flash-attn --no-build-isolation

Of course, you'll also want to download the models. Please note that the script expects BF16. For T5-xxl I recommend using the BF16 safetensors file listed here: https://huggingface.co/silveroxides/flan-t5-xxl-encoder-only/tree/main

---
Now you can run either script, we hope. You'll want to set the correct paths in the scripts, and you can look at some other configuration options. The inference script uses T5-xxl by default, so pass the --qwen flag to use Qwen3. (I'll change this behaviour when and if I get the model trained sufficiently).

Neither script supports any other form of quantisation, so don't expect Q-quants to work or anything like that. Memory use when training can sit between 16~18GB and you'll want overhead there so realistically, you might be best using this with a 24GB card.

Most of the inference code was taken from lodestone-rock's repo here: https://github.com/lodestone-rock/flow
Much credit to them for the code and, of course, the Chroma model itself.
A small amount was also taken from tdrussell's repo here: https://github.com/tdrussell/diffusion-pipe

Additionally, thanks to: 
Alibaba, who released these great & local-friendly Qwen LLMs with the permissive Apache-2.0 license;
Unsloth for the Qwen3 full fine-tuning support;
The Google engineers responsible for the T5-xxl model;
Deepseek, whose R1 0528 model I used for some of the script;
Ubergarm, who produced the very tiny (130GB!) Deepseek R1 quant I used & recommend; and
ZUN, because I 1CC'd a few Touhou games while my GPU was occupied.

Note: I'm a hack and I'd be surprised if the code doesn't have issues, even if it technically works.
