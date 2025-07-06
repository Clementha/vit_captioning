# VIT_CAPTIONING

**VIT Captioning** is an AI Model that can generate caption describing a picture.

The code uses 
 - "bert-base-uncased" as vocabuary and tokenizer
 -  Visual Encoder using ViT ("google/vit-base-patch16-224-in21k") or 
 - CLIP("openai/clip-vit-base-patch32")
 - Flickr30k as Dataset 
 
To create the decoder model for generating captions describing a picture.

## To train the model

Properly set the parameters in train.py:

```python
    wandb.init(
        project="vit_captioning",
        name="VIT_model",
        config={
        "model": "ViTEncoder", # ViTEncoder or CLIPEncoder
        "epochs": 5,
        "batch_size": 32,
        "max_length": 50,
        "learning_rate": 1e-4,
        "num_workers": 4,
        "unfreeze_pct": 0.5, # Best practice: unfreeze encoder after 30% of epochs
        "encoder_lr_pct": 0.1, # lower LR for encoder after unfreezing
        "flatten_captions": True  # Flatten captions for training
    })

```

Save the file then start training with:
```sh
$  python  train.py
```
which will then generate a model checkpoint (.pth) file.
  

## To generate caption for a picture

Syntax is:

```sh
$  python  generate.py  \
--checkpoint [pathname of model checktpointFile]  \
--model [CLIPEncoder or  ViTEncoder]  \
--image [pathname of  picture]
```

Example:

```sh
$  python  generate.py  --checkpoint  ./artifacts/CLIPEncoder_40epochs_unfreeze12.pth  --model  CLIPEncoder  --image  ./images/girl.png
```
