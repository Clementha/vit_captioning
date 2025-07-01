# VIT_CAPTIONING
VIT Captioning is an AI Model that can generate caption describing a picture

## To train the model
```sh
$ python train.py
```

## To evaluate a picture
Update this line to load a picture:
```python
image = Image.open("./images/manOnBike.png").convert("RGB")
```
then:
```sh
$ python evaluate.py
```