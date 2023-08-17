
# Model Converters
We  convert pre-trained tensorflow model to pytorch model, using [Converter](https://github.com/genforce/genforce/tree/master/converters) of genforce.

## Usage

The script to transform image to gradients by pytorch model:

```shell
wget -O karras2019stylegan-bedrooms-256x256_discriminator.pth

sh ./transform_img2grad_pytorch.sh {GPU-ID} {Data-Root-Dir} {Grad-Save-Dir}
```




