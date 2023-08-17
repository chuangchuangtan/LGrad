
# Model Converters
We convert pre-trained tensorflow model to pytorch model, using [Converter](https://github.com/genforce/genforce/tree/master/converters) of [enforce](https://github.com/genforce/genforce).

## Usage

The script to transform image to gradients using pytorch model:

```shell
wget https://lid-1302259812.cos.ap-nanjing.myqcloud.com/tmp/karras2019stylegan-bedrooms-256x256_discriminator.pth -O karras2019stylegan-bedrooms-256x256_discriminator.pth

sh ./transform_img2grad_pytorch.sh {GPU-ID} {Data-Root-Dir} {Grad-Save-Dir}
```




