# Face Swapping as A Simple Arithmetic Operation
This repository contains inference script for [Face Swapping as A Simple Arithmetic Operation](https://arxiv.org/abs/2211.10812).

![model](demo/model.jpg)

# Preparation
## Requirements

* Python
* PyTorch
* Opencv

## Pretrained Models

Download the following checkpoint to folder `weights`.

* [StyleGAN2 model](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view)
* [e4e FFHQ inversion](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view)
* [Face parser model](https://drive.google.com/file/d/1o1m-eT38zNCIFldcRaoWcLvvBtY8S4W3/view?usp=sharing)
* [Style extraction network](https://drive.google.com/file/d/1NTGSJLCGIp_edMX11XJgropFBv8v0ge0/view?usp=sharing)

## Inference
To swap two images, run
```
python inference.py \
    --src_img demo/source.jpg \
    --tgt_img demo/target.jpg \
    --save_dir demo/output.jpg
```

The swapped image will be saved as `demo/output.jpg`

Source image             |  Target image                |  Output
:-------------------------:|:-------------------------: |:-------------------------:
![](demo/source.jpg)  |  ![](demo/target.jpg)  |  ![](demo/output.jpg)