# DAA
Pytorch implementation of "Blind Image Super-Resolution with Degradation-Aware Adaptation" (ACCV 2022).


## Requirements
- Python 3.6
- PyTorch >= 1.6.0
- numpy
- skimage
- imageio
- matplotlib
- cv2


## Train
### 1. Prepare training data 

1.1 Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset and the [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset.

1.2 Combine the HR images from these two datasets in `your_data_path/DIV_FLI/HR` to build the DF2K dataset. 

### 2. Begin to train
Run `./main_IMBD_nn_tt.py` to train on the DF2K dataset. Please update `dir_data` in the `option_IMBD_nn_tt.py` file as `your_data_path`.


## Test
### 1. Prepare test data 
Download [benchmark datasets](https://github.com/xinntao/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) (Set14) and prepare HR/LR images in `your_data_path/benchmark`.
(Here, we recommand to generate the LR images with various blur kernel and noise from HR images in advance, and save the LR images in certain folders.)

Download [RealSR(V3)](https://github.com/csjcai/RealSR) and directly use the HR/LR images in `your_data_path/RealSR(V3)/train_x4`.


### 2. Begin to test
Run `./Eval_IMBD_nn_tt.py` to test on benchmark dataset (one in-domain data and one out-domain data as example and real world data with our pretrained model.

If you want to test on your model, please set the `pre_train` as `your_model_pth` in `option_eval_IMBD_nn_tt.py` to load your model.


## Citation
```
@InProceedings{Wang2022DAA,
  author    = {Wang, Yue and Ming, Jiawen and Jia, Xu and Elder, James and Lu, Huchuan},
  title     = {Blind Image Super-Resolution with Degradation-Aware Adaptation},
  booktitle = {Proceedings of the asian conference on computer vision},
  year      = {2022},
}
```

## Acknowledgements
This code is built on [Unsupervised Degradation Representation Learning for Blind Super-Resolution (PyTorch)](https://github.com/LongguangWang/DASR). We thank the authors for sharing the codes.

