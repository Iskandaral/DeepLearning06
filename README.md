# Learning-to-See-in-the-Dark

This is a Tensorflow implementation of Learning to See in the Dark in CVPR 2018, by [Chen Chen](http://cchen156.github.io/), [Qifeng Chen](http://cqf.io/), [Jia Xu](http://pages.cs.wisc.edu/~jiaxu/), and [Vladlen Koltun](http://vladlen.info/).  

[Project Website](http://cchen156.github.io/SID.html)<br/>
[Paper](http://cchen156.github.io/paper/18CVPR_SID.pdf)<br/>

![teaser](images/fig1.png "Sample inpainting results on held-out images")

This code includes the default model for training and testing on the See-in-the-Dark (SID) dataset. 


## Demo Video
https://youtu.be/qWKUFK7MWvg

## Setup

### Requirement
Required python (version 2.7) libraries: Tensorflow (>=1.1) + Scipy + Numpy + Rawpy.

Tested in Ubuntu + Intel i7 CPU + Nvidia Titan X (Pascal) with Cuda (>=8.0) and CuDNN (>=5.0). CPU mode should also work with minor changes but not tested.

### Dataset

**Update Aug, 2018:** We found some misalignment with the ground-truth for image 10034, 10045, 10172. Please remove those images for quantitative results, but they still can be used for qualitative evaluations.

You can download it directly from Google drive for the [Sony](https://storage.googleapis.com/isl-datasets/SID/Sony.zip) (25 GB)  and [Fuji](https://storage.googleapis.com/isl-datasets/SID/Fuji.zip) (52 GB) sets. 

There is download limit by Google drive in a fixed period of time. If you cannot download because of this, try these links: [Sony](https://drive.google.com/open?id=1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx) (25 GB)  and [Fuji](https://drive.google.com/open?id=1C7GeZ3Y23k1B8reRL79SqnZbRBc4uizH) (52 GB).

New: we provide file parts in [Baidu Drive](https://pan.baidu.com/s/1fk8EibhBe_M1qG0ax9LQZA) now. After you download all the parts, you can combine them together by running: "cat SonyPart* > Sony.zip" and "cat FujiPart* > Fuji.zip".


The file lists are provided. In each row, there are a short-exposed image path, the corresponding long-exposed image path, camera ISO and F number. Note that multiple short-exposed images may correspond to the same long-exposed image. 

The file name contains the image information. For example, in "10019_00_0.033s.RAF", the first digit "1" means it is from the test set ("0" for training set and "2" for validation set); "0019" is the image ID; the following "00" is the number in the sequence/burst; "0.033s" is the exposure time 1/30 seconds.  


### Testing
1. Clone this repository.
2. Download the pretrained models by running
```Shell
python download_models.py
```
3. Run "python test_Sony.py". This will generate results on the Sony test set.
4. Run "python test_Fuji.py". This will generate results on the Fuji test set.

By default, the code takes the data in the "./dataset/Sony/" folder and "./dataset/Fuji/". If you save the dataset in other folders, please change the "input_dir" and "gt_dir" at the beginning of the code. 

### Training new models
1. To train the Sony model, run "python train_Sony.py". The result and model will be save in "result_Sony" folder by default. 
2. To train the Fuji model, run "python train_Fuji.py". The result and model will be save in "result_Fuji" folder by default. 

By default, the code takes the data in the "./dataset/Sony/" folder and "./dataset/Fuji/". If you save the dataset in other folders, please change the "input_dir" and "gt_dir" at the beginning of the code.

Loading the raw data and processing by Rawpy takes significant more time than the backpropagation. By default, the code will load all the groundtruth data processed by Rawpy into memory without 8-bit or 16-bit quantization. This requires at least 64 GB RAM for training the Sony model and 128 GB RAM for the Fuji model. If you need to train it on a machine with less RAM, you may need to revise the code and use the groundtruth data on the disk. We provide the 16-bit groundtruth images processed by Rawpy: [Sony](https://drive.google.com/file/d/1wfkWVkauAsGvXtDJWX0IFDuDl5ozz2PM/view?usp=sharing) (12 GB)  and [Fuji](https://drive.google.com/file/d/1nJM0xYVnzmOZNacBRKebiXA4mBmiTjte/view?usp=sharing) (22 GB). 


## Citation
If you use our code and dataset for research, please cite our paper:

Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.

### License
MIT License.

## FAQ
1. Can I test my own data using the provided model? 

The proposed method is designed for sensor raw data. The pretrained model probably not work for data from another camera sensor. We do not have support for other camera data. It also does not work for images after camera ISP, i.e., the JPG or PNG data.

2. Will this be in any product?

This is a research project and a prototype to prove a concept. 

3. How can I train the model using my own raw data? 

Generally, you just need to subtract the right black level and pack the data in the same way of Sony/Fuji data. If using rawpy, you need to read the black level instead of using 512 in the provided code. The data range may also differ if it is not 14 bits. You need to normalize it to [0,1] for the network input. 

4. Why the results are all black?

It is often because the pre-trained model not downloaded properly. After downloading, you should get 4 checkpoint related files for the model. 


## Questions
If you have additional questions after reading the FAQ, please email to cchen156@illinois.edu.

#Reproducing "Learning to See in the Dark" with Reduced Dataset Size and Epochs

This is a reproduction of the model presented in the paper "Learning to See in the Dark" by Chen et al. (2018) with a smaller dataset and fewer epochs. The aim of this reproduction is to investigate the performance of different CNN architectures, specifically SegNet and DAnet, and compare them with the U-Net architecture used in the original paper. The source code has been rewritten in Python 3.8 from the original code, which was in Python 2.7.

##Steps

The steps to train and test the models are the same as the ones in the original paper, which can be found in the repository https://github.com/cchen156/Learning-to-See-in-the-Dark. The difference is that we have reduced the dataset size and the number of epochs for each of the three models used in this reproduction.

To train the desired model, run one of the following commands in the command line:
```bash
python train_Sony_Unet.py
python train_Sony_segnet.py
python train_Sony_danet.py
```

After training, to test the model, run one of the following commands:

```bash
python test_Sony_Unet.py
python test_Sony_segnet.py
python test_Sony_danet.py
```

If the user has a better GPU and more patience, they can change the dataset size and the number of epochs in the train files to train the model on the whole dataset with the same number of epochs as the original paper and thus get better results than we did.

##Conclusion

The results of this reproduction show that the U-Net architecture performs better than the SegNet and DAnet architectures on a reduced dataset size and fewer epochs. One reason for this could be that we did not change the hyperparameters of the U-Net architecture while using the same hyperparameters for the other architectures. The results also show that the DAnet architecture has a much faster training time and smaller model size than the other two architectures.
