# OCR_kor_en

This repo is an end to end code that **detection language in images and recognition Korean and English.**

# Getting Started
Download models

For the language detection model, (craft's model)[https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ] was used.
```Shell
git clone https://github.com/cjf8899/OCR_kor_en.git

cd OCR_kor_en

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar
mv ./VOCdevkit ./VOCtrainval2007

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xvf VOCtrainval_11-May-2012.tar
mv ./VOCdevkit ./VOCtrainval2012

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
mv ./VOCdevkit ./VOCtest2007
```

Download git and dataset
```Shell
git clone https://github.com/cjf8899/OCR_kor_en.git

cd OCR_kor_en

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar
mv ./VOCdevkit ./VOCtrainval2007

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xvf VOCtrainval_11-May-2012.tar
mv ./VOCdevkit ./VOCtrainval2012

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
mv ./VOCdevkit ./VOCtest2007
```

the structures would like
```
~/VOCtrainval2007/
    -- VOC2007
~/VOCtrainval2012/
    -- VOC2012  
~/VOCtest2007/
    -- VOC2007  
```
# Train

pre-train [VGG](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth) weights (https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth)

pre-train [ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) weights (https://download.pytorch.org/models/resnet18-5c106cde.pth)

pre-train [ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) weights (https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)
```
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
wget https://download.pytorch.org/models/resnet18-5c106cde.pth
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
```

Select and use the desired model

# Results

The [DSSD](https://arxiv.org/pdf/1701.06659.pdf) paper measured the performance of resnet101. The image size of the DSSD paper is 321.

|              Implementation              |     mAP     |
| :--------------------------------------: | :---------: |
| SSD paper - VGG |    77.2    |
| DSSD paper - ResNet101 |    77.1    |
|    this repo - VGG   | 78.9 |
|    this repo - ResNet18   | 70.1 |
|    this repo - ResNet101   | 77.5 |

The SSD paper used Xavier after the base network(VGG), but did not use Xavier because this repo used relu.

Other experiments..

|              Implementation              |     mAP     |
| :--------------------------------------: | :---------: |
|   ResNet18_No_pretrain   | 54.4 |
|    VGG_No_Augmentation   | 66.8 |

# Paper Review
VGG PR : https://cjf8899.github.io/image%20recognition/VGG/

ResNet PR : https://cjf8899.github.io/image%20recognition/ResNet/

ResNeXt PR : https://cjf8899.github.io/image%20recognition/ResNeXt/


# Referenced. Thank you all
ssd code : https://github.com/HosinPrime/simple-ssd-for-beginners
