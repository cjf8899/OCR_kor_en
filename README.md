# OCR_kor_en

This repo is an end to end code that **detection language in images and recognition Korean and English.**
## Demo
<img src="https://user-images.githubusercontent.com/53032349/129498905-0cb49bea-9843-4ab6-b45b-52892079b9e0.png" width="100%" height="100%" title="70px" alt="memoryblock"><br>

## Getting Started
Download models
* [craft_mlt_25k.pth](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
  * The language detection model used [craft's](https://github.com/clovaai/CRAFT-pytorch) model.
* [classifi_97.29.pth](https://drive.google.com/file/d/1h8A3thdxsLKHyNvjhR3qUtKW4tBxZF2w/view?usp=sharing)
  * This is a Korean and English classification model learned using data from aihub.
* [kor_reg_97.42.pth](https://drive.google.com/file/d/1e-MEl4sHn8B1w8xkECKtlIMyRzrneU3u/view?usp=sharing)
  * This is a korean recognition model learned using data from aihub.
* [TPS-ResNet-BiLSTM-Attn.pth](https://drive.google.com/file/d/1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9/view?usp=sharing)
  * The English recognition model used [clovaai's](https://github.com/clovaai/deep-text-recognition-benchmark) model.

the structures would like
```
~/OCR_kor_en/
    -- model
        -- craft_mlt_25k.pth
        -- classifi_97.29.pth
        -- kor_reg_97.42.pth
        -- TPS-ResNet-BiLSTM-Attn.pth
    -- sample_img
        -- your_img.jpg
    -- main.py
    ....
```
## Requirements
* PyTorch>=0.4.1
* torchvision>=0.2.1
* opencv-python>=3.4.2
* lmdb
* pillow
* nltk
* natsort
* check requiremtns.txt
```Shell
pip install -r requirements.txt
```

## Run
```Shell
python main.py
```

# Referenced. Thank you all:+1:
detection code : https://github.com/clovaai/CRAFT-pytorch<br>
recognition code : https://github.com/clovaai/deep-text-recognition-benchmark<br>
korean ocr code : https://github.com/parksunwoo/ocr_kor<br>
