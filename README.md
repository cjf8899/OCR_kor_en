# OCR_kor_en
This repo is an End to End code that **detection language in images and recognition Korean and English.**
## Demo
<img src="https://user-images.githubusercontent.com/53032349/170912999-ed3f35a7-1a8c-4ece-9147-ed518bd75795.gif" alt="ocr_demo" width="400"/>   <img src="https://user-images.githubusercontent.com/53032349/170913056-f999b44f-2c10-4b6d-b9b5-4565e0a02518.gif" alt="ocr_demo2" width="400"/>

<img src="https://user-images.githubusercontent.com/53032349/129499027-8610143e-7174-4278-baca-6c6c0b5c5453.png" width="100%" height="100%" title="70px" alt="memoryblock"><br>

<img src="https://user-images.githubusercontent.com/53032349/170912079-0c94c210-04ae-4daa-8919-7467ba110c7a.jpg" alt="ocr_demo3" width="400"/>   <img src="https://user-images.githubusercontent.com/53032349/170912170-000f15ac-6d85-49ab-97f4-6c50b6a23dbd.jpg" alt="ocr_demo4" width="400"/>


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

## Video2img

Crop the video frame by frame and apply a morphology operation for increase the recognition rate.<br>

<p align="center"><img src="https://user-images.githubusercontent.com/53032349/170910668-98864001-9ab4-41e6-b833-1c4905b44e43.png" width="70%" height="70%" title="70px" alt="memoryblock"></p><br>

Put your video name in video_name. (ex: ocr_test.mp4)<br>

```Shell
python video2img.py --video_name ocr_test.mp4
```

Also, you can use images, not video

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
    -- sample_video
        -- video_name
            -- 00000.jpg
            -- 00001.jpg
            -- 00002.jpg
            ...
        -- video_name_ori
            -- 00000.jpg
            -- 00001.jpg
            -- 00002.jpg
            ...
    -- main.py
    ....
```
## Requirements
* PyTorch>=0.4.1
* torchvision>=0.2.1
* opencv-python>=3.4.2
* ...
```Shell
pip install -r requirements.txt
```

## Run
```Shell
python main.py --test_folder ./sample_img/
```

or

```Shell
python main.py --test_folder ./sample_video/video_name
```

# Referenced. Thank you all:+1:
detection code : https://github.com/clovaai/CRAFT-pytorch<br>
recognition code : https://github.com/clovaai/deep-text-recognition-benchmark<br>
korean ocr code : https://github.com/parksunwoo/ocr_kor<br>
aihub dataset : https://aihub.or.kr/aidata/133<br>
