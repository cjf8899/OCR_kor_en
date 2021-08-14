"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import random
import string
import torchvision.transforms as transforms
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageDraw, ImageFont
import shutil
import cv2
from skimage import io
import craft_utils
import imgproc
import file_utils
import json
import zipfile
from craft import CRAFT
from collections import OrderedDict
from deep_utils import CTCLabelConverter, AttnLabelConverter, Averager
from deep_dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset,classification,lang_classifi
from deep_model import Model
from deep_test import validation
from apex.parallel import DistributedDataParallel as DDP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(kr_opt, en_opt, sample_img_path, crop_img_path, final_img_path):
    """ dataset preparation """

    AlignCollate_valid = AlignCollate(imgH=kr_opt.imgH, imgW=kr_opt.imgW, keep_ratio_with_pad=kr_opt.PAD)
    valid_dataset = classification(crop_img_path)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=kr_opt.batch_size,
        shuffle=False,  # 'True' to check training progress with validation function.
        num_workers=int(kr_opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    print('-' * 80)

    """ model configuration """
    
    converter = AttnLabelConverter(kr_opt.character)
    en_converter = AttnLabelConverter(en_opt.character)
    kr_opt.num_class = len(converter.character)
    en_opt.num_class = len(en_converter.character)
    

    kr_model = Model(kr_opt)
    en_model = Model(en_opt)
    
    kr_model.cuda(kr_opt.gpu)
    en_model.cuda(en_opt.gpu)
    kr_model.load_state_dict(torch.load(kr_opt.pretrain_model))
    state_dict = torch.load(en_opt.pretrain_model, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    en_model.load_state_dict(new_state_dict)
    
    print("Model...")
 
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    kr_model.eval()
    en_model.eval()
    with torch.no_grad():
        _, _, _, kr_preds, labels, _, _, real_label, img_path = validation(
            kr_model, criterion, valid_loader, converter, kr_opt)
        _, _, _, en_preds, en_labels, _, _, en_real_label, en_img_path = validation(
            en_model, criterion, valid_loader, en_converter, en_opt)
      
    
    kr_pred_list = list()
    en_pred_list = list()
    img_num_list = list()
    for kr_pred, gt, real, en_pred, en_gt, en_real, path in zip(kr_preds, labels, real_label, en_preds, en_labels, en_real_label, img_path):
        kr_pred = kr_pred[:kr_pred.find('[s]')]
        kr_pred_list.append(kr_pred)
        en_pred = en_pred[:en_pred.find('[s]')]
        en_pred_list.append(en_pred)
        img_num_list.append(path.split(crop_img_path)[1].split('_')[1])

    
    
    my_set = set(img_num_list) 
    img_num_list = list(my_set)
    fontpath = "./font/NanumBarunGothic.ttf"
    font = ImageFont.truetype(fontpath, 50)

    for img_num in img_num_list:
        image = cv2.imread(sample_img_path + str(img_num)+'.jpg')
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        for kr_pred, en_pred, path in zip(kr_pred_list, en_pred_list, img_path):
            if img_num == path.split(crop_img_path)[1].split('_')[1]:
                x, y, xw, yh = path.split('_')[-2].split('|')
                lang = path.split('_')[-1].split('.jpg')[0]
                cv2.rectangle(image, (int(x), int(y)), (int(xw), int(yh)) , (0, 0, 255), 5)
                draw.rectangle(((int(x), int(y)), (int(xw), int(yh))), outline=(0, 0, 255), width=2)
                cv2.rectangle(image, (int(x), int(y)), (int(xw), int(yh)), (0,0,255), 5)
            
                
                draw.text((int(x), int(y)+10), kr_pred, (255,0,0), font=font)
                draw.text((int(x), int(y)+60), en_pred, (0,255,0), font=font)
                
        image = np.array(img_pil)    
        cv2.imwrite(final_img_path + 'final_' + str(img_num) + '.jpg', image)
                
    

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")



def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, craft_opt,refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, craft_opt.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=craft_opt.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if craft_opt.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='./model/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.6, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.45, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=1, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='./sample_img/', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='./model/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    craft_opt = parser.parse_args()


    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(craft_opt.test_folder)

    result_folder = './ocr_result/'
    crop_img_folder = './ocr_crop_img/'
    rename_crop_img_folder = './ocr_rename_crop_img/'
    final_img_folder = './ocr_final_img/'

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    else:
        shutil.rmtree(result_folder)
        os.mkdir(result_folder)

    if not os.path.isdir(crop_img_folder):
        os.mkdir(crop_img_folder)
    else:
        shutil.rmtree(crop_img_folder)
        os.mkdir(crop_img_folder)

    if not os.path.isdir(rename_crop_img_folder):
        os.mkdir(rename_crop_img_folder)
    else:
        shutil.rmtree(rename_crop_img_folder)
        os.mkdir(rename_crop_img_folder)

    if not os.path.isdir(final_img_folder):
        os.mkdir(final_img_folder)

    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + craft_opt.trained_model + ')')
    if craft_opt.cuda:
        net.load_state_dict(copyStateDict(torch.load(craft_opt.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(craft_opt.trained_model, map_location='cpu')))

    if craft_opt.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()
    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print(image_path)
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, craft_opt.text_threshold, craft_opt.link_threshold, craft_opt.low_text, craft_opt.cuda, craft_opt.poly, craft_opt, None)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder, cimg_dirname=crop_img_folder)

        
    
    lang_transform_test = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    lang_classifi_dataset = lang_classifi(crop_img_folder,lang_transform_test)
    lang_classifi_loader = torch.utils.data.DataLoader(
        lang_classifi_dataset, batch_size=1,
        shuffle=False,  # 'True' to check training progress with validation function.
        num_workers=8,
        pin_memory=True)
    print("languages loder len : ", len(lang_classifi_loader))
    mode = 'efficientnet-b3'
    lang_classifi_net = EfficientNet.from_name(mode, num_classes=2)
    print("languages model...")
    state_dict = torch.load('./model/classifi_97.29.pth', map_location='cpu')
    
    new_state_dict = OrderedDict()
    for k, v in state_dict['net'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    lang_classifi_net.load_state_dict(new_state_dict)
    
    
    lang_classifi_net = lang_classifi_net.to(device)
    lang_classifi_net = torch.nn.DataParallel(lang_classifi_net)
    lang_classifi_net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, img_path) in enumerate(lang_classifi_loader):
            inputs = inputs.to(device)
            outputs = lang_classifi_net(inputs)
            outputs,indices = outputs.max(1)
            shutil.copy(img_path[0], rename_crop_img_folder + img_path[0].split('/')[-1].split('.jpg')[0]+'_'+str(int(indices[0]))+'.jpg')

    print("elapsed time : {}s".format(time.time() - t))
    
    
    ###################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model', type=str, default='./model/kor_reg_97.42.pth', help='korean pre-train model')
    parser.add_argument('--experiment_name', help='Where to store logs and models')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=200, help='Interval between each validation')
    parser.add_argument('--continue_model', default='', help="path to model to continue training")
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='data',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='1',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=50, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_false', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', required=False, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='VGG', required=False, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='None', required=False, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', required=False, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--local_rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    kr_opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    
    """ Seed and GPU setting """
    random.seed(kr_opt.manualSeed)
    np.random.seed(kr_opt.manualSeed)
    torch.manual_seed(kr_opt.manualSeed)
    torch.cuda.manual_seed(kr_opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    kr_opt.num_gpu = torch.cuda.device_count()
    kr_opt.gpu = kr_opt.local_rank
    torch.cuda.set_device(kr_opt.gpu)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model', type=str, default='./model/TPS-ResNet-BiLSTM-Attn.pth', help='english pre-train model')
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='data',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='1',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=50, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', required=False, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', required=False,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling',  type=str, default='BiLSTM', required=False, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn' ,required=False, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    en_opt = parser.parse_args()
    
    en_opt.num_gpu = torch.cuda.device_count()
    en_opt.gpu = kr_opt.local_rank


    train(kr_opt, en_opt, craft_opt.test_folder, rename_crop_img_folder, final_img_folder)
    
