import torch
import imgproc
import file_utils
from tqdm import tqdm
from text_recognition import ltr_utils
import opt
import time 


def test_net(model=None, mapper=None, spaces=None, load_from=None, save_to=None):
    device = torch.device('cuda:0,1,2,3' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad(): # 학습된 모델 사용 → 결과 확인
        image_name_nums = []
        res = []
        img_lists, _, _, name_list = file_utils.get_files(load_from)
        for name in name_list: image_name_nums.append(name.split('_')[0])
        for k, in_path in tqdm(enumerate(img_lists)):
            
            time.sleep(0.1)
            # data pre-processing for passing net
            image = imgproc.loadImage(in_path)
            image = imgproc.cvtColorGray(image)
            image = imgproc.tranformToTensor(image, opt.RECOG_TRAIN_SIZE).unsqueeze(0)
            image = image.to(device)
            y = model(image)
            _, pred = torch.max(y.data, 1)
            res.append(mapper[0][pred])
        print("-- line text recognition")
        # method for saving result, MODE: file | stdout | all
        ltr_utils.display_stdout(chars=res, space=spaces, img_name=image_name_nums, MODE='all', save_to=save_to)
