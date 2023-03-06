"""
This project name is WORD(Webtoon Object Recognition and Detection)
WORD consists of object detection (detection of speech bubble, cut) and OCR(detection and recognition of line text)
Yon can also meet results of translation with papago API of naver corp if you want.


Future Science Technology Internship
Ajou University.
Writer: Han Kim


referenced paper :
            Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
            CRAFT : Character Region Awareness for Text Detection
"""

# GPU 설정
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import os
import time
from tqdm import tqdm
import file_utils
import imgproc
import net_utils
import opt
from cut_off import cut_off_image as cut_off
from object_detection.bubble import test_net as bubble_detect
from object_detection.cut import test_opencv as cut_detect
from text_detection.line_text import test as line_text_detect
from text_recognition.line_text import test_net as line_text_recognize
from text_recognition.ltr_utils import gen_txt_to_image as gen_text_to_image
from translation.papago import translation as papago_translation
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# parser
parser = argparse.ArgumentParser(
    description="WORD(Webtoon Object Recognition and Detection)"
)

parser.add_argument(
    "--object_detector",
    default="./weights/Speech-Bubble-Detector.pth",
    type=str,
    help="pretrained",
)
parser.add_argument(
    "--text_detector",
    default="./weights/Line-Text-Detector.pth",
    type=str,
    help="pretrained",
)
parser.add_argument(
    "--text_recognizer",
    default="./weights/Line-Text-Recognizer.pth",
    type=str,
    help="pretrained",
)
parser.add_argument(
    "--object", action="store_true", default=True, help="enable objext detection"
)
parser.add_argument("--ocr", action="store_true", default=False, help="enable OCR")
parser.add_argument(
    "--papago",
    action="store_true",
    default=False,
    help="enable English translation with papago",
)
parser.add_argument(
    "--type", default="white", type=str, help="background type: white, black, classic"
)
parser.add_argument(
    "--cls", default=0.995, type=float, help="bubble prediction threshold"
)
parser.add_argument(
    "--box_size", default=7000, type=int, help="cut size filtering threshold"
)
parser.add_argument(
    "--large_scale",
    action="store_true",
    default=False,
    help="demo image is large scale",
)
parser.add_argument(
    "--ratio", default=2.0, type=float, help="height & width ratio of demo image"
)
parser.add_argument(
    "--demo_folder", default="./data/", type=str, help="folder path to demo images"
)
parser.add_argument(
    "--cuda", action="store_true", default=False, help="use cuda for inference"
)

args = parser.parse_args()

""" For test images in a folder """
image_list, _, _, name_list = file_utils.get_files(args.demo_folder)

file_utils.rm_all_dir(dir="./result/")  # clean directories for next test
file_utils.mkdir(
    dir=[
        "./result/",
        "./result/bubbles/",
        "./result/cuts/",
        "./result/demo/",
        "./result/chars/",
    ]
)

# load net
models = net_utils.load_net(args)  # initialize and load weights

spaces = []  # text recognition spacing word
text_warp_items = []  # text to warp bubble image
demos = []  # all demo image storage
t = time.time()

cnt = 0

# load data
for k, image_path in tqdm(enumerate(image_list)):
    time.sleep(0.1)
    print(
        "TEST IMAGE ({:d}/{:d}): INPUT PATH:[{:s}]".format(
            k + 1, len(image_list), image_path
        ),
        end="\r",
    )

    img = imgproc.loadImage(image_path)

    # scale
    if (
        args.large_scale
    ):  # cut off large scale images into several pieces (width : height = 1 : 2)
        images = cut_off(image=img, name=name_list[k], ratio=args.ratio)

    else:  # general scale case
        images = imgproc.uniformizeShape(image=img)

    for img in tqdm(images):  # image fragments divided from cut_off.py
        
        time.sleep(0.1)

        cnt += 1
        str_cnt = file_utils.resultNameNumbering(
            origin=cnt, digit=1000
        )  # ex: 1 -> 0001, 2 -> 0002
        
        img_blob, img_scale = imgproc.getImageBlob(img)
        f_RCNN_param = [img_blob, img_scale, opt.LABEL]  # LABEL: speech bubble
        
        # available
        
        demo, image, bubbles, dets_bubbles = bubble_detect(
            model=models["bubble_detector"],
            image=img,
            params=f_RCNN_param,
            cls=args.cls,
            bg=args.type,
        )
        
        demo, cuts = cut_detect(
            image=img, demo=demo, bg=args.type, size=args.box_size
        )

        demo, space, warps = line_text_detect(
            model=models["text_detector"],
            demo=demo,
            bubbles=imgproc.cpImage(bubbles),
            dets=dets_bubbles,
            img_name=str_cnt,
            save_to="./result/chars/",
        )

        spaces += space  # add temporarily space in an image to total spaces storage
        text_warp_items += warps  # add temporarily text & bubble image to text_warp_items which is kind of storage.
        demos.append(demo)  # demo image is stored demos storage
        

        # save segmented object(bubble & cut)
      
        file_utils.saveAllImages(
            save_to="./result/bubbles/", imgs=bubbles, index1=str_cnt, ext=".png"
        )
        print("---- bubble detection is done")
        
        file_utils.saveAllImages(
            save_to="./result/cuts/", imgs=cuts, index1=str_cnt, ext=".png"
        )

#------------------------------------------------------------#

if args.ocr:  # ocr

    # save spaces word information
    file_utils.saveText(save_to="./result/", text=spaces, name="spaces")

    # mapping one-hot-vectors to hangul labels
    label_mapper = file_utils.makeLabelMapper(
        load_from="./text_recognition/labels-2213.txt"
    )

    # load spacing word information
    spaces, _ = file_utils.loadSpacingWordInfo(load_from="./result/spaces.txt")

    x = time.time()
    print("\n[processing ocr.. please wait..]", end=" ")
    line_text_recognize(
        model=models["text_recognizer"],
        mapper=label_mapper,
        spaces=spaces,
        load_from="./result/chars/",
        save_to="./result/ocr.txt",
    )
    print("[ocr time: {:.6f} sec]".format(time.time() - x))
    
#------------------------------------------------------------#

if args.papago:  # translation

    print("[translating korean to English..]")
    papago_translation(
        load_from="./result/ocr.txt",
        save_to="./result/english_ocr.txt",
        id=opt.PAPAGO_ID,
        pw=opt.PAPAGO_PW,
    )
    gen_text_to_image(load_from="./result/english_ocr.txt", warp_item=text_warp_items)

# save final demo images

file_utils.saveAllImages(save_to="./result/demo/", imgs=demos, ext=".png")

print("[elapsed time : {:.6f} sec]".format(time.time() - t))
