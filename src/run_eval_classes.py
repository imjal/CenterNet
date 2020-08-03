import _init_paths
from utils.coco import COCO
import pickle
from pycocotools.cocoeval import COCOeval

save_dict = pickle.load(open('/data2/jl5/centdt_exp/ctdet_stream/downsample3_dla34_pt_driving1002/raw_save_dict.pkl', 'rb'))

gt_coco = COCO(save_dict['gt_dict'])
dt_coco = gt_coco.loadRes(save_dict['dt_dict'])
E = COCOeval(gt_coco, dt_coco, iouType='bbox')
E.params.catIds = [0]
E.evaluate()
E.accumulate()
E.summarize()