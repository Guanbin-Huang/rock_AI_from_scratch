import cv2
import sys

from torch import is_tensor
sys.path.append(".")



def draw_bbox(xyrb_bbox, img):
    for item in xyrb_bbox:
        x,y,r,b = item[:4]
        img = cv2.rectangle(img, (int(x),int(y)), (int(r), int(b)), (0, 255, 0), 5)
    return img

def draw_line(image, p1, p2, color):
    p1, p2 = (int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1]))
    cv2.line(image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)

def show(img,i=0):
    import cv2
    # quickly show a tmp img
    cv2.imwrite(f"./temp_{i}.jpg",img)


def draw_skeleton(skeleton_mat, img):
    """
    skeleton:  这个函数只考虑单人单帧
    [max_num_persons, i_th frame, num_keypoits, x_y_score] e.g. 1 x 1 x 16 x 3
    """
    import torch
    import numpy as np
    if torch.is_tensor(skeleton_mat):
        skeleton_mat = skeleton_mat.numpy()


    coco_joint_sys = {
    }

    mpii_joint_sys = {
        0:[1],      # refer to coco_and_mpii.png
        1:[0,2],
        2:[1,6],
        3:[4,6],
        4:[3,5],
        5:[4],
        6:[2,3,7],
        7:[6,8,12,13],
        8:[7,9],
        9:[8],
        10:[11],
        11:[10,12],
        12:[7,11],
        13:[7,14],
        14:[13,15],
        15:[14]
    }

    color = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255)]
    for i_person in range(skeleton_mat.shape[0]):

        for j_kpt in range(skeleton_mat.shape[2]):
            x,y = skeleton_mat[i_person, 0, j_kpt, :2] # i_person, 当前帧
            cv2.circle(img, (int(x),int(y)), radius=3, color = color[i_person], thickness=3)
            
            # for k_adj_p in mpii_joint_sys[j_kpt]: # mpii_joint_sys里的第j_kpt个点的第k_adj_p 个邻接点(adj_p:ajacent point)
                # x_k, y_k = skeleton_mat[0,0,k_adj_p,:2]         # 第k个邻接点的x和y

                # draw_line(img, (x,y),(x_k,y_k), color = (0,0,255))


    return img


#region detection result filter_func

def keep_the_biggset_score(bboxes_for_single_frame):
    import torch
    """
    det_result[tensor]: nx7 xyrb, bbox_score, class_score, class
    """

    # if len(bboxes_for_single_frame) == 0:
    #     return 
    if bboxes_for_single_frame == None:
        return None

    rb = bboxes_for_single_frame[:,2:4]
    xy = bboxes_for_single_frame[:,0:2]
    wh = rb - xy
    area = wh[:,0] * wh[:,1]

    if area.numel() == 0:
        return None
    
    max_area_idx = area.argmax()

    return bboxes_for_single_frame[max_area_idx:max_area_idx+1]



#endregion detection result filter_func


def save_var_to_pkl(file_name,var):
    import pickle as pkl
    pkl.dump(var,open(f"{file_name}.pkl","wb"))

def load_var_from_pkl(file_name):
    import pickle as pkl
    return pkl.load(open(f"{file_name}.pkl","rb"))


def on_which_device(model):
    print(next(model.parameters()).device)


def denormalize(array):
    # 2d arr
    max_value = array.max().max()
    min_value = array.min().min()

    norm_arr = (array - min_value) /(max_value - min_value)
    return norm_arr




#region REPRODUCIBILITY
def one_func_set_all_random_seed(seed=0):
    # different random seeds
    import torch
    torch.manual_seed(seed)

    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    torch.use_deterministic_algorithms(True)

    # for dataloader
    g = torch.Generator()
    g.manual_seed(seed)

    return g

def seed_worker(worker_id):
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

_ = one_func_set_all_random_seed(3)


# DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     worker_init_fn=seed_worker
#     generator=g,
# )



#endregion REPRODUCIBILITY



