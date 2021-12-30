# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def compute_loss(p, targets, model):  # predictions, targets, model
    """
    e.g. batch size = 16
    p:       prediction : p[0] p[1] p[2]  torch.Size([16, 3, 80, 80, 25])  40 40    20 20
    targets: (71, 6)
      
      """


    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device) # init losses
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    """ 
        From the #link what targets look like
            you should know that
            targets : gt in the current batch.
     """
    
    
    
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))  # weight=model.class_weights)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))  #todo What is pos_weight

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0) # cp:class positive   cn: class negative

    # Focal loss #todo focal loss
    g = h['fl_gamma']  # focal loss gamma # fl:focal loss
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g) # their reductino are mean. Check them by typing BCEcls.__dict__

    # Losses
    nt = 0       # number of targets
    no = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.3, 0.1, 0.03]  # P3-P7
    for i, pi in enumerate(p):     # layer index, layer predictions # pi: prediction i-layer
        b, a, gj, gi = indices[i]  # image-idx, anchor-idx, gridy, gridx for the current predictions across one batch images
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj has the same shape with the prediction i-layer

        n = b.shape[0]  # number of targets across 16 images(if total_batchsize = 16), all gt 
        if n:           # if obj exits in the current batch
            nt += n     # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets 
            """ 
            Recall that we only for loop prediction layer here.
                for each pred layer, we extract all objs
                such that
                pi: torch.Size([16, 3, 80, 80, 25])
                ps: torch.Size([99, 25])

                meaning that
                for the current pred layer, from 16 images we use 3 anchors and extract
                99 objs each of which can be represented by a 1x25 vector(5 + 20 in VOC).

                ps: preds
                    the below is only for boosting understanding.
                    ps = pi[image12, anchor1, 28, 49]
                            image12, anchor1, 38, 53
                            image32, anchor0, 13, 23
                            ...
                            image49, anchor2, 44, 55
            
             """


            #! Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i] # predict based on the current the given normalized anchor
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target) # tbox has index i meaning that we are doing cross-stage-matching
            lbox += (1.0 - iou).mean()       # iou: how much they match. 1-iou: how much they don't match for the current batch #todo why +=
                                             # Now I know to what extent the current batch don't match their targets.
                                             # Notice that here it's mean rather than /batch_size. mean refers to mean of the detections. This is in accordance with the lcls, lobj computed by BCEobj and BCEcls
                                            
            #! Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio # tobj torch.Size([16, 3, 80, 80])

            #! Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # t has a same shape with ps. t is initialized with all class_negative(0).
                t[range(n), tcls[i]] = cp     # all targets(e.g. from 0-th to 98-th target) # yolov5.jpg  
                lcls += BCEcls(ps[:, 5:], t)  # Focal loss(pred, gt) if g > 0 #todo why +=
                """ 
                visualize t
                    cv2.imwrite("target_class_vis.jpg", (255*temp_t.cpu().numpy()).astype("uint8"))
                    
                    tcls: target class
                    
                    ps[:, 5:] torch.Size([99, 20])
                    t         torch.Size([99, 20]) 
                        
                 """

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling # usually, if we do scaling, the dst is denominator. the src is the numberator.
    lbox *= h['box'] * s
    lobj *= h['obj'] 
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach() # here we scale the loss


""" 

 """
def build_targets(p, targets, model):
    """ 
        Build targets for compute_loss(), input targets(image,class,cx, cy,w,h)
        p[0]  torch.Size([16, 3, 80, 80, 25])
        p[1]  torch.Size([16, 3, 40, 40, 25])
        p[2]  torch.Size([16, 3, 20, 20, 25])

        targets: for 16 images  # cxcywh
    
     """
    #region #! predefined variable
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], [] #TODO
    fm_sz_gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
    """ 
        before:
        targets.shape  (71, 6)
        
        ---> repeat and cat with anchors

        after:
        targets.shape  (3, 71, 7) 

        intra_batch_img_idx   class     cx       cy     w         h         anchor_idx
        tensor([[[ 0.00000,  8.00000,  0.46263,  ...,  0.10852,  0.22772,  0.00000],
                [ 0.00000,  8.00000,  0.41726,  ...,  0.09607,  0.18680,  0.00000],
                ...,
                [15.00000, 18.00000,  0.73072,  ...,  0.33299,  0.07502,  0.00000],
                [15.00000,  5.00000,  0.27723,  ...,  0.12441,  0.09843,  0.00000],

                -------------------------------------------------------------------
                [[ 0.00000,  8.00000,  0.46263,  ...,  0.10852,  0.22772,  1.00000],
                [ 0.00000,  8.00000,  0.41726,  ...,  0.09607,  0.18680,  1.00000],
                ...,
                [15.00000, 18.00000,  0.73072,  ...,  0.33299,  0.07502,  1.00000],
                [15.00000,  5.00000,  0.27723,  ...,  0.12441,  0.09843,  1.00000],

                ------------------------------------------------------------------
                [[ 0.00000,  8.00000,  0.46263,  ...,  0.10852,  0.22772,  2.00000],
                [ 0.00000,  8.00000,  0.41726,  ...,  0.09607,  0.18680,  2.00000],
                ...,
                [15.00000, 18.00000,  0.73072,  ...,  0.33299,  0.07502,  2.00000],
                [15.00000,  5.00000,  0.27723,  ...,  0.12441,  0.09843,  2.00000],

            from the above, we know for a batch(16 imgs), their targets info.

     """

    #endregion predefined variable

    #region #! The positve-sample-extended direction (offset)
    ext_bias = grid_ct = 0.5
    offsets = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],      # j,k,l,m # 4-neighbors
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm # 8-neighbors
                        ], device=targets.device).float() * ext_bias  # offsets
    
    #endregion The positve-sample-extended direction 

    
    # process each detection layer
    for i in range(det.nl):
        anchors = det.anchors[i] # current 3 anchors already divided by the stride
        fm_sz_gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

        # Match targets to anchors
        t_on_fm_sz = targets * fm_sz_gain # convert cxcywh from normalized (0-1) to featuremap scale
        
        """ 
            gain
                                 cx   cy   w    h
               tensor([ 1.,  1., 80., 80., 80., 80.,  1.], device='cuda:0')

            targets(intra_batch_img_idx, class, x, y, w, h, anchor_idx)

         """
        """ 
            So far both targets and anchors are featuremap scale. We measure the targets shape against
            the anchors shape based on anchor_t. The disqualified ratio indicates that the anchors doesn't fit
            in the target. Therefore prediction wouldn't be done on this anchor.
         """
        if nt: # if targets exists on the batch
            wh_anchor_r = t_on_fm_sz[:, :, 4:6] / anchors[:, None]  # wh ratio
            good_rs = torch.max(wh_anchor_r, 1. / wh_anchor_r).max(2)[0] < model.hyp['anchor_t']  # compare # the 1-st max is for choosing the one bigger than 1. The 2nd is for choosing the worse one
            
            """
                given a scale(e.g. 80x80) for the current batch, we have 16 images as one batch which has 58 targets.
                
                e.g.
                    t: (58, 7)
                     img-id      
                    [ 6.00000,  2.00000, 41.99864, 31.49398,  4.67210,  2.57323,  0.00000],
                    [ 6.00000,  2.00000, 44.01248, 30.52902,  4.83321,  3.21654,  0.00000],
                    ...
                    [ 1.00000, 15.00000, 53.66125, 23.93892,  6.49017,  4.77133,  1.00000]

                targets are on featuremap-scale. 

                the 1-st max --> use the bigger-than-1 ratio to measure against.
                the 2-nd max --> choose the worse one among the hhww ratios.

                measure all the worse against the thr
                And get the indexes returned such that we know which prediction matches which anchor on which scale.

             """
            selected_t_on_fm_sz = t_on_fm_sz[good_rs]  # filter # j indicates the matching # 

            # Offsets
            t_cxcy = selected_t_on_fm_sz[:, 2:4]
            t_cxcy_inv = fm_sz_gain[[2, 3]] - t_cxcy  # inverse
            left, top = ((t_cxcy % 1. < grid_ct) & (t_cxcy > 1.)).T  # no need to extend at the boundary
            right, bottom = ((t_cxcy_inv % 1. < grid_ct) & (t_cxcy_inv > 1.)).T
            ext_filter = torch.stack((torch.ones_like(left), left, top, right, bottom)) # (5, 33)
            
            # 5 offsets
            selected_t_on_fm_sz = selected_t_on_fm_sz.repeat((5, 1, 1))[ext_filter]
            ext_offsets = (torch.zeros_like(t_cxcy)[None] + offsets[:, None])[ext_filter]
        else:
            selected_t_on_fm_sz = targets[0]
            ext_offsets = 0

        b, cls = selected_t_on_fm_sz[:, :2].long().T  # image, class
        t_cxcy = selected_t_on_fm_sz[:, 2:4]  # gt cxcy
        t_wh = selected_t_on_fm_sz[:, 4:6]  # gt wh
        ext_t_cxcy = (t_cxcy - ext_offsets).long() # which grid the target lands on.(4-neighbors)
        t_col, t_row = ext_t_cxcy.T  # gt cxcy indices

        # Append
        a = selected_t_on_fm_sz[:, 6].long()  # anchor indices
        indices.append((b, a, t_row.clamp_(0, fm_sz_gain[3] - 1), t_col.clamp_(0, fm_sz_gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((t_cxcy - ext_t_cxcy, t_wh), 1)) # t_cxcy - t_ij is the offsets to be regressed. 
        anch.append(anchors[a])  # anchors
        tcls.append(cls)  # class
        """ 
        indices:
            given a target, we know in which intra-batch img it is, which anchor it matches, and its w and h. 
         """
        
    return tcls, tbox, indices, anch
