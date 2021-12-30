import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import MixConv2d, CrossConv
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc                    # number of classes
        self.no = nc + 5                # number of outputs per anchor
        self.nl = len(anchors)          # number of detection layers
        self.na = len(anchors[0]) // 2  # number of type of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2) # [tensor([0.]), tensor([0.]), tensor([0.])]
        self.register_buffer('anchors', a)                     # shape(nl,na,2) such that we have anchors from different layers of differnt shape
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        """ 
            Two kinds of things should be saved:
                1. the things required to be optimized called parameters
                2. the parameters no need to be optimzed called buffer
                    - requires us to create tensor then register the tensor by using register_buffer()
                    - After rigstering, they would be saved into the OrderDict
                    - returned by model.buffers()
                    - Note that optim.step() only update the nn.parameters.

            Therefore, we register anchors and anchor_grid as the buffers without the need to update.
         """
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output 3 conv are created.
        """ 
        ModuleList(
            (0): Conv2d(128, 75, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(256, 75, kernel_size=(1, 1), stride=(1, 1))
            (2): Conv2d(512, 75, kernel_size=(1, 1), stride=(1, 1))
            )
         """

    def forward(self, x):               #! Note that in yolov5 inference time, the batch size is 1, which is different from the batchsize in training time.
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export    # a = a|b same as  a |= b   True exists then True
        for i in range(self.nl):        # for loop the prediction layer
            x[i] = self.m[i](x[i])      # Differnt conv is applied to different pred layer specified by i index.
            bs, _, ny, nx = x[i].shape  # x[i].shape:([1, 75, 32, 32]) in VOC the info of anchors are mingled, which will be slit.
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            """ 
                x[i].shape:([1, 75, 32, 32]) 
                    --> view --> # split the info in anchors
                x[i].shape:([1, 3, 25, 32, 32]) 
                    --> permute --> # put the 25 into the last position for the future convenience e.g. use x[...,a:b] to slice the tensor
                x[i].shape:([1, 3, 32, 32, 25]) 
             """

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]: #! ensuring the grid size equals the size of feature map
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)    #? where will it go x[0] ([1, 3, 32, 32, 25])

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module): # Parse yaml
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):        # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg                                       # model dict
        else:                                                     # is *.yaml
            import yaml                                           # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict # then we got all info from yaml file

        #! Define model by using yaml 
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)            # input channels
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc                                  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect): 
            s = 256  # 2x min stride # 
            """ 
            the max downsampling stride = 128
                meaning that given a 128 x 128 image, the final feature map is 1 x 1
                but given 256 x 256, the final feature map is 2 x 2
            

                
             """
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            """ 
            torch.zeros(1, ch, s, s)
                (1, 3, 256, 256)
                for x in self.forward( a tensor with the shape(1, 3, 256, 256)):
                    

             """

            m.anchors /= m.stride.view(-1, 1, 1) # normalized by stride 
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once #todo 
            # print('Strides: %s' % m.stride.tolist()) 

        # Init weights, biases
        initialize_weights(self) #todo
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs # dt: d_time
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run 
            y.append(x if m.i in self.save else None)  # save output # y is not used.
            """ 
                if m.i in self.save: # if the idx of the module is in the save_list
                    y.append(x)
                else:   
                    return None         
             """

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'number', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'] #! gd, gw are the coefficients to modify the strucuture depending on which yolov5.yaml is used.
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]                           # layers, savelist, ch out
    """ 
    view yolov5.jpg
        from the jpg, we can see sometimes for a given layer the inputs might come from different layers
        rather than just coming from its last layer. Therefore we need a save list to save the inputs/outputs when
        we do the inference.
     """

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  #!parse yaml  #  from, number, module, args
        # # for debug
        # if m == "C3":
        #     a = 1

        m = eval(m) if isinstance(m, str) else m                # eval strings 
        for j, a in enumerate(args): # j:j-th  a:args
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # if "a" can be parsed as a valid python expression, then do it. If not, nothing changed.

            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n               # n is the modified n given gd(depth_multiple) and gw(width_multiple)
        
        #region #! different types of layer
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0] #! c_in c_out
            """ 
                imgs/snapshot_structure.jpg
                e.g. layer0  for Focus
                    c1 = 3  c2 = 64

                     layer12 for Concat
                    c1 = -1 c2 = ch[6]
            
             """

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:
            
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2 #todo why 8
            """ 
                control the width of the network
                make_divisible is a function widely used in many tasks
                     - make_divisible(A,B) find an C bigger than A and also can be exactly divded
                        by B
                     ref: https://zhuanlan.zhihu.com/p/186014243#:~:text=2)-,%E6%8E%A7%E5%88%B6%E5%AE%BD%E5%BA%A6%E7%9A%84%E4%BB%A3%E7%A0%81,-%E5%9C%A8Yolo%20v5            

             """
            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]] 
            # the args is refactor from [c_out, kernel_size, stride etc] to [c_in, c_out ...]
            
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x if x < 0 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f if f < 0 else f + 1] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f if f < 0 else f + 1] // args[0] ** 2
        else:
            c2 = ch[f if f < 0 else f + 1]
        
        #endregion different types of layer

        # # debug
        # n = 3 # force n to be bigger than 1 for debugging

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module

        """ 
        m_ is the extended module. When we want a bigger network, we can just scale up some specific module,
            specifically, we repeat it a given times.

        e.g.
            [Focus(
            (conv): Con...LU()
            )
            ), Focus(
            (conv): Con...LU()
            )
            ), Focus(
            (conv): Con...LU()
            )
            )]
        
         """

        # # debug
        # if "main" in str(m):
        #     abc = 1 # after running it again, nothing happen. Just a way to avoid sth wrong happen.


        t = str(m)[8:-2].replace('__main__.', '')  # ensure the module type is correct
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        """ 
            So far, we've known, given a module, to what index it attaches(layer-idx), c_in source, module type and num of parameters
            in this module
         """
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist #todo x % i
        """ 
            refer to imgs/snapshot_structure.jpg
            f can be only one number or a list, meaning that the c_in comes from several layers.
            We want to for loop each c_in. 
            So the first step is to evaluate it is list or an int.
            
            The above line of code is mixing up
            Let's decouple it 
            for x in a_f_list:
                if x != -1:
                    x % i
                else:
                    pass
            
            what is a_f_list:
                if f is an integer
                    make it as a list
                else:
                    return it directly
         """

        layers.append(m_) #! We keep appending the m_ into the layers and finally the layers make the whole network.
        ch.append(c2)     # always append the c2(aka c_out) of the current layer as the c1(aka c_in) of the next layer
    return nn.Sequential(*layers), sorted(save) # e.g. save : [6, 4, 14, 10, 17, 20, 23]


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
