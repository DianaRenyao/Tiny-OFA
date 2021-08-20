import argparse
import cv2
import numpy as np

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

def select_device(device='', apex=False, batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

# 改动了权重数据的量化
def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k  - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply

class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    # 符号位 占一位
    self.uniform_q = uniform_quantize(k=w_bit - 1) 

  def forward(self, x):
    if self.w_bit == 32:
      weight = torch.tanh(x)
      weight_q = weight / torch.max(torch.abs(weight))
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = (self.uniform_q(x / E) + 1) / 2 * E
    else:
      weight = torch.tanh(x)
      weight = weight / torch.max(torch.abs(weight))
      # 想量化到带符号的 k bit
      weight_q = self.uniform_q(weight)
    return weight_q 

class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = torch.clamp(x, 0, 6)
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q

def conv2d_Q_fn(w_bit):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight)
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q

def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny

class YOLOLayer(nn.Module):
    def __init__(self, anchors):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.no = 6  # number of outputs
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
    def forward(self, p, img_size):
        
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride  # 原始像素尺度
            
            torch.sigmoid_(io[..., 4:])     
            
            return io.view(bs, -1, self.no), p

class TestNetQua(nn.Module):
    def __init__(self):
        super(TestNetQua, self).__init__()
        W_BIT = 8
        A_BIT = 8
        conv2d_q = conv2d_Q_fn(W_BIT)
        # act_q = activation_quantize_fn(4)

        self.layers = nn.Sequential(
            conv2d_q(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            activation_quantize_fn(A_BIT),
            nn.MaxPool2d(2, stride=2),

            conv2d_q(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),


            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),


            # nn.Conv2d(256, 18, kernel_size=1, stride=1, padding=0)
            conv2d_q(64, 36, kernel_size=1, stride=1, padding=0)
            
        )
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x = self.layers(x)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x  

def inference(weights=None,
         batch_size=16,
         img_size=416,
         model=None,
         path = None):
    # Initialize/load model and set device
    if model is None:
        device = select_device(opt.device, batch_size=batch_size)

        # Initialize model
        model = TestNetQua().to(device)

        model.nc = 1
        model.arc = 'default'

        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    # load image
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    
    fig = plt.figure()
    plt.subplot()
    
    interp = cv2.INTER_LINEAR  # LINEAR for training, AREA for testing
    img = cv2.resize(img, (img_size, img_size // 2), interpolation=interp)
    plt.imshow(img)
    img = np.expand_dims(np.transpose(img,[2,0,1]),axis=0).copy()
    img= torch.from_numpy(img)

    model.eval()

    with torch.no_grad():
        img = img.to(device).float()/255.0 # uint8 to float32, 0 - 255 to 0.0 - 1.0
        # run the model
        inference_out, training_out = model(img)
        inference_out = inference_out.view(inference_out.shape[0], 6, -1)
        print(inference_out.shape)
        inference_out_t = torch.zeros_like(inference_out[:, 0, :])
        for i in range(inference_out.shape[1]):
          inference_out_t += inference_out[:, i, :]
        inference_out_t = inference_out_t.view(inference_out_t.shape[0], -1, 6) / 6
        print(inference_out_t.shape)
        FloatTensor = torch.cuda.FloatTensor if inference_out_t.is_cuda else torch.FloatTensor
        n = inference_out_t.size(0)
        p_boxes = FloatTensor(n, 4)
        pred_boxes = inference_out_t[...,:4]
        pred_conf = inference_out_t[...,4]
        for i in range(n):
            _, index = pred_conf[i].max(0) # 返回每一列最大值组成的数据
            p_boxes[i] = pred_boxes[i][index]
        print(p_boxes.shape)
        print(p_boxes)
        img = img.cpu().numpy()
        p_boxes = p_boxes.cpu().numpy()
        print(p_boxes)
        bs, channel, h, w = img.shape
        # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
        y = torch.zeros_like(p_boxes) if isinstance(p_boxes, torch.Tensor) else np.zeros_like(p_boxes)
        y[:, 0] = p_boxes[:, 0] - p_boxes[:, 2] / 2
        y[:, 1] = p_boxes[:, 1] - p_boxes[:, 3] / 2
        y[:, 2] = p_boxes[:, 0] + p_boxes[:, 2] / 2
        y[:, 3] = p_boxes[:, 1] + p_boxes[:, 3] / 2
        y=y.T
        print(y)
        
        plt.plot(y[[0,2,2,0,0]],y[[1,1,3,3,1]],'.-')
        plt.axis('off')
        fig.savefig("test.png")
        plt.close()
    return 
           

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', type=str, default='weights/test_best.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--path',type=str,default= "../data/data_test/boat1/000001.jpg")
    opt = parser.parse_args()
    print(opt)

    # Test
    inference(opt.weights,
            opt.batch_size,
            opt.img_size,
            path = opt.path)
