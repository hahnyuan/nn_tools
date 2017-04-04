

class Resize_preprocess(object):
    """Rescales the input PIL.Image to the given 'size_w,size_h'.
    """

    def __init__(self, size_w,size_h):
        self.size = (size_w,size_h)

    def __call__(self, img):
        return img.resize(self.size)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
