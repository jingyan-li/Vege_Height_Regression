from torchvision import models

fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
