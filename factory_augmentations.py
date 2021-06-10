import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from PIL import Image
from config import ModelsAvailable

def get_normalize_parameter_by_model(model_enum:ModelsAvailable):
    prefix_name=model_enum.name[0:3]
    if model_enum==ModelsAvailable.vit_base_patch16_224_miil_in21k:
        mean=(0,0,0)
        std=(1, 1, 1)
        interpolation=Image.BILINEAR
    elif model_enum.name.split("_")[-1]=="overlap":
        mean=IMAGENET_DEFAULT_MEAN
        std=IMAGENET_DEFAULT_STD
        interpolation=Image.BILINEAR
        
    elif prefix_name==ModelsAvailable.vit_base_patch16_224_miil_in21k.name[0:3] \
        and model_enum!=ModelsAvailable.vit_base_patch16_224_miil_in21k:
        
        mean=(0.5,0.5,0.5)
        std=(0.5, 0.5, 0.5)
        interpolation=Image.BICUBIC

    else:
        mean=IMAGENET_DEFAULT_MEAN
        std=IMAGENET_DEFAULT_STD
        interpolation=Image.BICUBIC
    return mean,std,interpolation
def get_test_transforms(img_size:int,mean:tuple,std:tuple,interpolation):

    test_transforms = transforms.Compose([
        transforms.Resize((600, 600), interpolation),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        )
    ])
    return test_transforms

def get_train_classifier_transforms(img_size:int,mean:tuple,std:tuple,interpolation):

    transform=transforms.Compose([
                                        transforms.Resize((600, 600), interpolation),
                                        transforms.RandomCrop((img_size, img_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.AutoAugment(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
                        )                            
    return transform