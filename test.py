# import torchvision
#
# import src.dataset
#
# batch_size = 64
#
# helper = src.dataset.DatasetHelper()
#
# # path = helper.download_dataset(folder_path="data")
# # classes, id_dict = helper.get_classes(path)
# path = '/media/abhijit/DATA/Development/TSAI/EVA/MobileNetDS/Dataset'
#
# train_imgs, train_labels, test_imgs, test_labels = helper.get_train_test_data(path)
#
import torchdata as td
#
# batch_size = 128

# torch.backends.cudnn.benchmark = True
import src.dataset
from src.preprocessing import preprochelper

helper = src.dataset.DatasetHelper()

# imagnet mean and std dev
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# path = helper.download_dataset(folder_path="data")
# classes, id_dict = helper.get_classes(path)
path = '/media/abhijit/Windows/MobileNetDS/Dataset'

classes = ['Flying Birds', 'Large QuadCopters', 'Small QuadCopters', 'Winged Drones']

train_imgs, train_labels, test_imgs, test_labels = helper.get_train_test_data(path)

train_transforms, test_transforms = preprochelper.PreprocHelper.getmnvetv2traintesttransforms(mean, std)

train_dataset = src.dataset.GenericDataset(train_imgs, train_labels, train_transforms, False).cache(
    td.modifiers.FromIndex(0, td.cachers.Memory()))
# .cache(td.modifiers.FromIndex(10000, td.cachers.Pickle("./cache")))


print(len(train_imgs))
print(len(test_imgs))
