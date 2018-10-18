import os
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset


def get_dadaloder(root_dir):
    class AgriDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.files = os.listdir(root_dir)

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, self.files[idx])
            image = Image.open(img_name)
            sample = {'img_name': self.files[idx], 'image': image}

            if self.transform:
                sample['image'] = self.transform(sample['image'])

            return sample

    transform_test = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = AgriDataset(root_dir=root_dir,
                          transform=transform_test
                          )
    dataloder = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=0)
    return dataloder
    pass


def test(model, dataloder):
    image_id, disease_class = [], []
    model.train(False)
    for data in dataloder:
        inputs, img_name = data['image'], data['img_name']
        inputs = Variable(inputs.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        image_id += img_name
        disease_class += list(preds.cpu().numpy())
    keys = ['image_id', 'disease_class']
    result = []
    temp = [int(x) for x in disease_class]
    for i in range(0, len(image_id)):
        values = [image_id[i], temp[i]]
        dictionary = dict(zip(keys, values))
        result.append(dictionary)
    return result
    pass


if __name__ == '__main__':
    model = torch.load('model.pkl')
    path = 'C:/Users/styu/Workspace/ai_challenger_pdr2018_testA_20180905/AgriculturalDisease_testA/images'
    dataloder = get_dadaloder(path)
    result = test(model, dataloder)
