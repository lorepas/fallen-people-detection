from utils.custom_utils import *

class FallenPeople(Dataset):

    def __init__(self, dataframe, img_dir, transforms):
        super().__init__()
        self.image_ids = dataframe['img_path'].unique()
        self.df = dataframe
        self.img_dir = img_dir
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        records = self.df[self.df['img_path'] == image_id]

        image = cv2.imread(f'{self.img_dir}/{image_id}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        boxes = records[['x0', 'y0', 'x1', 'y1']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = records['label'].values
        labels = torch.as_tensor(labels, dtype = torch.int64)
        img_id = torch.tensor([idx])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.uint8)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target["image_id"] = img_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            augmented = self.transforms(image=image, bboxes = target['boxes'], class_labels = labels)
            if len(augmented['bboxes'])==0:
                convert_tensor = T.ToTensor()
                image = convert_tensor(image)
            else:
                image = augmented['image']
                target['boxes'] = torch.as_tensor(augmented['bboxes'], dtype=torch.float32)

        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)
    
    def train_transform():
        return A.Compose([
            A.RandomCrop(width=640, height=480, p=0.6),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.2),
                A.ToGray(p=0.5),
                A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            ], p=0.5),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=1024, min_visibility=0.4, label_fields=['class_labels']))
    
    def real_train_transform():
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.2),
                A.ToGray(p=0.5),
                A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            ], p=0.5),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))        
 
    def valid_test_transform():
        return A.Compose([
      ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
def load_training(target_txt):
    return pd.read_csv(target_txt, sep=',', header=None, names=['img_path', 'x0', 'y0', 'x1', 'y1', 'label'])

def get_values(dataset):
    return dataset['img_path'].nunique()

def get_labels_distribution(dataset):
    return dataset['label'].value_counts()

def load_data(d):
    path = os.path.join("datasets",d)
    return pd.read_csv(path, sep=',', header=None, names=['img_path', 'x0', 'y0', 'x1', 'y1', 'label'])

def split_train_valid(dataset, split_perc):
    images_ids = dataset['img_path'].unique()
    split_len = round(len(images_ids)*float(split_perc)) #80% -> train & 20% -> val
    train_ids = images_ids[:split_len]
    valid_ids = images_ids[split_len:]
    train = dataset[dataset['img_path'].isin(train_ids)]
    valid = dataset[dataset['img_path'].isin(valid_ids)]
    return train, valid

def make_weights(dataset):
    labels = list(dataset['label'].values)
    class_counts = dataset['label'].value_counts().tolist()
    num_samples = len(labels)
    class_weights = [class_counts[1]/class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[0] if l == 1 else class_weights[1] for l in labels]
    dataset_grouped = dataset.copy()
    dataset_grouped['weights'] = weights
    train_grouped = dataset_grouped.groupby(['img_path']).mean()
    return list(train_grouped['weights'].values)