from utils.load_dataset import *
from utils.custom_utils import *

def get_dataset(cfg_dataset):
    if cfg_dataset['train']['virtual'] != "" and cfg_dataset['train']['real'] == "":
        print("Load only virtual dataset")
        train_df = load_data(cfg_dataset['train']['virtual'])
    elif cfg_dataset['train']['virtual'] == "" and cfg_dataset['train']['real'] != "":
        print("Load only real dataset")
        path = f"{cfg_dataset
        train_df = load_data(cfg_dataset['train']['real'])
    else:
        print("Load real and virtual dataset!")
    test_df = load_data(cfg_dataset['test'])
    if cfg_dataset['validation']['valid'] == "":
        path_valid = cfg_dataset['root_train']
        train, valid = split_train_valid(train_df, cfg_dataset['percentage_val'])
    else:
        train = train_df
        path_valid = cfg_dataset['root_valid']
        valid = load_data(cfg_dataset['validation']['valid'])

    train_dt = FallenPeople(train, cfg_dataset['root_train'], FallenPeople.train_transform())
    valid_dt = FallenPeople(valid, path_valid, FallenPeople.valid_test_transform())
    test_dt = FallenPeople(test_df, cfg_dataset['root_test'], FallenPeople.valid_test_transform())
    return train, train_dt, valid_dt, test_dt


def get_model(cfg):    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=cfg['pretrained'])
    num_classes = 1 + cfg['num_class'] # num_class + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained model's head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if cfg['device'] == "cuda":
        device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg['device'])
    model.to(device)
    return model, device

def prep_train(cfg_dataset, cfg_train):

    train, train_dataset, valid_dataset, test_dataset = get_dataset(cfg_dataset)
    
    def collate_fn(batch):
        return tuple(zip(*batch))

    if cfg_train['sampler']:
        weights = make_weights(train)
        sampler = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights), int(len(weights)))

        train_data_loader = DataLoader(
          train_dataset,
          batch_size=cfg_train['batch_size'],
          num_workers = cfg_train['num_workers'],
          sampler = sampler,
          collate_fn=collate_fn          
        )
    else:
        train_data_loader = DataLoader(
          train_dataset,
          batch_size=cfg_train['batch_size'],
          num_workers = cfg_train['num_workers'],
          shuffle = False,
          collate_fn=collate_fn
        )

    valid_data_loader = DataLoader(
      valid_dataset,
      batch_size=cfg_train['batch_size'],
      num_workers = cfg_train['num_workers'],
      shuffle=False,
      collate_fn=collate_fn
    )

    test_data_loader = DataLoader(
      test_dataset,
      batch_size=cfg_train['batch_size'],
      num_workers = cfg_train['num_workers'],
      shuffle=False,
      collate_fn=collate_fn
    )
    return train_data_loader, valid_data_loader, test_data_loader

def train(args):

    # Opening configuration file
    with open(args.conf_file, 'r') as stream:
        try:
            cfg_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print("Reading the configuration from YALM file...")
    # Retrieving cfg
    cfg_train = cfg_file['train']
    cfg_model = cfg_file['model']
    cfg_dataset = cfg_file['dataset']
    print("Loading dataset...")
    _,_,valid_dataset, test_dataset = get_dataset(cfg_dataset)
    print("Loading model...")
    model, device = get_model(cfg_model)
    print(f"Preparation to train the model in {device}...")
    train_data_loader, valid_data_loader, test_data_loader = prep_train(cfg_dataset, cfg_train)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg_train['lr'], momentum=cfg_train['momentum'], weight_decay=cfg_train['weight_decay'])

    if cfg_train['lr_scheduler']['flag'] == False:
        lr_scheduler = None
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg_train['lr_scheduler']['lr_step_size'],
                                                                       gamma=cfg_train['lr_scheduler']['lr_gamma'])            
    itr = 1
    total_train_loss = []
    total_valid_loss = []
    losses_value = 0.0
    f_log = open(cfg_train['log_file'], "w")
    early_stopping = EarlyStopping(patience=3, verbose=True, path="models/checkpoint_train_virtual.pth")
    print("Training...\n")
    for epoch in range(cfg_train['epochs']):
        start_time = time.time()
        # train ------------------------------
        running_corrects = 0
        model.train()
        train_loss = []
         
        for images, targets in tqdm(train_data_loader, desc='Training...'):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses_value = losses.item()
            train_loss.append(losses_value)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            #f_log.write(f"Epoch: {epoch+1}, Batch: {itr}, Loss: {losses_value}\n")
            #pbar.set_description(f"Epoch: {epoch+1}, Batch: {itr}, Loss: {losses_value}")
            itr += 1
        epoch_train_loss = np.mean(train_loss)
        total_train_loss.append(epoch_train_loss)
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        # valid -------------------------------------
        with torch.no_grad():
            valid_loss = []

        for images, targets in valid_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            valid_loss.append(loss_value)
        epoch_valid_loss = np.mean(valid_loss)
        total_valid_loss.append(epoch_valid_loss)  
        # print losses ------------------------------
        f_log.write("\nVALIDATION PHASE: ")
        print("\nVALIDATION PHASE: ")
        f_log.write(f"Epoch Completed: {epoch+1}/{num_epochs}, Time: {time.time()-start_time}, "
              f"Train Loss: {epoch_train_loss}, Valid Loss: {epoch_valid_loss}\n")  
        print(f"Epoch Completed: {epoch+1}/{num_epochs}, Time: {time.time()-start_time}, "
              f"Train Loss: {epoch_train_loss}, Valid Loss: {epoch_valid_loss}")

        #mAP and accuracy over validation
        sys.stdout = f_log
        evaluate(model, valid_data_loader, device=device)
        classifier_performance(valid_dataset, model, device)
        sys.stdout = original_stdout
        
        #mAP and accuracy over testing every 2 epochs
        if (epoch+1) % 2 == 0:
            f_log.write(f"\nTESTING PHASE EPOCH {epoch+1}: ")  
            print(f"\nTESTING PHASE EPOCH {epoch+1}: ")
            sys.stdout = f_log
            evaluate(model, test_data_loader, device=device)
            classifier_performance(test_dataset, model, device)
            sys.stdout = original_stdout
            
        early_stopping(epoch_valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        else:
            print("Saving checkpoint...")
            early_stopping.save_checkpoint(epoch_valid_loss, model)
            
    f_log.close()
    print("Training completed!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--conf_file', required=True, help="YAML config file path")

    args = parser.parse_args()

    train(args)
