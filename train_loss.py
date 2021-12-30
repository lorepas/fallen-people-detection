from utils.load_dataset import *
from utils.custom_utils import *

def get_dataset(cfg_dataset):
    if cfg_dataset['ensamble']['flag'] == False:
        if cfg_dataset['train']['virtual'] != "" and cfg_dataset['train']['real'] == "":
            print("Load only virtual dataset")
            transforms = FallenPeople.train_transform()
            train_df = load_data(cfg_dataset['train']['virtual'])
        elif cfg_dataset['train']['virtual'] == "" and cfg_dataset['train']['real'] != "":
            print("Load only real dataset")
            transforms = FallenPeople.real_train_transform()
            train_df = load_data(cfg_dataset['train']['real'])
#         else:
#             print("Load real and virtual dataset!")
        test_df = load_data(cfg_dataset['test'])
        if cfg_dataset['validation']['valid'] == "":
            path_valid = cfg_dataset['root_train']
            train, valid = split_train_valid(train_df, cfg_dataset['validation']['percentage_val'])
        else:
            train = train_df
            path_valid = cfg_dataset['root_valid']
            valid = load_data(cfg_dataset['validation']['valid'])

        train_dt = FallenPeople(train, cfg_dataset['root_train'], transforms)
        valid_dt = FallenPeople(valid, path_valid, FallenPeople.valid_test_transform())
        test_dt = FallenPeople(test_df, cfg_dataset['root_test'], FallenPeople.valid_test_transform())
        return train, train_dt, valid_dt, test_dt
    else:
        print("Load virtual and real dataset!")
        train_df_vir = load_data(cfg_dataset['train']['virtual'])
        train_df_real = load_data(cfg_dataset['train']['real'])
        train_vir, valid_vir = split_train_valid(train_df_vir, cfg_dataset['validation']['percentage_val'])
        valid_real = load_data(cfg_dataset['validation']['valid'])
        test_df = load_data(cfg_dataset['test'])
        
        train_dt_vir = FallenPeople(train_vir, cfg_dataset['ensamble']['train_virtual'], FallenPeople.train_transform())
        train_dt_real = FallenPeople(train_df_real, cfg_dataset['ensamble']['train_real'], FallenPeople.real_train_transform())
        valid_dt_vir = FallenPeople(valid_vir, cfg_dataset['ensamble']['train_virtual'], FallenPeople.valid_test_transform())
        valid_dt_real = FallenPeople(valid_real, cfg_dataset['ensamble']['valid_real'], FallenPeople.valid_test_transform())
        test_dt = FallenPeople(test_df, cfg_dataset['root_test'], FallenPeople.valid_test_transform())
        return train_dt_vir, train_dt_real, valid_dt_vir, valid_dt_real, test_dt

def get_model(cfg):    
    print("Loading model pretrained on resnet50...")
    if cfg['trainable_layers'] == "":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=cfg['pretrained'],
                                                                    pretrained_backbone=cfg['pretrained_backbone'])
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=cfg['pretrained'], 
                                                                     pretrained_backbone=cfg['pretrained_backbone'],
                                                                     trainable_backbone_layers= cfg['trainable_layers'])
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

def load_model(cfg, path):
    print("Loading model pretrained over artificial dataset...")
    if cfg['device'] == "cuda":
        device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg['device'])
    # create a Faster R-CNN model without pre-trained
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=cfg['pretrained'],
                                                                 pretrained_backbone=cfg['pretrained_backbone'])
    num_classes = 1 + cfg['num_class'] # num_class + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained model's head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    return model, device

def prep_train_ensamble(cfg_dataset, cfg_train):

    train_v, train_r, valid_v, valid_r, test_dataset = get_dataset(cfg_dataset)
    #training
    train_len = max(len(train_v),len(train_r))
    len_vir_t = round(train_len*float(cfg_dataset['ensamble']['split'])) #split is for virtual over real
    len_real_t = train_len - len_vir_t
    train_vir = Subset(train_v, torch.randperm(len(train_v))[:len_vir_t])
    train_real = Subset(train_r, torch.randperm(len(train_r))[:len_real_t])
    #validation
    valid_len = max(len(valid_v),len(valid_r))
    len_vir_v = round(valid_len*float(cfg_dataset['ensamble']['split'])) #split is for virtual over real
    len_real_v = valid_len - len_vir_v
    valid_vir = Subset(valid_v, torch.randperm(len(valid_v))[:len_vir_v])
    valid_real = Subset(valid_r, torch.randperm(len(valid_r))[:len_real_v])
    
    train_dataset = ConcatDataset([train_vir,train_real])
    valid_dataset = ConcatDataset([valid_vir,valid_real])

    def collate_fn(batch):
        return tuple(zip(*batch))

    
    train_data_loader = DataLoader(
      train_dataset,
      batch_size=cfg_train['batch_size'],
      num_workers = cfg_train['num_workers'],
      shuffle=True,
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
    return valid_dataset, test_dataset, train_data_loader, valid_data_loader, test_data_loader

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
    return valid_dataset, test_dataset, train_data_loader, valid_data_loader, test_data_loader

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
    
    if cfg_train['pretrained_model'] == "":
        model, device = get_model(cfg_model)
    else:
        model, device = load_model(cfg_model, cfg_train['pretrained_model'])
    print(f"Preparation to train the model in {device}...")
    
    if cfg_dataset['ensamble']['flag'] == False:
        valid_dataset, test_dataset, train_data_loader, valid_data_loader, test_data_loader = prep_train(cfg_dataset, cfg_train)
    else:
        valid_dataset, test_dataset, train_data_loader, valid_data_loader, test_data_loader = prep_train_ensamble(cfg_dataset, cfg_train)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg_train['lr'], momentum=cfg_train['momentum'], weight_decay=cfg_train['weight_decay'])

    if cfg_train['lr_scheduler']['flag'] == False:
        lr_scheduler = None
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg_train['lr_scheduler']['lr_step_size'],
                                                                       gamma=cfg_train['lr_scheduler']['lr_gamma'])
    start_epoch = 0
    itr = 1
    total_train_loss = []
    total_valid_loss = []

    if cfg_train['checkpoint']:
        print("Resuming checkpoint...")
        checkpoint = torch.load(cfg_train['checkpoint'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler != None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        itr = checkpoint['itr']
        total_train_loss = checkpoint['total_train_loss']
        total_valid_loss = checkpoint['total_valid_loss']
        print(f"From epoch: {start_epoch}")

    losses_value = 0.0
    f_log = open(cfg_train['log_file'], "w")
    early_stopping = EarlyStopping(patience=3, verbose=True, path=cfg_train['path_saved_model'])
    num_epochs = cfg_train['epochs']
    print("Training...\n")
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        # train ------------------------------
        running_corrects = 0
        model.train()
        train_loss = []
         
        for images, targets in tqdm(train_data_loader, desc=f'EPOCH [{epoch+1}]: '):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses_value = losses.item()
            train_loss.append(losses_value)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
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
        f_log.write(f"Epoch Completed: {epoch+1}/{num_epochs}, Time: {time.time()-start_time}, "
              f"Train Loss: {epoch_train_loss}, Valid Loss: {epoch_valid_loss}\n")  
        print(f"Epoch Completed: {epoch+1}/{num_epochs}, Time: {time.time()-start_time}, "
              f"Train Loss: {epoch_train_loss}, Valid Loss: {epoch_valid_loss}")

        #mAP and accuracy over validation
        f_log.write("\nVALIDATION PHASE: ")
        print("\nVALIDATION PHASE: ")
        sys.stdout = f_log
        elem = evaluate(model, valid_data_loader, device=device)
        elem.summarize()
        #classifier_performance(valid_dataset, model, device)
        sys.stdout = original_stdout
        
        #mAP and accuracy over testing every 2 epochs
        if (epoch+1) % 3 == 0:
            f_log.write(f"\nTESTING PHASE EPOCH {epoch+1}: ")  
            print(f"\nTESTING PHASE EPOCH {epoch+1}: ")
            sys.stdout = f_log
            el = evaluate(model, test_data_loader, device=device)
            el.summarize()
            #classifier_performance(test_dataset, model, device)
            sys.stdout = original_stdout
            
        checkpoint_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': 
                        lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    'epoch': epoch,
                    'itr': epoch+1,
                    'total_train_loss' : total_train_loss,  
                    'total_valid_loss' : total_valid_loss
                }

        early_stopping(epoch_valid_loss, checkpoint_dict)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    f_log.close()
    print("Training completed!")
    #plot valid-train loss
    if len(total_train_loss) > 0 and len(total_valid_loss) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(total_train_loss, label="Train Loss")
        plt.plot(total_valid_loss, label="Valid Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(cfg_train['plot'])
        plt.show()
        print("Plot training/validation loss saved!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--conf_file', required=True, help="YAML config file path")

    args = parser.parse_args()

    train(args)
