from utils.load_dataset import *
from utils.custom_utils import *

def load_model(path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create a Faster R-CNN model without pre-trained
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    num_classes = 3 # wheat or not(background)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained model's head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    # load the trained weights
    #model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    # move model to the right device
    model.to(device)
    return model, device

def take_prediction(prediction, threshold):
    boxes = prediction['boxes'].tolist()
    labels = prediction['labels'].tolist()
    scores = prediction['scores'].tolist()
    if len(boxes) == 0:
        return [([0,0,0,0],1,0.)]
    
    res = [t for t in zip(boxes,labels,scores) if t[2]>threshold]
    if len(res) == 0:
        res = [([0,0,0,0],1,0.)]
    return res

def visualize_prediction(dataset, list_imgs, model, device, path=None, thr=0.7):
    for l in list_imgs:
        img,target = dataset[l]
        with torch.no_grad():
            prediction = model([img.to(device)])
        p = take_prediction(prediction[0],thr)
        for bb,label,score in p:
            if label == 1:
                color = "green"
                text = f"no fallen: {score:.3f}"
            else:
                color = "red"
                text = f"fallen: {score:.3f}"
            x0,y0,x1,y1 = bb
            im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
            draw = ImageDraw.Draw(im)
            draw.rectangle(((x0, y0),(x1,y1)), outline=color, width=3)
            draw.text((x0, y0), text)
        if path is None:
            ImageShow.show(im)
        else:
            im.save(f'snapshots/{path}{l}.png')

def classifier_performance(dataset, model, device, fallen_or_not, tr = 0.7):
    tn, tp, fn, fp = 0,0,0,0
    for im,target in tqdm(dataset):
        gt_labels = target['labels'].tolist()
        with torch.no_grad():
            prediction = model([im.to(device)])
        p = take_prediction(prediction[0], tr)
        pred_labels = [l for _,l,_ in p]
        len_gt_lab = len(gt_labels)
        len_pred_lab = len(pred_labels)
        fall_gt = [i for i in gt_labels if i==fallen_or_not]
        num_fall_gt = len(fall_gt)
        num_no_fall_gt = len_gt_lab - num_fall_gt

        fall_pred = [i for i in pred_labels if i==fallen_or_not] 
        num_fall_pred = len(fall_pred)
        num_no_fall_pred = len_pred_lab - num_fall_pred

        if num_fall_gt == num_fall_pred:
            tp += num_fall_gt
        elif num_fall_gt > num_fall_pred:
            tp += num_fall_pred
            fn += (num_fall_gt - num_fall_pred)
        else:
            tp += num_fall_gt
            fp += (num_fall_pred - num_fall_gt)
    return tp, fp, fn

def inference(mod, dat, filename):
    assert dat == "fpds" or dat == "ufd" or dat == "elderly", "Dataset for inference must be one of: fpds, ufd or elderly"
    
    log_result = open(os.path.join("results",filename), "w")

    if not os.path.exists(os.path.join("models",mod)):
        print("Insert a right model!")
        sys.exit(1)
    else:
        model_path = os.path.join("models",mod)
        log_result.write(f"******PERFORMANCES USING {mod}******\n")


    if dat == "fpds":
        test_set = load_data("test_set.txt")
        test_dataset = FallenPeople(test_set, "test_imgs", FallenPeople.valid_test_transform())
        log_result.write(">>> Used FPDS dataset\n")
    elif dat == "ufd":	#we discard images too similar one from each other
        test_udf = load_data("test_ufd.txt")
        test_d = FallenPeople(test_udf, "test_ufd", FallenPeople.valid_test_transform())
        subset_indeces = [i for i in range(len(test_d)) if i%5==0]
        test_dataset = Subset(test_d, subset_indeces)
        log_result.write(">>> Used Up-Fall Detection dataset\n")
    elif dat == "elderly":
        test_elderly = load_data("test_elderly.txt")
        test_dataset = FallenPeople(test_elderly, "test_elderly", FallenPeople.valid_test_transform())
        log_result.write(">>> Used Elderly dataset\n")

    def collate_fn(batch):
        return tuple(zip(*batch))

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    model, device = load_model(model_path)

    sys.stdout = log_result
    #mAP evaluation
    print("\n--- mAP metrics ---\n")
    elm = evaluate(model, test_data_loader, device=device)
    elm.summarize()
    print("\n--- Classification metrics FALL ---\n")
    #classification metrics fall
    recall, precision, f1_score = [],[],[]
    for thr in np.arange(0,1,0.1):
        tp, fp, fn = classifier_performance(test_dataset, model, device, 2, thr)
        if (tp+fn) == 0 or (tp+fp) == 0:
            print("Bad performances")
            rec = 0
            prec = 0
            f1_s = 0
        else:
            rec = tp / (tp+fn)
            prec = tp/ (tp+fp)
            f1_s = 2/((1/rec)+(1/prec))
        recall.append(round(rec,2))
        precision.append(round(prec,2))
        f1_score.append(round(f1_s,2))
        print(f"-------------------- THR: {thr} --------------------")
        print(f"TP: {tp}\tFN: {fn}\tFP: {fp}")
        print(f"Recall: {rec:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"F1-score: {f1_s:.2f}")
        print("-----------------------------------------------------------")
    
    print(f"Recall@[0.0:0.9]: {recall}")
    print(f"Precision@[0.0:0.9]: {precision}")
    print(f"F1-score@[0.0:0.9]: {f1_score}")
    
    print("\n--- Classification metrics NO FALL ---\n")
    #classification metrics no fall
    recall, precision, f1_score = [],[],[]
    for thr in np.arange(0,1,0.1):
        tp, fp, fn = classifier_performance(test_dataset, model, device, 1, thr)
        if (tp+fn) == 0 or (tp+fp) == 0:
            print("Bad performances")
            rec = 0
            prec = 0
            f1_s = 0
        else:
            rec = tp / (tp+fn)
            prec = tp/ (tp+fp)
            f1_s = 2/((1/rec)+(1/prec))
        recall.append(round(rec,2))
        precision.append(round(prec,2))
        f1_score.append(round(f1_s,2))
        print(f"-------------------- THR: {thr} --------------------")
        print(f"TP: {tp}\tFN: {fn}\tFP: {fp}")
        print(f"Recall: {rec:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"F1-score: {f1_s:.2f}")
        print("-----------------------------------------------------------")
    
    print(f"Recall@[0.0:0.9]: {recall}")
    print(f"Precision@[0.0:0.9]: {precision}")
    print(f"F1-score@[0.0:0.9]: {f1_score}")
    sys.stdout = original_stdout
    log_result.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help="Choose between real, virtual, vtr, var")
    parser.add_argument('--dataset', required=True, help="Choose between fpds, ufd, elderly")
    parser.add_argument('--filename', required=True, help="Insert filename")
    args = parser.parse_args()

    if ".txt" not in args.filename:
        print("Insert the name with the extension")
        sys.exit(1)
    inference(args.model, args.dataset, args.filename)
