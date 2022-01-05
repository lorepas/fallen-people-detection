from utils.load_dataset import *
from utils.custom_utils import *
from torchvision import transforms

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


def visualize_prediction(img, model, thr=0.7):
    convert_tensor = transforms.ToTensor()
    img = convert_tensor(img)
    with torch.no_grad():
        prediction = model([img.to("cuda")])
    p = take_prediction(prediction[0],thr)
    for bb,label,score in p:
        if label == 0:
            im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
            continue
        elif label == 1:
            color = "green"
            text = f"no fallen: {score:.3f}"
        else:
            color = "blue"
            text = f"fallen: {score:.3f}"
        x0,y0,x1,y1 = bb
        im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        draw = ImageDraw.Draw(im)
        draw.rectangle(((x0, y0),(x1,y1)), outline=color, width=3)
        draw.text((x0, y0), text, fill=(0,0,0,0))
    image_array = np.array(im)
    return image_array
    
def take_prediction(prediction, threshold):
    boxes = prediction['boxes'].tolist()
    labels = prediction['labels'].tolist()
    scores = prediction['scores'].tolist()
    if len(boxes) == 0:
        return [([0,0,0,0],0,0.)]
    
    res = [t for t in zip(boxes,labels,scores) if t[2]>threshold]
    if len(res) == 0:
        res = [([0,0,0,0],0,0.)]
    return res

def play_video(path, model_path, out, thr):

    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ext = path.split(".")
    model, _ = load_model(model_path)
    file_out = out
    if ext[1] == "avi":
        out = cv2.VideoWriter(out, cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)
    else:
        out = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        im = visualize_prediction(frame,model,float(thr))
        out.write(im)

    cap.release()
    out.release()
    print(f"Video {file_out} created!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, help="Select a video_path")
    parser.add_argument('--model', required=True, help="Choose between real, virtual, vtr, var")
    parser.add_argument('--output', required=True, help="Insert filename")
    parser.add_argument('--thr', required=True, help="Insert threshold")
    args = parser.parse_args()

    play_video(args.video, args.model, args.output, args.thr)
