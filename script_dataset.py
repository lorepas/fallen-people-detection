from utils.load_dataset import *
from utils.custom_utils import *

DELTA = 1

def build_virtual(filename):
    
    if os.path.exists(filename):
        print("File already created")
    else:
        files = glob.glob('data/**/*.txt', recursive = True) #find all *.txt files    
        #FILES NOT ORIG (normalized)
        fw = open(os.path.join("datasets",filename),'w')
        i = 0
        print("Creating virtual dataset...")
        for fi in tqdm(files):
            if "orig" not in fi: #discard all *_orig.txt
                f = open(fi, 'r')
                file_text = f.read()
                file_text = file_text.strip('\n') #remove possible newline character at the end of file
                lines = file_text.split('\n')
                file_name = os.path.join(fi.split("/")[-2],fi.split("/")[-1]) #last / and first part before the . with the name of the folder
                if len(lines) == 0:
                    continue            
                for l in lines:
                    if l == '':
                        continue
                    if float(l.split(" ")[6]) < 0.81: #discard images with occlusion too high
                        label = l.split(" ")[0]
                        if int(label) == 6 or int(label) == 7:
                            name = file_name.split(".")[0] + ".png"
                            image = cv2.imread(os.path.join('data',name))
                            height,width, _ = image.shape #values to denormalize data
                            xc = float(l.split(" ")[1]) * width 
                            yc = float(l.split(" ")[2]) * height
                            w = float(l.split(" ")[3]) * width
                            h = float(l.split(" ")[4]) * height
                            #take values as (x0,y0) and (x1,y1) for top-left and bottom-right points of BB
                            x0 = float(xc - w / 2) 
                            y0 = float(yc - h / 2) 
                            x1 = float(xc + w / 2) 
                            y1 = float(yc + h / 2)

                            if int(label) == 6: #label 1 -> NO FALL
                                fw.write(name + ',' + str(x0) + ',' + str(y0) + "," + str(x1) + "," + str(y1) + "," + str(1) + "\n")
                            elif int(label) == 7: #label 2 -> FALL
                                fw.write(name + ',' + str(x0) + ',' + str(y0) + "," + str(x1) + "," + str(y1) + "," + str(2) + "\n")
        fw.close()
        print("Completed!")
        
def build_real(filename):
    if os.path.exists(filename):
        print("File already created")
    else:
        filename = filename.split(".")
        filenamev = filename[0]+"_valid.txt"
        filenamet = filename[0]+"_train.txt"
        files = glob.glob('real_dataset/valid/**/*.png', recursive = True) #find all *.png files
        fw = open(os.path.join("datasets",filenamev),'w')
        i = 0
        print("Creating validation dataset...")
        for fi in tqdm(files):
            fi = fi.split(".")[0] + ".txt"
            f = open(fi, 'r')
            file_text = f.read()
            if file_text == "": #discard empty files
                continue
            file_text = file_text.strip('\n') #remove possible newline character at the end of file
            lines = file_text.split('\n')
            file_name = os.path.join(fi.split("/")[-2],fi.split("/")[-1])
            for l in lines:
                name = file_name.split(".")[0] + ".png"
                image = cv2.imread(f'real_dataset/valid/{name}')
                y,x,_ = image.shape

                label = l.split(" ")[0]
                x0 = l.split(" ")[1]
                x1 = l.split(" ")[2]
                y0 = l.split(" ")[3]
                y1 = l.split(" ")[4]

                if int(x0) < 0 or int(x1) > x:
                    continue
                if int(y0) < 0 or int(y1) > y:
                    continue

                if int(label) == -1: #label 1 -> NO FALL
                    fw.write(name + ',' + str(x0) + ',' + str(y0) + "," + str(x1) + "," + str(y1) + "," + str(1) + "\n")
                elif int(label) == 1: #label 2 -> FALL
                    fw.write(name + ',' + str(x0) + ',' + str(y0) + "," + str(x1) + "," + str(y1) + "," + str(2) + "\n")
        fw.close()
        print("Validation dataset completed!")
        
        files = glob.glob('real_dataset/train/**/*.png', recursive = True) #find all *.txt files
        fw = open(os.path.join("datasets",filenamet),'w')
        i = 0
        print("Creating training dataset...")
        for fi in tqdm(files):
            fi = fi.split(".")[0] + ".txt"
            f = open(fi, 'r')
            file_text = f.read()
            if file_text == "": #discard empty files
                continue
            file_text = file_text.strip('\n') #remove possible newline character at the end of file
            lines = file_text.split('\n')
            file_name = os.path.join(fi.split("/")[-2],fi.split("/")[-1])
            for l in lines:
                name = file_name.split(".")[0] + ".png"
                image = cv2.imread(f'real_dataset/train/{name}')
                y,x,_ = image.shape

                label = l.split(" ")[0]
                x0 = l.split(" ")[1]
                x1 = l.split(" ")[2]
                y0 = l.split(" ")[3]
                y1 = l.split(" ")[4]

                if int(x0) < 0 or int(x1) > x:
                    continue
                if int(y0) < 0 or int(y1) > y:
                    continue

                if int(label) == -1: #label 1 -> NO FALL
                    fw.write(name + ',' + str(x0) + ',' + str(y0) + "," + str(x1) + "," + str(y1) + "," + str(1) + "\n")
                elif int(label) == 1: #label 2 -> FALL
                    fw.write(name + ',' + str(x0) + ',' + str(y0) + "," + str(x1) + "," + str(y1) + "," + str(2) + "\n")
        fw.close()
        print("Train dataset completed!")

def build_test(filename):

    if os.path.exists(filename):
        print("File already created")    
    else:
        files = glob.glob('test_imgs/*.txt', recursive = True) #find all *.txt files
        fw = open(os.path.join("datasets",filename),'w')
        i = 0
        print("Creating test dataset...")
        for fi in files:
            f = open(fi, 'r')
            file_text = f.read()
            if file_text == "": #discard empty files
                continue
            file_text = file_text.strip('\n') #remove possible newline character at the end of file
            lines = file_text.split('\n')
            file_name = fi.split("/")[-1] #the part before the . with the name of the folder
            for l in lines:
                name = file_name.split(".")[0] + ".png"
                #image = cv2.imread(os.path.join('test_imgs',name))
                label = l.split(" ")[0]
                x0 = l.split(" ")[1]
                x1 = l.split(" ")[2]
                y0 = l.split(" ")[3]
                y1 = l.split(" ")[4]

                if int(label) == -1: #label 1 -> NO FALL
                    fw.write(name + ',' + str(x0) + ',' + str(y0) + "," + str(x1) + "," + str(y1) + "," + str(1) + "\n")
                elif int(label) == 1: #label 2 -> FALL
                    fw.write(name + ',' + str(x0) + ',' + str(y0) + "," + str(x1) + "," + str(y1) + "," + str(2) + "\n")
        fw.close()
        print("Test dataset created!")
        
def build_test_udf(filename):

    if os.path.exists(filename):
        print("File already created")    
    else:
        files = glob.glob('test_ufd/*.txt', recursive = True) #find all *.txt files
        fw = open(os.path.join("datasets",filename),'w')
        i = 0
        print("Creating test dataset...")
        for fi in files:
            f = open(fi, 'r')
            file_text = f.read()
            file_name = fi.split("/")[-1] #the part before the . with the name of the folder
            if file_text == "": #discard empty files
                i += 1
                name = file_name.split(".")[0] + ".png"
                #image = cv2.imread(os.path.join('test_ufd',name))
                fw.write(name + ',' + str(DELTA) + ',' + str(DELTA) + "," + str(0) + "," + str(0) + "," + str(1) + "\n")
                continue
            file_text = file_text.strip('\n') #remove possible newline character at the end of file
            lines = file_text.split('\n')      
            for l in lines:
                name = file_name.split(".")[0] + ".png"
                #image = cv2.imread(os.path.join('test_ufd',name))
                label = l.split(" ")[0]
                x0 = l.split(" ")[1]
                x1 = l.split(" ")[2]
                y0 = l.split(" ")[3]
                y1 = l.split(" ")[4]

                if int(label) == -1: #label 1 -> NO FALL
                    fw.write(name + ',' + str(x0) + ',' + str(y0) + "," + str(x1) + "," + str(y1) + "," + str(1) + "\n")
                elif int(label) == 1: #label 2 -> FALL
                    fw.write(name + ',' + str(x0) + ',' + str(y0) + "," + str(x1) + "," + str(y1) + "," + str(2) + "\n")
        fw.close()
        print(f"Empty annotations are: {i}")
        print("Test dataset created!")
        
def build_test_elderly(filename):

    if os.path.exists(filename):
        print("File already created")    
    else:
        files = glob.glob('test_elderly/*.txt', recursive = True) #find all *.txt files
        fw = open(os.path.join("datasets",filename),'w')
        e = 0
        nv = 0
        print("Creating test dataset...")
        for fi in files:
            f = open(fi, 'r')
            file_text = f.read()
            file_name = fi.split("/")[-1] #the part before the . with the name of the folder
            if file_text == "": #discard empty files
                e += 1
                continue
            file_text = file_text.strip('\n') #remove possible newline character at the end of file
            lines = file_text.split('\n')      
            for l in lines:
                name = file_name.split(".")[0] + ".png"
                image = cv2.imread(os.path.join('test_elderly',name))
                y,x,_ = image.shape
                
                label = l.split(" ")[0]
                x0 = l.split(" ")[1]
                x1 = l.split(" ")[2]
                y0 = l.split(" ")[3]
                y1 = l.split(" ")[4]
                
                if float(x0) < 0 or float(x1) > x:
                    nv += 1
                    continue
                if float(y0) < 0 or float(y1) > y:
                    nv += 1
                    continue

                if int(label) == -1: #label 1 -> NO FALL
                    fw.write(name + ',' + str(x0) + ',' + str(y0) + "," + str(x1) + "," + str(y1) + "," + str(1) + "\n")
                elif int(label) == 1: #label 2 -> FALL
                    fw.write(name + ',' + str(x0) + ',' + str(y0) + "," + str(x1) + "," + str(y1) + "," + str(2) + "\n")
        fw.close()
        print(f"Empty annotations are: {e}")
        print(f"Not valid and discard annotations are: {nv}")
        print("Test dataset created!")
        
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--dt_type', required=True, help="Choose between real or virtual")
    
    parser.add_argument('--filename', required=True, help="Insert filename")

    args = parser.parse_args()

    dt = args.dt_type
    if ".txt" not in args.filename:
        print("Insert the name with the extension")
        sys.exit(1)
    if dt != "train_real" and dt != "train_virtual" and dt != "test" and dt != "test_ufd" and dt != "test_elderly":
        print("Insert \"train_real\",\"train_virtual\" or \"test\"")
        sys.exit(1)
    elif dt == "train_real":
        build_real(args.filename)
    elif dt == "train_virtual":
        build_virtual(args.filename)
    elif dt == "test":
        build_test(args.filename)
    elif dt == "test_ufd":
        build_test_udf(args.filename)
    elif dt == "test_elderly":
        build_test_elderly(args.filename)