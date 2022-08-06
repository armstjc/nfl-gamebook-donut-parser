from pdf2image import convert_from_path

from donut import DonutModel
from PIL import Image
import torch

import os
import glob

def getPdfInFolder(folder=""):
    f = []
    dir_list = os.listdir(folder)
    l = list(filter(lambda x: '.pdf' in x, dir_list))
    #print(l)
    for i in l:
        f.append(folder+'/'+i)
    #print(f)
    return f

def removeTempFiles(folder=""):
    #dir_list = os.listdir(f'{folder}')
    files = glob.glob(f'{folder}/*.jpg')

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

def reformatFolderString(folder=""):
    r = folder.replace("\\","/")
    print(f'Working Directory: \n{r}')
    return r

def testCUDA():
    ## If the result is false, run this:
    ## pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

    ## If you aren't using pip, follow the guide here:
    ## https://pytorch.org/
    print("CUDA available:", torch.cuda.is_available())
    #!nvcc --version

def convertPdfToJpg(fileExtension:str):
    try:
        os.mkdir('temp')
    except:
        pass
    pdfImages = convert_from_path(fileExtension)
    
    for i in range(len(pdfImages)):
        pdfImages[i].save(f'temp/{i}.jpg','JPEG')

def convertImagesV1(imageFile=""):
    model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    if torch.cuda.is_available():
        model.half() 
        device = torch.device("cuda") 
        model.to(device) 
    else: 
        model.encoder.to(torch.bfloat16)
    model.eval() 
    image = Image.open(imageFile).convert("RGB")
    output = model.inference(image=image, prompt="<s_cord-v2>")
    print(output)



def main():
    #testCUDA()
    convertPdfToJpg('test.pdf')
    
    convertImagesV1("temp/2.jpg")
if __name__ == "__main__":
    main()