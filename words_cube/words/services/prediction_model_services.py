
import os
import json
import numpy as np
import pickle
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.io import imread,imsave
from skimage import data
from skimage.color import rgb2gray
# import matplotlib.pyplot as plt
import torch
 


services_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.dirname(services_dir))
new_root=os.path.abspath(os.path.dirname(project_root))
print('new_root',new_root)
path_data = os.path.join(project_root, 'resources\\')
path_hubconfig= os.path.abspath(os.path.dirname(new_root)) +'\yolov5'
print("****path_hubconfig***", path_hubconfig)
path_weightfile=path_hubconfig+'\models\yolov5s.pt'
print("******path_weightfile******",path_weightfile)



def predict(predict_model_request):
    
    data = predict_model_request
    data=data.data
    data=data["data"]
    print("finsish read rquest dataaa")
    # path_hubconfig=r'E:\words_cube\words_cube\yolov5'
    # path_weightfile=r'E:\words_cube\words_cube\yolov5\models\yolov5s.pt'
    model = torch.hub.load(path_hubconfig, 'custom',path=path_weightfile, source='local')
    
    data=np.array(data)
    imsave("out1.jpg",data)
    img=imread('out1.jpg')
    results = model(img, size=640)
    results.save(path_data)
    print(results.pandas().xyxy[0].to_json(orient="records"))
    for img1 in results.imgs:
        img_base64 = Image.fromarray(img1)

        img_base64.save(path_data+"image0.jpg", format="JPEG")
    


    filename=path_data+"image0.jpg"
    image = Image.open(filename)
    json_data =np.array(image).tolist()

    
    
    return {'data':json_data}
    # return results.pandas().xyxy[0].to_json(orient="records")



