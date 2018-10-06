# Student Name: Vikrant Bachhav
# Basic usage: 
#           Base minimum input: python predict.py input checkpointbasic usage
# options are:
#           Return top K most likely classes: python predict.py input checkpoint --top_k 3
#           Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#           Use GPU for inference: python predict.py input checkpoint --gpu

import argparse
import json
import numpy as np
import torch

from PIL import Image
from torch.autograd import Variable

def main():
    args = get_arguments()
    cuda = args.cuda
    model = load_checkpoint(args.checkpoint, cuda)
    model.idx_to_class = dict([[v,k] for k, v in model.class_to_idx.items()])
    
    with open(args.categories, 'r') as f:
        cat_to_name = json.load(f)
      
    prob, classes = predict(args.input, model, topk=int(args.top_k))
    print([cat_to_name[x] for x in classes])
    
    
def get_arguments():
    parser_msg = 'Predict.py takes 2 manditory command line arguments, \n\t1.The image to have a predition made and \n\t2. the checkpoint from the trained nerual network'
    parser = argparse.ArgumentParser(description = parser_msg)

    # Manditory arguments
    parser.add_argument("input", action="store")
    parser.add_argument("checkpoint", action="store")

    # Optional arguments
    parser.add_argument("--top_k", action="store", dest="top_k", default=5, help="Number of top results you want to view.")
    parser.add_argument("--category_names", action="store", dest="categories", default="cat_to_name.json", 
                        help="Number of top results you want to view.")
    parser.add_argument("--cuda", action="store_true", dest="cuda", default=False, help="Set Cuda True for using the GPU")

    return parser.parse_args()

        
def process_image(image):
    expects_means = [0.485, 0.456, 0.406]
    expects_std = [0.229, 0.224, 0.225]
           
    pil_image = Image.open(image).convert("RGB")
    
    # Any reason not to let transforms do all the work here?
    in_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(expects_means, expects_std)])
    pil_image = in_transforms(pil_image)

    return pil_image
    
def predict(image_path, model, topk=5):
    model.eval()
    
    # cpu mode
    model.cpu()
    
    # load image as torch.Tensor
    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, topk)
        
        # Calculate the exponentials
        top_prob = top_prob.exp()
        
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
        
    return top_prob.numpy()[0], mapped_classes


def load_checkpoint(filepath, cuda):
    if cuda:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
        
    return model

main()