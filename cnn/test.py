import torch
import time
import numpy as np
from PIL import Image
import os
import torchvision
import argparse

#from networks import model
from sl_loader import SL_Dataset_Test

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_file', type=str, default='untitled')
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--image_ext', type=str, default='jpg')
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu?')
    parser.add_argument('--output_file', type=str, default='untitled_output.txt')
    args = parser.parse_args()
    return args
    

class TestNetwork:
    def __init__(self, args):
        self.args = args

    def run(self):

        #model = Network.Net()
        model.load_state_dict(torch.load(self.args.model_file))

        if self.args.gpu:
            assert(torch.cuda.is_available())
            model.cuda()

        #grayscale = torchvision.transforms.Grayscale()
        #resize = torchvision.transforms.Resize(512)
        image_transforms = []

        loader = SL_Dataset_Test(
            self.args.image_dir,
            self.args.image_ext,
            image_transforms=image_transforms
        )

        dataset = torch.utils.data.DataLoader(loader, batch_size=1)

        completion = 0
        completion_incr = 100/len(dataset)

        total_startt = time.time()
        for i, image in enumerate(dataset):
            print("[{}%] Processing image {}".format(int(completion), i+1))
            completion += completion_incr
            iter_startt = time.time()

            image.unsqueeze_(1)

            if self.args.gpu: image = image.cuda()

            output = model(image)
            outstring = dataset.data[i]
            for j in range(24): outstring += ' {}'.format(output[j])
            outstring += '\n'

            with open(self.args.output_file, 'a') as f:
                f.write(outstring)
            
            iter_endt = time.time()

            print("Prediction {0} took {1:.4f} seconds".format(i+1,iter_endt-iter_startt))

        total_endt =time.time()
        print("{0:.4f} seconds to complete predictions".format(total_endt-total_startt))

if __name__ == "__main__":
    args = get_args()
    model = TestNetwork(args)
    model.run()