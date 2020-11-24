import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from HN3G1 import HN3G1


MODEL_DIR = 'models/gen3/trained-state/'
MODEL_SAVE = [MODEL_DIR+'ix0-e28.pt', MODEL_DIR+'ix1-e36.pt', MODEL_DIR+'ix2-e69.pt', MODEL_DIR+'ix3-e69.pt', MODEL_DIR+'ix4-e34.pt', MODEL_DIR+'ix1-e36.pt', MODEL_DIR+'ix6-e69.pt',MODEL_DIR+'ix7-e69.pt', MODEL_DIR+'ix8-e69.pt']  # would be a list of save files for all the models

class Model:
    def __init__(self):
        assert len(MODEL_SAVE) == 9
        self.models = []

        for i in range(9):
            try:
                net = HN3G1()
                net.load_state_dict(torch.load(MODEL_SAVE[i]))
                net.eval()

                self.models.append(net)
            except Exception as e:
                print(f'Could not load model {i}')
                print(str(e))
                sys.exit(1)


    def pass_image(self, path: str):
        try:
            # opening image
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 200))  # images should already be cropped!
        except Exception as e:
            print('Image invalid! (wrong path or image corrupted)')
            print(str(e))
            sys.exit(1)

        img_tensor = torch.Tensor(img).view(-1, 1, 200, 200)
        outs = []

        for m in self.models:
            outs.append(float(m.forward(img_tensor).view(1)))

        # conditioning some values to work with blender
        outs[3] = abs(outs[3])  # c_offset
        outs[6] = abs(outs[6])  # p_offset
        outs[7] = abs(round(outs[7]))  # count (a natural number)

        # count randomness (a real number between 0 and 1)
        if outs[8] < 0:
            outs[8] = 0
        elif outs[8] > 1:
            outs[8] = 1

        return outs
