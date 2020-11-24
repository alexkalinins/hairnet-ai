import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


class DataBuilder():
    IMG_DIR = 'training-data/gaussian/images'
    IMG_SIZE = 200  # 200x200
    PARAM_DIR = 'training-data/gaussian/parameters.csv'
    data = []
    data_count = 0

    def open_data(self):
        params_df = pd.read_csv(self.PARAM_DIR)

        for file in tqdm(os.listdir(self.IMG_DIR)):
            try:
                # load image
                path = os.path.join(self.IMG_DIR, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                # load param for the image
                param = params_df.loc[params_df['file_name'] == file]  # get params with that file name
                param = param.drop(columns=['file_name'])  # drops filename column

                # adding to data
                self.data.append([np.array(img), param.to_numpy()])
                self.data_count += 1
            except Exception as e:
                print(str(e))

        assert self.data_count == 6000
        np.random.shuffle(self.data)
        np.save('data.npy', self.data)
        print(f'count: {self.data_count}')


if __name__ == '__main__':
    builder = DataBuilder()
    builder.open_data()
