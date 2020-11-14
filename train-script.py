import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os

from HairNet1 import HairNet1
from HairNet2 import HairNet2
from HairNet3 import HairNet3

# configuring for cuda
device = torch.device('cpu')
print(f'using: {device}')

# loading data
data = np.load('data.npy', allow_pickle=True)

try:
    os.mkdir('models')
    print('Created models directory')
except Exception as e:
    # incase because dir exists
    print('Model directory likely exists')


def split_train_test_data():
    imgs = torch.Tensor([d[0] for d in data])
    imgs /= 255.0  # to get pixel values on [0, 1]
    params = torch.Tensor([d[1] for d in data])

    test_size = len(imgs) // 10  # 1/10 of all data

    # Spliting data into training and testing
    test_imgs = imgs[-test_size:]
    test_params = params[-test_size:]

    train_imgs = imgs[:-test_size]
    train_params = params[:-test_size]

    assert len(test_imgs) == len(test_params)
    assert len(train_imgs) == len(train_params)

    return train_imgs, train_params, test_imgs, test_params


# splitting data into training and testing
print('splitting data into training and testing datasets')
train_imgs, train_params, test_imgs, test_params = split_train_test_data()
print('done')

simple_loss_func = nn.L1Loss()  # loss function for analysis


def fwd_pass(model, optimizer, model_loss_func, imgs, params, train=False):
    if train:
        model.zero_grad()

    outs = model(imgs)
    model_loss = model_loss_func(outs.view(-1, 1, 9), params)
    simple_loss = simple_loss_func(outs.view(-1, 1, 9), params)

    if train:
        model_loss.backward()
        optimizer.step()

    return model_loss, simple_loss


def test(model, model_loss_func, size=32):
    # sample without replacement:
    index = random.sample(range(len(test_imgs)), size)
    index = torch.tensor(index)

    batch_imgs = test_imgs[index]
    batch_params = test_params[index]

    with torch.no_grad():
        m_loss, s_loss = fwd_pass(model, None, model_loss_func, batch_imgs.view(-1, 1, 200, 200).to(device),
                                  batch_params.view(-1, 1, 9).to(device))
    return m_loss, s_loss


def train(model, name, optimizer, model_loss_func):
    BATCH_SIZE = 20
    EPOCHS = 20  # the best will be selected before overfit

    try:
        os.mkdir(f'models/{name}')
    except Exception as e:
        # incase dir exists
        pass

    with open(f'models/{name}/{name}.log', 'a') as file:
        file.write(f'epoch,step,train_m_loss,train_s_loss,test_m_loss,test_s_loss')

        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_imgs), BATCH_SIZE)):
                batch_imgs = train_imgs[i:i + BATCH_SIZE].view(-1, 1, 200, 200).to(device)
                batch_params = train_params[i:i + BATCH_SIZE].to(device)

                train_m_loss, train_s_loss = fwd_pass(model, optimizer, model_loss_func, batch_imgs, batch_params,
                                                      train=True)

                # test every 135 steps; 2 times per epoch
                if i % 135 == 0:
                    test_m_loss, test_s_loss = test(model, model_loss_func, size=32)

                    # logging model progress:
                    file.write(
                        f'{epoch},{i},{round(float(train_m_loss), 4)},{round(float(train_s_loss), 4)},'
                        f'{round(float(test_m_loss), 4)},{round(float(test_s_loss), 4)}')

            # saving model every epoch
            pt_file_path = f'models/{name}/e{epoch}.pt'
            torch.save(model.state_dict(), pt_file_path)
            print(f'Saved model after {epoch}th epoch')


# creating models
hn1 = HairNet1().to(device)
hn2 = HairNet2().to(device)
hn3 = HairNet3().to(device)

hn1_loss_func = nn.MSELoss()
hn2_loss_func = nn.MSELoss()
hn3_loss_func = nn.MSELoss()

hn1_optim = optim.Adam(hn1.parameters(), lr=0.001)
hn2_optim = optim.Adam(hn2.parameters(), lr=0.001)
hn3_optim = optim.Adam(hn3.parameters(), lr=0.001)

# training models!
print('Training model 1')
train(hn1, 'HairNet1', hn1_optim, hn1_loss_func)
print('Done training model 1')

print('Training model 2')
train(hn2, 'HairNet2', hn2_optim, hn2_loss_func)
print('Done training model 2')

print('Training model 3')
train(hn3, 'HairNet3', hn3_optim, hn3_loss_func)
print('Done training model 3')
