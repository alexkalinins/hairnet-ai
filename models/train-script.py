import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os

from HN3G1 import HN3G1

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
    model_loss = model_loss_func(outs.view(-1, 1, 1), params)
    simple_loss = simple_loss_func(outs.view(-1, 1, 1), params)

    if train:
        model_loss.backward()
        optimizer.step()

    return model_loss, simple_loss


def test(model, model_loss_func, param_index, size=32):
    # sample without replacement:
    index = random.sample(range(len(test_imgs)), size)
    index = torch.tensor(index)

    batch_imgs = test_imgs[index]
    batch_params = test_params[index]
    batch_params = torch.Tensor([p[0][param_index] for p in batch_params])  # Selecting for the index.

    with torch.no_grad():
        m_loss, s_loss = fwd_pass(model, None, model_loss_func, batch_imgs.view(-1, 1, 200, 200).to(device), batch_params.view(-1, 1, 1).to(device))
    return m_loss, s_loss


def train(model, name, optimizer, model_loss_func, param_index):
    BATCH_SIZE = 20
    EPOCHS = 70  # the best will be selected before overfit

    try:
        os.mkdir(f'models/{name}')
    except Exception as e:
        # incase dir exists
        pass

    with open(f'models/{name}/{name}.log', 'a') as file:
        file.write(f'epoch,step,train_m_loss,train_s_loss,test_m_loss,test_s_loss\n')

        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_imgs), BATCH_SIZE)):
                batch_imgs = train_imgs[i:i + BATCH_SIZE].view(-1, 1, 200, 200).to(device)
                batch_params = train_params[i:i + BATCH_SIZE].to(device)
                batch_params = torch.Tensor([p[0][param_index] for p in batch_params])

                train_m_loss, train_s_loss = fwd_pass(model, optimizer, model_loss_func, batch_imgs, batch_params.view(-1, 1, 1), train=True)

                # test every 135 steps; 2 times per epoch
                if i % 135 == 0:
                    test_m_loss, test_s_loss = test(model, model_loss_func, param_index, size=20)

                    # logging model progress:
                    file.write(
                        f'{epoch},{i},{round(float(train_m_loss), 4)},{round(float(train_s_loss), 4)},'
                        f'{round(float(test_m_loss), 4)},{round(float(test_s_loss), 4)}\n')

            if epoch >=40:
                # saving model every epoch (after 5th)
                pt_file_path = f'models/{name}/e{epoch}.pt'
                torch.save(model.state_dict(), pt_file_path)
                print(f'Saved model after {epoch}th epoch')


START = 0
END = 9

for i in range(START, END):
    model = HN3G1().to(device)
    loss_func = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print('Training model: ', str(i))
    train(model, f'3g-index{i}', optimizer, loss_func, i)
    print('Done training model: ', str(i))



