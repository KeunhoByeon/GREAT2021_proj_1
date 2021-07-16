import argparse
from time import time

import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np

import onlinehd
import os
from tqdm import tqdm


def index2alphabet(index):
    '''
    Convert index to alphabet
    index: 0 A, 1 B, ... , 25 Z
    '''
    return chr(65 + index)


# loads isolet data
def load():
    trainset_path = './data/train/isolet1+2+3+4.data'
    testset_path = './data/test/isolet5.data'

    x, x_test = [], []
    y, y_test = [], []

    # Load trainset
    with open(trainset_path, 'r') as rf:
        for line in rf.readlines():
            # Remove new line char and split line
            line = line.replace('\n', '').split(', ')

            # Get x and y
            data = np.asarray(line[:-1]).astype(float)
            index = int(float(line[-1])) - 1  # Change label range from 1~26 to 0~25

            x.append(data)
            y.append(index)

    # Load testset
    with open(testset_path, 'r') as rf:
        for line in rf.readlines():
            # Remove new line char and split line
            line = line.replace('\n', '').split(', ')

            # Get x and y
            data = np.asarray(line[:-1]).astype(float)
            index = int(float(line[-1])) - 1  # Change label range from 1~26 to 0~25

            x_test.append(data)
            y_test.append(index)

            # print(index, index2alphabet(index))

    x = np.array(x).astype(np.float64)
    y = np.array(y).astype(np.float64)
    x_test = np.array(x_test).astype(np.int32)
    y_test = np.array(y_test).astype(np.int32)

    # Normalize
    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    # Changes data to pytorch's tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    # print(x.size(), y.size())
    # print(x_test.size(), y_test.size())
    # print(y.unique(), y_test.unique())

    return x, x_test, y, y_test


# simple OnlineHD training
def main(args):
    # print('Loading...')
    x, x_test, y, y_test = load()
    classes = y.unique().size(0)
    features = x.size(1)
    model = onlinehd.OnlineHD(classes, features, dim=args.dimension)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

    # print('Training...')
    t = time()
    model = model.fit(x, y, bootstrap=args.bootstrap, lr=args.lr, epochs=args.epochs, one_pass_fit=args.one_pass_fit)
    t = time() - t

    # print('Validating...')
    yhat = model(x)
    yhat_test = model(x_test)
    acc = (y == yhat).float().mean()
    acc_test = (y_test == yhat_test).float().mean()

    if args.default_test:
        '''
        Save the result as .csv if you are doing default test
        Header: lr,epochs,dimension,bootstrap,acc,acc_test,t
        '''
        with open(os.path.join(args.results, 'results.csv'), 'a') as wf:
            wf.write('{},{},{},{},{},{},{}\n'.format(args.lr, args.epochs, args.dimension, args.bootstrap, acc, acc_test, t))
        # print(f'{acc_test = :6f}')
    else:
        # or just print the result
        print(f'{acc = :6f}')
        print(f'{acc_test = :6f}')
        print(f'{t = :6f}')

    return acc, acc_test, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.2, metavar='L')
    parser.add_argument('--epochs', default=20, metavar='E')
    parser.add_argument('--dimension', default=5000, metavar='D')
    parser.add_argument('--bootstrap', default=0.25, metavar='B')
    parser.add_argument('--one_pass_fit', default=True, metavar='O')
    parser.add_argument('--results', default='./results', metavar='R')
    parser.add_argument('--default_test', default=False, action='store_true')
    args = parser.parse_args()

    if args.default_test:  # Run default test
        os.makedirs(args.results, exist_ok=True)

        # Write result csv header
        with open(os.path.join(args.results, 'results.csv'), 'w') as wf:
            wf.write('lr,epochs,dimension,bootstrap,acc,acc_test,t\n')

        pbar = tqdm(total=4 * 3 * 3 * 2)  # Progress bar iterator
        for lr in (0.2, 0.3, 0.4, 0.5):
            for epochs in (20, 40, 60):
                for dimension in (5000, 7500, 10000):
                    for bootstrap in (0.25, 0.5):
                        # Set args
                        args.lr = lr
                        args.epochs = epochs
                        args.dimension = dimension
                        args.bootstrap = bootstrap
                        args.one_pass_fit = True

                        main(args)

                        pbar.update(1)
    else:
        main(args)
