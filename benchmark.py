import os
import time
import json

from dataprep import load_data
from neuralnet import Net, train, verify


if not os.path.isdir('./data'):
    os.mkdir('./data')

if not os.path.isdir('./models'):
    os.mkdir('./models')

performances = []
widths = [6, 10, 50, 100, 200, 350, 500]

# load the data
trainloader, testloader, classes = load_data()

# start experiment
for gpu in [False, True]:
    for width in widths:
        # init net
        net = Net(n=width)

        # train neural net
        t0 = time.clock()
        trained_net = train(net, trainloader, gpu=gpu, filename=width,
                            verbose=False)
        t_elapsed = time.clock() - t0

        # verify neural net
        performance = verify(net, testloader, classes, verbose=True)

        # save to dict
        performance['width'] = width
        performance['device'] = f"{'gpu' if gpu else 'cpu'}"
        performance['elapsed'] = t_elapsed
        performances.append(performance)

# performances to disk, append to json if already exists
# json is very small so don't mind the efficiency
try:
    with open('./models/traindata.json', 'r') as fr:
        data = json.load(fr)
        updated = [*data, *performances]
except FileNotFoundError:
    updated = performances

with open('./models/traindata.json', 'w') as fw:
    json.dump(updated, fw)
