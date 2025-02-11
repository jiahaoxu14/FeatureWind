import sys
# import DualNum
import csv
import numpy as np
from tsne import tsne
import torch

class ProjectionRunner:
    def __init__(self, projection, params=None):
        self.params = params
        self.projection = projection
        self.firstRun = False

    def calculateValues(self, points, perturbations=None):
        self.points = points
        # self.origPoints = points

        # Convert data to PyTorch tensor with requires_grad=True
        data = torch.tensor(points, dtype=torch.float32, requires_grad=True)

        # Usage for DimReader:
        with torch.no_grad():
            Y_base, params = tsne(data, 2, 999, 50, 20.0, save_params = True)
        Y, params = tsne(data, no_dims=2, maxIter = 1, initial_dims=50, perplexity=20.0, save_params = False,
                            initY = params[0], initBeta = params[2], betaTries = 50, initIY =params[1])
        
        # Compute gradients
        self.outPoints = Y
        grads = []

        for i in range(len(Y)):
            # Compute gradients for x and y coordinates
            print("Computing gradients for point %d" % (i))
            grad_x = torch.autograd.grad(Y[i, 0], data, retain_graph=True)[0][i]
            grad_y = torch.autograd.grad(Y[i, 1], data, retain_graph=True)[0][i]
            grads.append(torch.stack([grad_x, grad_y], dim=0))

        # Convert gradients to NumPy array
        self.grads = torch.stack(grads).detach().numpy()

projections = ["tsne", "Tangent-Map"]
projectionClasses = [tsne, None]
projectionParamOpts = [["Perplexity", "Max_Iterations", "Number_of_Dimensions"], []]

# read points of cvs file
def readFile(filename):
    read = csv.reader(open(filename, 'rt'))

    points = []
    firstLine = next(read)
    headers = []
    rowDat = []
    head = False
    for i in range(0, len(firstLine)):
        try:
            rowDat.append(float(firstLine[i]))
        except:
            head = True
            break
    if head:
        headers = firstLine
    else:
        points.append(rowDat)

    for row in read:
        rowDat = []
        for i in range(0, len(row)):
            try:
                rowDat.append(float(row[i]))
            except:
                print("invalid data type - must be numeric")
                exit(0)
        points.append(rowDat)
    return points