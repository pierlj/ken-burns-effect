import math
import torch

def reduceNet(net, architecture):
    children = list(net.children())
    if children != []:
        for child in children:
            architecture = architecture + reduceNet(child, [])
        return architecture
    else:
        return [net]

def extractParams(architecture):
    convnet = []
    names = []
    for layer in architecture:
        if 'Conv' in layer.__class__.__name__:
            convnet.append([layer.kernel_size[0], layer.stride[0], layer.padding[0], layer.dilation[0]])
            names.append(layer.__class__.__name__)
        elif 'MaxPool' in layer.__class__.__name__:
            convnet.append([layer.kernel_size, layer.stride, layer.padding, layer.dilation])
            names.append(layer.__class__.__name__)
        elif 'AvgPool' in layer.__class__.__name__:
            convnet.append([layer.kernel_size, layer.stride, layer.padding, 1])
            names.append(layer.__class__.__name__)
    return convnet, names

def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]
    d = conv[3]

    n_out = math.floor((n_in + 2*p - d*k)/s) + 1
    actualP = (n_out-1)*s - n_in + k 
    pR = math.ceil(actualP/2)
    pL = math.floor(actualP/2)

    j_out = j_in * s
    r_out = r_in + d * (k - 1) * j_in
    start_out = start_in + ((k-1)/2 - pL)*j_in
    return n_out, j_out, r_out, start_out

def printLayer(layer, layer_name):
    print(layer_name + ":")
    # print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))
    print("\t n features: %s \n \t receptive size: %s" % (layer[0], layer[2]))

def netSummary(net, imsize):
    layerInfos = []
    print ("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]
    printLayer(currentLayer, "input image")
    convnet, layer_names = extractParams(reduceNet(net, []))
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        printLayer(currentLayer, layer_names[i])
    return layerInfos