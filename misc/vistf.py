import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

def matrix_to_svg(m , operation = "mul"):
    ul = [-2.0 , 1.0]
    maxAbs = np.abs(m).max()
    mScaled = np.copy(m)
    if(maxAbs > 0):
        mScaled = m / maxAbs
    mRed = np.where(mScaled >= 0.0 , 255.0 * mScaled , 0.0).astype(np.int32)
    mGreen = np.zeros(shape = mScaled.shape , dtype = mScaled.dtype).astype(np.int32)
    mBlue = np.where(mScaled < 0.0 , -255.0 * mScaled , 0.0).astype(np.int32)
    svg = []
    svg = ['<?xml version="1.0"?>\n']
    svg += ["<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" viewBox=\"" + str(-m.shape[1] - 3.0) + " 0.0 " + str(m.shape[1] + 2.0) + " " + str(m.shape[0] + 2.0) + 
            "\"" + ">\n"]
    svg += ['<rect width="100%" height="100%" fill="rgb(255 , 255 , 255)"/>\n']
    for ic in range(m.shape[1]):
        cx = (-m.shape[1] + ic) + ul[0]
        svg += ['<text x = "' + str(cx + 0.5) + '" y = "' + str(0.75) + '" style = "text-anchor : middle" font-size = "0.33">' + str(ic) + '</text>']
    for ir in range(m.shape[0]):
        cx = (-m.shape[1] + 0) + ul[0] 
        cy = ir + ul[1]
        svg += ['<text x = "' + str(cx - 0.5) + '" y = "' + str(cy + 0.85) + '" style = "text-anchor : middle" font-size = "0.33">' + str(ir) + '</text>']
    for ir , ic in np.ndindex(m.shape):
        cx , cy = (-m.shape[1] + ic) + ul[0] , ir + ul[1]
        square = '<rect x="' + str(cx) + '" y="' + str(cy) + '" width="1.0" height="1.0" fill="rgb(' + str(mRed[ir , ic]) + ',' + str(mGreen[ir , ic]) + ',' + str(mBlue[ir , ic]) + ')" />'
        svg += [square]
    if(operation == "mul"):
        svg += ['<circle cx="' + str(ul[0] + 0.5) +  '" cy="' + str(ul[1] - 0.5 + m.shape[0]) + '" r="0.25" />']
    elif(operation == "add"):
        svg += ['<rect x="' + str(ul[0] + 0.5 - 0.25) +  '" y="' + str(ul[1] - 0.5 - 0.05 + m.shape[0]) + '" width = "0.5" height = "0.1" />']
        svg += ['<rect x="' + str(ul[0] + 0.5 - 0.05) +  '" y="' + str(ul[1] - 0.5 - 0.25 + m.shape[0]) + '" width = "0.1" height = "0.5" />']
    
    svg += ["</svg>\n"]

    return "\n".join(svg)

def activation_to_svg(height , n):
    ul = [-2.0 , 1.0]
    nd = {"relu" : "RELU" , "linear" : "LIN"}
    name = n
    if n in nd:
        name = nd[n]
    svg = []
    svg = ['<?xml version="1.0"?>\n']
    svg += ["<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" viewBox=\"" + str(-1.0 - 3.0) + " 0.0 " + str(1.0 + 2.0) + " " + str(height + 2.0) + 
            "\"" + ">\n"]
    cx , cy = (-1.0 + 0) + ul[0] , 0.0 + ul[1]
    svg += ['<rect x="' + str(cx) + '" y="' + str(cy) + '" width="1.0" height="' + str(height) + '" fill="rgb(255 , 255 , 255)" stroke = "rgb(0,0,0)" stroke-width = "0.1"/>']
    svg += ['<text x = "' + str(cx + 0.5) + '" y = "' + str(cy + 0.5 * height) + '" style = "text-anchor : middle" font-size = "0.3" transform = "rotate(90 , ' + str(cx + 0.5) + ' , ' + str(cy + 0.5 * height) + ')">' + str(name) + '</text>']
    svg += ['<circle cx="' + str(ul[0] + 0.5) +  '" cy="' + str(ul[1] - 0.5 + height) + '" fill = "rgb(0 , 0 , 0)" r="0.25" />']
    svg += ['<circle cx="' + str(ul[0] + 0.5) +  '" cy="' + str(ul[1] - 0.5 + height) + '" fill = "rgb(255 , 255 , 255)" r="0.15" />']
    
    svg += ["</svg>\n"]

    return "\n".join(svg)

def make_html_home(path , log):
    html =  ["<!DOCTYPE html>"]
    html += ["<html>"]
    html += ["<head></head>"]
    html += ["<body>"]
    html += ["<p>"]
    html += ['<a href="element_0_0_.html" style = "text-decoration : none">FIRST_ELEMENT</a>']
    html += ["</p>"]
    html += ["<listing>"]
    html += [log] 
    html += ["</listing>"]
    html += ["</body>"]
    html += ["</html>"]

    with open(os.path.join(path , "index.html") , 'w') as f:
        f.write("\n".join(html))

def make_html(layer , trueLayer , width , height , 
        iteration = 0 , 
        nextLayer = None , 
        previousLayer = None , 
        largest = None ,
        smallest = None ,
        stdv = None ,
        difference = None ,
        previousIteration = None ,
        nextIteration = None):
    html =  ["<!DOCTYPE html>"]
    html += ["<html>"]
    html += ["<head></head>"]
    html += ["<body>"]
    html += ["<p>"]
    html += ['<a href="index.html" style = "text-decoration : none">HOME</a>']
    html += ["</p>"]
    html += ['<p>']
    if(previousLayer is not None):
        html += ['<a href="element_' + str(previousLayer) + '_' + str(iteration) + '_.html" style = "text-decoration : none">UP</a>']
    else:
        html += ['UP']
    if(nextLayer is not None):
        html += ['<a href="element_' + str(nextLayer) + '_' + str(iteration) + '_.html" style = "text-decoration : none">DOWN</a>']
    else:
        html += ['DOWN']
    html += ["</p>"]

    html += ['<p>']
    if(previousIteration is not None):
        html += ['<a href="element_' + str(layer) + '_' + str(previousIteration) + '_.html" style = "text-decoration : none">PREVIOUS_ITERATION</a>']
    else:
        html += ['PREVIOUS_ITERATION']
    if(nextIteration is not None):
        html += ['<a href="element_' + str(layer) + '_' + str(nextIteration) + '_.html" style = "text-decoration : none">NEXT_ITERATION</a>']
    else:
        html += ['NEXT_ITERATION']
    html += ["</p>"]
    
    html += ['<p> LAYER : ' + str(trueLayer) + '</p>']
    html += ['<p> INDEX : ' + str(layer) + '</p>']
    html += ['<p> ITERATION : ' + str(iteration) + '</p>']
    
    html += ['<p>ELEMENT:</p><img src = "element_' + str(layer) + '_' + str(iteration) + '_.svg" width = "' + str(width) + 'px" height = "' + str(height) + 'px" object-fit = "contain"/>']

    if(smallest is not None):
        html += ['<p> MIN : ' + str(smallest) + '</p>']
    if(largest is not None):
        html += ['<p> MAX : ' + str(largest) + '</p>']
    if(stdv is not None):
        html += ['<p> STD : ' + str(stdv) + '</p>']

    if difference is not None:
        p_smallest , p_largest , p_stdv = difference
        html += ['<p>CHANGE:</p><img src = "elementD_' + str(layer) + '_' + str(iteration) + '_.svg" width = "' + str(width) + 'px" height = "' + str(height) + 'px" object-fit = "contain"/>']
        if(p_smallest is not None):
            html += ['<p> MIN : ' + str(p_smallest) + '</p>']
        if(p_largest is not None):
            html += ['<p> MAX : ' + str(p_largest) + '</p>']
        if(p_stdv is not None):
            html += ['<p> STD : ' + str(p_stdv) + '</p>']
  
    html += ["</body>"]
    html += ["</html>"]

    return "\n".join(html)

def sequential_to_svg(model , path , iteration = 0 , previousIteration = None , nextIteration = None , pixelsPerME = 20.0):

    if not os.path.isdir(path):
        os.mkdir(path)

    model.save(os.path.join(path , "iteration_" + str(iteration)))

    config = model.get_config()
    a = [l['config']['activation'] for l in config['layers'] if 'activation' in l['config']]
    wb = [l.get_weights() for l in model.layers]
    w = [e[0] for e in wb]
    b = [e[1] for e in wb]
  
    if(not len(a) == len(w) == len(b)):
        print("Lengths of activations, weights and biases do not match.")
        return None

    previous = None
    p_a , p_w , p_b = None , None , None
    if previousIteration is not None:
        previous = keras.models.load_model(os.path.join(path , "iteration_" + str(iteration - 1)))
        p_config = previous.get_config()
        p_a = [l['config']['activation'] for l in config['layers'] if 'activation' in l['config']]
        p_wb = [l.get_weights() for l in previous.layers]
        p_w = [e[0] for e in p_wb]
        p_b = [e[1] for e in p_wb]

        if(not len(p_a) == len(p_w) == len(p_b) == len(a)):
            print("Lengths of activations, weights and biases do not match in previous layer.")
            return None
    
    layer = 0

    for i in range(len(a)):
        weights = w[i].transpose()
        bias = b[i].reshape((b[i].shape[0]) , 1)

        with open(os.path.join(path , "element_" + str(layer) + "_" + str(iteration) + "_.svg") , 'w') as f:
            svg = matrix_to_svg(weights , operation = "mul")
            f.write(svg)

        if previous is not None:
            with open(os.path.join(path , "elementD_" + str(layer) + "_" + str(iteration) + "_.svg") , 'w') as f:
                svg = matrix_to_svg(weights - p_w[i].transpose() , operation = "none")
                f.write(svg)

        with open(os.path.join(path , "element_" + str(layer) + "_" + str(iteration) + "_.html") , 'w') as f:
            prvL = layer - 1
            if(prvL < 0):
                prvL = None
            nxtL = layer + 1
            if previous is not None:
                dw = weights - p_w[i].transpose()
                html = make_html(layer , i , (weights.shape[1] + 3) * pixelsPerME , (weights.shape[0] + 2) * pixelsPerME , iteration = iteration , nextLayer = nxtL , 
                        previousLayer = prvL , smallest = np.min(weights) , largest = np.max(weights) , stdv = np.std(weights) ,
                        difference = (np.min(dw) , np.max(dw) , np.std(dw)),
                        previousIteration = previousIteration , nextIteration = nextIteration)
            else:
                html = make_html(layer , i , (weights.shape[1] + 3) * pixelsPerME , (weights.shape[0] + 2) * pixelsPerME , iteration = iteration , nextLayer = nxtL , 
                        previousLayer = prvL , smallest = np.min(weights) , largest = np.max(weights) , stdv = np.std(weights) ,
                        previousIteration = previousIteration , nextIteration = nextIteration)
            f.write(html)
        
        layer += 1

        with open(os.path.join(path , "element_" + str(layer) + "_" + str(iteration) + "_.svg") , 'w') as f:
            svg = matrix_to_svg(bias , operation = "add")
            f.write(svg)

        if previous is not None:
            with open(os.path.join(path , "elementD_" + str(layer) + "_" + str(iteration) + "_.svg") , 'w') as f:
                svg = matrix_to_svg(bias - p_b[i].reshape((b[i].shape[0]) , 1) , operation = "none")
                f.write(svg)
        
        with open(os.path.join(path , "element_" + str(layer) + "_" + str(iteration) + "_.html") , 'w') as f:
            prvL = layer - 1
            if(prvL < 0):
                prvL = None
            nxtL = layer + 1
            if previous is not None:
                dw = bias - p_b[i].reshape((b[i].shape[0]) , 1)
                html = make_html(layer , i , 4.0 * pixelsPerME , (bias.shape[0] + 2.0) * pixelsPerME , iteration = iteration , nextLayer = nxtL , 
                        previousLayer = prvL , smallest = np.min(bias) , largest = np.max(bias) , stdv = np.std(bias) ,
                        difference = (np.min(dw) , np.max(dw) , np.std(dw)),
                        previousIteration = previousIteration , nextIteration = nextIteration)
            else:
                html = make_html(layer , i , 4.0 * pixelsPerME , (bias.shape[0] + 2.0) * pixelsPerME , iteration = iteration , nextLayer = nxtL , 
                        previousLayer = prvL , smallest = np.min(bias) , largest = np.max(bias) , stdv = np.std(bias) ,
                        previousIteration = previousIteration , nextIteration = nextIteration)
            f.write(html)

        layer += 1

        with open(os.path.join(path , "element_" + str(layer) + "_" + str(iteration) + "_.svg") , 'w') as f:
            svg = activation_to_svg(bias.shape[0] , a[i])
            f.write(svg)
        
        with open(os.path.join(path , "element_" + str(layer) + "_" + str(iteration) + "_.html") , 'w') as f:
            prvL = layer - 1
            nxtL = layer + 1
            if(i == len(a) - 1):
                nxtL = None
            html = make_html(layer , i , 4.0 * pixelsPerME , (bias.shape[0] + 2.0) * pixelsPerME , iteration = iteration , nextLayer = nxtL , previousLayer = prvL ,
                        previousIteration = previousIteration , nextIteration = nextIteration)
            f.write(html)
        
        layer += 1

