import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import json

def make_html_home(path , log , starting_node):
    html =  ["<!DOCTYPE html>"]
    html += ["<html>"]
    html += ["<head></head>"]
    html += ["<body>"]
    html += ["<p>"]
    html += ['<a href="' + starting_node + '_0.html" style = "text-decoration : none">FIRST_ELEMENT</a>']
    html += ["</p>"]
    html += ["<listing>"]
    html += [log] 
    html += ["</listing>"]
    html += ["</body>"]
    html += ["</html>"]

    with open(os.path.join(path , "index.html") , 'w') as f:
        f.write("\n".join(html))

def matrix_to_svg(m , operation = "mul"):
    """
    Produce svg of matrix.

    Arguments:

        m -                     Numpy array

    Optional arguments:

        operation -             "mul" , "add" mean matrix multiplication and matrix addition respectively

    Returns:

        str -                   Svg
    """
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

def get_nodes(model):
    """
    Get nodes of a tensor flow model.

    Arguments:

        model -                 tensof flow model

    Returns:

        (node_names , connections_before , connections_after)
    
    where
        
        node_names -            set of node names
        connections_before -    dictionary, connections_before['name'] returna a set of node names
                                that connect to node with name 'name'
        connections_after -     dictionary, connections_after['name'] returns a set of node names
                                that node with name 'name' connects to
    """
    config = model.get_config()
    names = set()
    connections_before = {}
    connections_after = {}
    for l in config['layers']:
        start = None
        if 'name' in l:
            start = l['name']
            names.add(start)
        if 'inbound_nodes' in l:
            for lev0 in l['inbound_nodes']:
                for lev1 in lev0:
                    for lev2 in lev1:
                        if isinstance(lev2 , str):
                            names.add(lev2)
                            
                            if start in connections_before:
                                connections_before[start].add(lev2)
                            else:
                                connections_before.update({start : {lev2}})
                            
                            if lev2 in connections_after:
                                connections_after[lev2].add(start)
                            else:
                                connections_after.update({lev2 : {start}})

    if(len(connections_before) == 0 and len(connections_after) == 0 and len(names) == 0):
        previous = None
        for l in model.layers:
            l_config = l.get_config()
            current = l_config['name']
            names.add(current)
            if(previous is not None):
                if current in connections_before:
                    connections_before[current].add(previous)
                else:
                    connections_before.update({current : {previous}})
                
                if previous in connections_after:
                    connections_after[previous].add(current)
                else:
                    connections_after.update({previous : {current}})
            previous = current

    if(len(names) == 0):
        raise ValueError('No node names in model.')
    return (names , connections_before , connections_after)

def get_wb(model):
    """
    Get weights and biases for model.

    Arguments:

        model -                 tensof flow model

    Returns:

        wb -                    dictionary, wb['name'] returns a list of weights and
                                biases for node with name 'name'
    """
    weights = {}
    
    for l in model.layers:
        l_config = l.get_config()
        if 'name' not in l_config:
            raise ValueError('The key "name" is not in model.get_config()["layers"]')
        name = l_config['name']
        wb = l.get_weights()
        weights.update({name : wb})
    
    return weights

def get_info(model):
    """
    Get information for model.

    Arguments:

        model -                 tensof flow model

    Returns:

        info -                  dictionary, info['name'] returns the configuration
                                for node with name 'name'
    """
    weights = {}
    
    for l in model.layers:
        l_config = l.get_config()
        if 'name' not in l_config:
            raise ValueError('The key "name" is not in model.get_config()["layers"]')
        name = l_config['name']
        weights.update({name : l_config})
    
    return weights

def save_visualization(path , model , iteration = 0 , 
        first_iteration = False , 
        last_iteration = False ,
        width = 500):
    """
    Save model visualization.

    Arguments:

        path -                  Path to directory for the saved files.
        model -                 Tensor flow model.
    
    Optional arguments:

        iteration -             Iteration / epoch number.
        first_iteration -       Set to true if this is the first iteration / epoch to save.
        last_iteration -        Set this to true if this is the last iteration / epoch to save.
        width -                 Pixel width for images.

    """
    if not os.path.isdir(path):
        os.mkdir(path)

    model.save(os.path.join(path , "iteration_" + str(iteration)))

    previous_iteration_model = None
    p_names , p_connections_before , p_connections_after = None , None , None
    p_wb = None
    p_info = None

    previous_iteration_path = os.path.join(path , 'iteration_' + str(iteration - 1))
    if os.path.isdir(previous_iteration_path):
        previous_iteration_model = keras.models.load(previous_iteration_model)
        p_names , p_connections_before , p_connections_after = get_nodes(previous_iteration_model)
        p_wb = get_wb(previous_iteration_model)
        p_info = get_info(previous_iteration_model)
    
    names , connections_before , connections_after = get_nodes(model)
    wb = get_wb(model)
    info = get_info(model)

    for n in names:
        
        n_path = os.path.join(path , n + "_" + str(iteration) + ".html")
        
        n_html =  ["<!DOCTYPE html>"]
        n_html += ["<html>"]
        n_html += ["<head></head>"]
        n_html += ["<body>"]

        n_html += ["<p>NODE : " + n + "</p>"]

        n_html += ['<p>ITERATION : ' + str(iteration) + '</p>']

        n_html += ['<p>PREVIOUS ITERATION : ']
        if(not first_iteration):
            n_previous_path = os.path.join(path , n + "_" + str(iteration - 1) + ".html")
            n_html += ['<a href="' + n_previous_path + '" style = "text-decoration : none">' + str(iteration - 1)  + '</a>']
        n_html += ["</p>"]

        n_html += ['<p>NEXT ITERATION : ']
        if(not last_iteration):
            n_next_path = os.path.join(path , n + "_" + str(iteration + 1) + ".html")
            n_html += ['<a href="' + n_next_path + '" style = "text-decoration : none">' + str(iteration + 1)  + '</a>']
        n_html += ["</p>"]

        n_html += ["<p>UP : "]
        if n in connections_before:
            for previous in connections_before[n]:
                previous_path = previous + "_" + str(iteration) + ".html"
                n_html += ['<a href="' + previous_path + '" style = "text-decoration : none">' + previous  + '</a>']
        n_html += ["</p>"]

        n_html += ["<p>DOWN : "]
        if n in connections_after:
            for after in connections_after[n]:
                after_path = after + "_" + str(iteration) + ".html"
                n_html += ['<a href="' + after_path + '" style = "text-decoration : none">' + after  + '</a>']
        n_html += ["</p>"]

        arr_ind = 0
        for arr in wb[n]:
            oper = "none"
            if(len(arr.shape) == 2):
                oper = "mul"
                arr_svg = matrix_to_svg(arr , operation = oper)
                arr_path = n + "_arr_" + str(arr_ind) + "_" + str(iteration) + ".svg"
                with open(os.path.join(path , arr_path) , 'w') as f:
                    f.write(arr_svg)
                n_html += ['<p>WEIGHTS ' + str(arr_ind) + ':</p><img src = "' +  arr_path + '" width = "' + str(width) + 'px" height = "' + str(width) + 'px" object-fit = "contain"/>']
                mina , maxa , stda = np.min(arr) , np.max(arr) , np.std(arr)
                n_html += ['<p>min , max , standard deviation = ' + str(mina) + " , " + str(maxa) + " , " + str(stda) + '</p>']

                if previous_iteration_model is not None:                
                    oper = "none"
                    darr = arr - p_wb[n][arr_ind]
                    arr_svg = matrix_to_svg(darr , operation = oper)
                    arr_path = n + "_darr_" + str(arr_ind) + "_" + str(iteration) + ".svg"
                    with open(os.path.join(path , arr_path) , 'w') as f:
                        f.write(arr_svg)
                    mina , maxa , stda = np.min(darr) , np.max(darr) , np.std(darr)
                    n_html += ['<p>min , max , standard deviation = ' + str(mina) + " , " + str(maxa) + " , " + str(stda) + '</p>']

            elif(len(arr.shape) == 1):
                oper = "add"
                bias = arr.reshape((1 , arr.shape[0]))
                arr_svg = matrix_to_svg(bias , operation = oper)
                arr_path = n + "_arr_" + str(arr_ind) + "_" + str(iteration) + ".svg"
                with open(os.path.join(path , arr_path) , 'w') as f:
                        f.write(arr_svg)
                n_html += ['<p>BIAS ' + str(arr_ind) + ':</p><img src = "' +  arr_path + '" width = "' + str(width) + 'px" height = "' + str(width) + 'px" object-fit = "contain"/>']
                mina , maxa , stda = np.min(arr) , np.max(arr) , np.std(arr)
                n_html += ['<p>min , max , standard deviation = ' + str(mina) + " , " + str(maxa) + " , " + str(stda) + '</p>']

                if previous_iteration_model is not None:                
                    oper = "none"
                    darr = bias - p_wb[n][arr_ind].reshape((1 , p_wb[n][arr_ind].shape[0]))
                    arr_svg = matrix_to_svg(darr , operation = oper)
                    arr_path = n + "_darr_" + str(arr_ind) + "_" + str(iteration) + ".svg"
                    with open(os.path.join(path , arr_path) , 'w') as f:
                        f.write(arr_svg)
                    mina , maxa , stda = np.min(darr) , np.max(darr) , np.std(darr)
                    n_html += ['<p>min , max , standard deviation = ' + str(mina) + " , " + str(maxa) + " , " + str(stda) + '</p>']

            else:
                n_html += ['<p>ARRAY ' + str(arr_ind) + '</p>']
                n_html += ["<listing>"]
                n_html += [str(arr)] 
                n_html += ["</listing>"]
            arr_ind += 1
        n_html += ["<p>NODE CONFIGURATION : </p>"]
        n_html += ["<listing>"]
        n_html += [json.dumps(info[n] , indent = 2)] 
        n_html += ["</listing>"]
        n_html += ["</body>"]
        n_html += ["</html>"]
        
        with open(n_path , 'w') as f:
            f.write("\n".join(n_html))



