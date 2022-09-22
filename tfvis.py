import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import json

def make_html_home(path , log , starting_node , last_iteration = None):
    if last_iteration is not None:
        html = ""
        with open(os.path.join(path , "index.html") , 'r') as f:
            html = f.read()
        i_html = []
        for i in range(last_iteration):
            i_html += ['<a href="' + starting_node + '_' + str(i) + '.html" style = "text-decoration : none">' + str(i) + '</a>']
        html = html.replace("REPLACE_ITERATIONS" , " ".join(i_html))            
        with open(os.path.join(path , "index.html") , 'w') as f:
            f.write(html)
    else:
        html =  ["<!DOCTYPE html>"]
        html += ["<html>"]
        html += ["<head></head>"]
        html += ["<body>"]
        html += ["<p>"]
        html += ['<a href="' + starting_node + '_0.html" style = "text-decoration : none">FIRST_ELEMENT_FIRST_ITERATION</a>']
        html += ["</p>"]
        html += ["<h1># ITERATIONS : </h1>"]
        html += ["<div width = \"500px\">REPLACE_ITERATIONS</div>"]
        html += ["<h1># CONFIGURATION : </h1>"]
        html += ["<listing>"]
        html += [log] 
        html += ["</listing>"]
        html += ["</body>"]
        html += ["</html>"]
        with open(os.path.join(path , "index.html") , 'w') as f:
            f.write("\n".join(html))

def matrix_to_svg(mm , operation = "mul" , transpose = False):
    """
    Produce svg of matrix.

    Arguments:

        m -                     Numpy array

    Optional arguments:

        operation -             "mul" , "add" mean matrix multiplication and matrix addition respectively
        transpose -             By default tensor flow weights are multiplied from the left.
                                If transpose = true, then the matrices will be transposed before turning into svg.

    Returns:

        str -                   Svg
    """
    m = mm
    if(transpose):
        m = mm.transpose()
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

def activation_to_svg(height , name , transpose = False):
    ul = [-2.0 , 1.0]
    svg = []

    if transpose:
        svg = ['<?xml version="1.0"?>\n']
        svg += ["<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" viewBox=\"" + str(-1.0 - 3.0) + " 0.0 " + str(1.0 + 2.0) + " " + str(height + 2.0) + 
                "\"" + ">\n"]
        cx , cy = (-1.0 + 0) + ul[0] , 0.0 + ul[1]
        svg += ['<rect x="' + str(cx) + '" y="' + str(cy) + '" width="1.0" height="' + str(height) + '" fill="rgb(255 , 255 , 255)" stroke = "rgb(0,0,0)" stroke-width = "0.1"/>']
        svg += ['<text x = "' + str(cx + 0.5) + '" y = "' + str(cy + 0.5 * height) + '" style = "text-anchor : middle" font-size = "0.3" transform = "rotate(90 , ' + str(cx + 0.5) + ' , ' + str(cy + 0.5 * height) + ')">' + str(name) + '</text>']
        svg += ['<circle cx="' + str(ul[0] + 0.5) +  '" cy="' + str(ul[1] - 0.5 + height) + '" fill = "rgb(0 , 0 , 0)" r="0.25" />']
        svg += ['<circle cx="' + str(ul[0] + 0.5) +  '" cy="' + str(ul[1] - 0.5 + height) + '" fill = "rgb(255 , 255 , 255)" r="0.15" />']
        
        svg += ["</svg>\n"]
    else:
        svg = ['<?xml version="1.0"?>\n']
        svg += ["<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" viewBox=\"" + str(-1.0 - 3.0) + " 0.0 " + str(height + 2.0) + " " + str(1.0 + 2.0) + " " + 
                "\"" + ">\n"]
        cx , cy = (-1.0 + 0) + ul[0] , 0.0 + ul[1]
        svg += ['<rect x="' + str(cx) + '" y="' + str(cy) + '" height="1.0" width="' + str(height) + '" fill="rgb(255 , 255 , 255)" stroke = "rgb(0,0,0)" stroke-width = "0.1"/>']
        svg += ['<text x = "' + str(cx + 0.5 * height) + '" y = "' + str(cy + 0.5) + '" style = "text-anchor : middle" font-size = "0.3">' + str(name) + '</text>']
        svg += ['<circle cx="' + str(cx + height + 0.5) +  '" cy="' + str(cy + 0.5) + '" fill = "rgb(0 , 0 , 0)" r="0.25" />']
        svg += ['<circle cx="' + str(cx + height + 0.5) +  '" cy="' + str(cy + 0.5) + '" fill = "rgb(255 , 255 , 255)" r="0.15" />']
        
        svg += ["</svg>\n"]

    return "\n".join(svg)

def get_nodes(model):
    """
    Get nodes of a tensor flow model.

    Arguments:

        model -                 tensof flow model

    Returns:

        (node_names , connections_before , connections_after , starting_layer)
    
    where
        
        node_names -            set of node names
        connections_before -    dictionary, connections_before['name'] returna a set of node names
                                that connect to node with name 'name'
        connections_after -     dictionary, connections_after['name'] returns a set of node names
                                that node with name 'name' connects to
        starting_layer -        first layer
    """
    config = model.get_config()
    names = set()
    connections_before = {}
    connections_after = {}
    starting_layer = None
    for l in config['layers']:
        start = None
        if 'name' in l:
            start = l['config']['name']
            if starting_layer is None:
                starting_layer = name
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

    if(len(connections_before) == 0 and len(connections_after) == 0):
        names = set()
        connections_before = {}
        connections_after = {}
        starting_layer = None
        previous = None
        for l in model.layers:
            l_config = l.get_config()
            current = l_config['name']
            if starting_layer is None:
                starting_layer = current
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

    return (names , connections_before , connections_after , starting_layer)

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
        width = 500 ,
        transpose = False ,
        otherinfo = "" ,
        addActivation = True):
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
        transpose -             By default tensor flow weights are multiplied from the left.
                                If transpose = true, then the matrices will be transposed before turning into svg.
        otherinfo -             Other information for iteration.
        addActivation -         Add activations svg.

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
        previous_iteration_model = keras.models.load_model(previous_iteration_path)
        p_names , p_connections_before , p_connections_after , p_starting_layer = get_nodes(previous_iteration_model)
        p_wb = get_wb(previous_iteration_model)
        p_info = get_info(previous_iteration_model)
    
    names , connections_before , connections_after , starting_layer = get_nodes(model)
    wb = get_wb(model)
    info = get_info(model)

    if iteration == 0:
        make_html_home(path , json.dumps(model.get_config() , indent = 2) , starting_layer)
    if last_iteration:
        make_html_home(path , json.dumps(model.get_config() , indent = 2) , starting_layer , 
                last_iteration = iteration)

    for n in names:
        
        n_path = os.path.join(path , n + "_" + str(iteration) + ".html")
        
        n_html =  ["<!DOCTYPE html>"]
        n_html += ["<html>"]
        n_html += ["<head></head>"]
        n_html += ["<body>"]

        n_html += ['<p><a href="index.html" style = "text-decoration : none">START</a></p>']
        
        n_html += ["<h1># NAVIGATION</h1>"]
    
        n_html += ["<p>- NODE : " + n + "</p>"]

        n_html += ['<p>- ITERATION : ' + str(iteration) + '</p>']

        n_html += ['<p>- PREVIOUS ITERATION : ']
        if(not first_iteration):
            n_previous_path = n + "_" + str(iteration - 1) + ".html"
            n_html += ['<a href="' + n_previous_path + '" style = "text-decoration : none">' + str(iteration - 1)  + '</a>']
        n_html += ["</p>"]

        n_html += ['<p>- NEXT ITERATION : ']
        if(not last_iteration):
            n_next_path = n + "_" + str(iteration + 1) + ".html"
            n_html += ['<a href="' + n_next_path + '" style = "text-decoration : none">' + str(iteration + 1)  + '</a>']
        n_html += ["</p>"]

        n_html += ["<p>- UP : "]
        if n in connections_before:
            for previous in connections_before[n]:
                previous_path = previous + "_" + str(iteration) + ".html"
                n_html += ['<a href="' + previous_path + '" style = "text-decoration : none">' + previous  + '</a>']
        n_html += ["</p>"]

        n_html += ["<p>- DOWN : "]
        if n in connections_after:
            for after in connections_after[n]:
                after_path = after + "_" + str(iteration) + ".html"
                n_html += ['<a href="' + after_path + '" style = "text-decoration : none">' + after  + '</a>']
        n_html += ["</p>"]

        activation_height = None

        arr_ind = 0
        for arr in wb[n]:
            oper = "none"
            if(len(arr.shape) == 2):
                oper = "mul"
                arr_svg = matrix_to_svg(arr , operation = oper , transpose = transpose)
                arr_path = n + "_arr_" + str(arr_ind) + "_" + str(iteration) + ".svg"
                with open(os.path.join(path , arr_path) , 'w') as f:
                    f.write(arr_svg)
                if not transpose:
                    n_html += ['<h1># WEIGHTS ' + str(arr_ind) + ':</h1><img src = "' +  arr_path + '" width = "' + str(width) + 'px" height = "' + str(width) + 'px" object-fit = "contain"/>']
                else:
                    n_html += ['<h1># WEIGHTS TRANSPOSED ' + str(arr_ind) + ':</h1><img src = "' +  arr_path + '" width = "' + str(width) + 'px" height = "' + str(width) + 'px" object-fit = "contain"/>']
                mina , maxa , stda = np.min(arr) , np.max(arr) , np.std(arr)
                n_html += ['<p>min , max , sigma = ' + str(mina) + " , " + str(maxa) + " , " + str(stda) + '</p>']

                if previous_iteration_model is not None:                
                    oper = "none"
                    darr = arr - p_wb[n][arr_ind]
                    arr_svg = matrix_to_svg(darr , operation = oper , transpose = transpose)
                    arr_path = n + "_darr_" + str(arr_ind) + "_" + str(iteration) + ".svg"
                    with open(os.path.join(path , arr_path) , 'w') as f:
                        f.write(arr_svg)
                    n_html += ['<h2>## DIFFERENCE FROM LAST ITERATION ' + str(arr_ind) + ':</h2><img src = "' +  arr_path + '" width = "' + str(width) + 'px" height = "' + str(width) + 'px" object-fit = "contain"/>']
                    mina , maxa , stda = np.min(darr) , np.max(darr) , np.std(darr)
                    n_html += ['<p>min , max , sigma = ' + str(mina) + " , " + str(maxa) + " , " + str(stda) + '</p>']

            elif(len(arr.shape) == 1):
                oper = "add"
                bias = arr.reshape((1 , arr.shape[0]))

                activation_height = arr.shape[0]

                arr_svg = matrix_to_svg(bias , operation = oper , transpose = transpose)
                arr_path = n + "_arr_" + str(arr_ind) + "_" + str(iteration) + ".svg"
                with open(os.path.join(path , arr_path) , 'w') as f:
                        f.write(arr_svg)
                if not transpose:
                    n_html += ['<h1># BIAS ' + str(arr_ind) + ':</h1><img src = "' +  arr_path + '" width = "' + str(width) + 'px" height = "' + str(width) + 'px" object-fit = "contain"/>']
                else:
                    n_html += ['<h1># BIAS TRANSPOSED ' + str(arr_ind) + ':</h1><img src = "' +  arr_path + '" width = "' + str(width) + 'px" height = "' + str(width) + 'px" object-fit = "contain"/>']
                mina , maxa , stda = np.min(arr) , np.max(arr) , np.std(arr)
                n_html += ['<p>min , max , sigma = ' + str(mina) + " , " + str(maxa) + " , " + str(stda) + '</p>']

                if previous_iteration_model is not None:                
                    oper = "none"
                    darr = bias - p_wb[n][arr_ind].reshape((1 , p_wb[n][arr_ind].shape[0]))
                    arr_svg = matrix_to_svg(darr , operation = oper , transpose = transpose)
                    arr_path = n + "_darr_" + str(arr_ind) + "_" + str(iteration) + ".svg"
                    with open(os.path.join(path , arr_path) , 'w') as f:
                        f.write(arr_svg)
                    n_html += ['<h2>## DIFFERENCE FROM LAST ITERATION ' + str(arr_ind) + ':</h2><img src = "' +  arr_path + '" width = "' + str(width) + 'px" height = "' + str(width) + 'px" object-fit = "contain"/>']
                    mina , maxa , stda = np.min(darr) , np.max(darr) , np.std(darr)
                    n_html += ['<p>min , max , sigma = ' + str(mina) + " , " + str(maxa) + " , " + str(stda) + '</p>']

            else:
                n_html += ['<p># ARRAY ' + str(arr_ind) + '</p>']
                n_html += ["<listing>"]
                n_html += [str(arr)] 
                n_html += ["</listing>"]
            arr_ind += 1

        if info[n]['activation'] is not None:
            n_html += ['<p>' + info[n]['activation'] + '</p>']
            if addActivation:
                ac_svg = activation_to_svg(activation_height , info[n]['activation'] , transpose = transpose)
                ac_path = n + "_carr_" + str(iteration) + ".svg"
                with open(os.path.join(path , ac_path) , 'w') as f:
                    f.write(ac_svg)
                if transpose: 
                    n_html += ['<h2># ACTIVATION TRANSPOSED :</h2><img src = "' +  ac_path + '" width = "' + str(width) + 'px" height = "' + str(width) + 'px" object-fit = "contain"/>']
                else:
                    n_html += ['<h2># ACTIVATION :</h2><img src = "' +  ac_path + '" width = "' + str(width) + 'px" height = "' + str(width) + 'px" object-fit = "contain"/>']
                n_html += ['<p>' + info[n]['activation'] + '</p>']
            else:
                n_html += ['<h1># ACTIVATION</h1><p>' + info[n]['activation'] + '</p>']



        n_html += ["<h1># NODE CONFIGURATION : </h1>"]
        n_html += ["<listing>"]
        n_html += [json.dumps(info[n] , indent = 2)] 
        n_html += ["</listing>"]
        n_html += ['<h1># OTHER </h1><listing>' + otherinfo + '</listing>']
        n_html += ["</body>"]
        n_html += ["</html>"]
        
        with open(n_path , 'w') as f:
            f.write("\n".join(n_html))



