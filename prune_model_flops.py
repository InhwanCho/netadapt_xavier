import torch
import torch.nn as nn
import nets as models
import functions_postech as fns
from argparse import ArgumentParser
import os
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))





'''
    `MIN_CONV_FEATURE_SIZE`: The sampled size of feature maps of layers (conv layer)
        along channel dimmension are multiples of 'MIN_CONV_FEATURE_SIZE'.
        
    `MIN_FC_FEATURE_SIZE`: The sampled size of features of FC layers are 
        multiples of 'MIN_FC_FEATURE_SIZE'.
'''
MIN_CONV_FEATURE_SIZE = 8
MIN_FC_FEATRE_SIZE    = 64

'''
    `MEASURE_LATENCY_BATCH_SIZE`: the batch size of input data
        when running forward functions to measure latency.
    `MEASURE_LATENCY_SAMPLE_TIMES`: the number of times to run the forward function of 
        a layer in order to get its latency.
'''
MEASURE_LATENCY_BATCH_SIZE = 128
MEASURE_LATENCY_SAMPLE_TIMES = 500


arg_parser = ArgumentParser(description='Build latency lookup table')
arg_parser.add_argument('--dir', metavar='DIR', default='latency_lut/lut_alexnet.pkl',
                    help='path to saving lookup table')
arg_parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
arg_parser.add_argument('--num_classes', type=int, default=1000)
arg_parser.add_argument('--load_dir', type=str, default='models/', dest='save_dir', 
                            help='path to save models (default: models/')
if __name__ == '__main__':
    
    args = arg_parser.parse_args()
    print(args)
    NUM_CLASSES = args.num_classes
    if NUM_CLASSES == 100:
        INPUT_DATA_SHAPE = (3, 32, 32)
    else:
        INPUT_DATA_SHAPE = (3, 224, 224)
    build_lookup_table = True
    lookup_table_path = args.dir
    model_arch = args.arch
    
    print('Load', model_arch)
    print('--------------------------------------')
    model = models.__dict__[model_arch](num_classes=NUM_CLASSES)
    
    '''
        return network def (OrderedDict) of the input model
        
        network_def only contains information about FC, Conv2d, ConvTranspose2d
        not includes batchnorm ...
        Output:
            `network_def`: (OrderedDict)
                           keys(): layer name (e.g. model.0.1, feature.2 ...)
                           values(): layer properties (dict)
    '''
    network_def = fns.get_network_def_from_model(model, INPUT_DATA_SHAPE)
    for layer_name, layer_properties in network_def.items():
        print(layer_name)
        print('    ', layer_properties, '\n')
    print('-------------------------------------------')
    
    num_w = fns.compute_resource(network_def, 'WEIGHTS')
    flops = fns.compute_resource(network_def, 'FLOPS')
    num_param = fns.compute_resource(network_def, 'WEIGHTS')
    print('Number of FLOPs:      ', flops)
    print('Number of weights:    ', num_w)
    print('Number of parameters: ', num_param)
    #print(model)
    model = model.cuda()
    latency = fns.compute_resource(network_def, 'LATENCY', lookup_table_path)
    print('Computed latency:     ', latency)
    latency = fns.measure_latency(model, 
        [MEASURE_LATENCY_BATCH_SIZE, *INPUT_DATA_SHAPE])
    print('Exact latency:        ', latency)   
    print('-------------------------------------------')
    

    filename = os.path.join(args.save_dir)
    model = torch.load(filename) 
    network_def = fns.get_network_def_from_model(model, INPUT_DATA_SHAPE)
    for layer_name, layer_properties in network_def.items():
        print(layer_name)
        print('    ', layer_properties, '\n')
    print('-------------------------------------------')
    
    num_w = fns.compute_resource(network_def, 'WEIGHTS')
    flops = fns.compute_resource(network_def, 'FLOPS')
    num_param = fns.compute_resource(network_def, 'WEIGHTS')
    print('Number of FLOPs:      ', flops)
    print('Number of weights:    ', num_w)
    print('Number of parameters: ', num_param)
    print('-------------------------------------------')

    model = model.cuda()
    latency = fns.compute_resource(network_def, 'LATENCY', lookup_table_path)
    print('Computed latency:     ', latency)
    latency = fns.measure_latency(model, 
        [MEASURE_LATENCY_BATCH_SIZE, *INPUT_DATA_SHAPE])
    print('Exact latency:        ', latency)    
