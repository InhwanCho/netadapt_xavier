import torch
import torch.nn as nn
import nets as models
import functions as fns
from argparse import ArgumentParser

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



INPUT_DATA_SHAPE = (3, 224, 224)


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

if __name__ == '__main__':
    
    args = arg_parser.parse_args()
    print(args)
    NUM_CLASSES = args.num_classes
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
    print('-------------------------------------------')
    
    model = model.cuda()
    
    print('Building latency lookup table for', 
          torch.cuda.get_device_name())
    if build_lookup_table:
        '''
         `network_def`: defined in get_network_def_from_model()
            `lookup_table_path`: (string) path to save the file of lookup table
            `min_conv_feature_size`: (int) The size of feature maps of simplified layers (conv layer)
                along channel dimmension are multiples of 'min_conv_feature_size'.
                The reason is that on mobile devices, the computation of (B, 7, H, W) tensors 
                would take longer time than that of (B, 8, H, W) tensors.
            `min_fc_feature_size`: (int) The size of features of simplified FC layers are 
                multiples of 'min_fc_feature_size'.
            `measure_latency_batch_size`: (int) the batch size of input data
                when running forward functions to measure latency.
            `measure_latency_sample_times`: (int) the number of times to run the forward function of 
                a layer in order to get its latency.
            `verbose`: (bool) set True to display detailed information.
        '''
        fns.build_latency_lookup_table(network_def, lookup_table_path=lookup_table_path, 
            min_fc_feature_size=MIN_FC_FEATRE_SIZE, 
            min_conv_feature_size=MIN_CONV_FEATURE_SIZE, 
            measure_latency_batch_size=MEASURE_LATENCY_BATCH_SIZE,
            measure_latency_sample_times=MEASURE_LATENCY_SAMPLE_TIMES,
            verbose=True)
    print('-------------------------------------------')
    print('Finish building latency lookup table.')
    print('    Device:', torch.cuda.get_device_name())
    print('    Model: ', model_arch)    
    print('-------------------------------------------')
    
    latency = fns.compute_resource(network_def, 'LATENCY', lookup_table_path)
    print('Computed latency:     ', latency)
    latency = fns.measure_latency(model, 
        [MEASURE_LATENCY_BATCH_SIZE, *INPUT_DATA_SHAPE])
    print('Exact latency:        ', latency)    
    
    