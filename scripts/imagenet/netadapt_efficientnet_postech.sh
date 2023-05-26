echo 'efficientnet postech'
CUDA_VISIBLE_DEVICES=4,5,6,7 python master.py -working_folder models_imagenet/efficientnet_es_postech/prune-by-latency -input_data_shape 3 224 224 \
    -im models_imagenet/efficientnet_es_postech/model.pth.tar -gp 4 5 6 7 \
    -mi 30 -bur 0.25 -rt LATENCY  -irr 0.025 -rd 0.96 \
    -lr 0.001 -st 500 -lt latency_lut/lut_efficientnet_es_postech.pkl \
    --dataset_path /home/nvidia/netapapt/mount/ILSVRC2012_img_val --arch efficientnet_es_postech --dataset imagenet