echo 'test mobilenet v2'
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 master.py -working_folder models_imagenet/mobilenetv2/prune-by-latency -input_data_shape 3 224 224 \
    -im models_imagenet/mobilenetv2/model.pth.tar -gp 0 1 2 3 \
    -mi 30 -bur 0.25 -rt LATENCY  -irr 0.025 -rd 0.96 \
    -lr 0.001 -st 500 -lt latency_lut/mobilenet_v2.pkl \
    --dataset_path /home/nvidia/netapapt/mount/ILSVRC2012_img_val --arch mobilenetv2 --dataset imagenet