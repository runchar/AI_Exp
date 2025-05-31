python main.py \
    --config configs/resnet_filter.yaml \
    --output_dir output \
    --mode train \
    > train.log 2>&1

shutdown -h now