## environment

conda
```bash
conda env create -f environment.yml
```

## training
The dataset should be placed in the `./data` directory, and you can also modify the dataset path in `./configs/vit.yaml`.
```bash
bash run_train.sh
```