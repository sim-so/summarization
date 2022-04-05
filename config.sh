# Read json file and convert tsv file
python ./preprocessing/preprocessing.py --dir 

# Train a model
python ./train.py --model_fn {} --train_fn {} --gpu_id 0 --batch_size 32 --lr 1e-4 --lr_step 0 --use_adam