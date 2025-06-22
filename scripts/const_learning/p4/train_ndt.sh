model="ndt"
tree_depth=8
lr=1e-4
batch_size=64
epochs=40
experiment="const_learning"
augmentation="gaussian_blur"
train_type="const_learning"

# train model
python main.py --arch=${model} --epochs=${epochs} --tree-depth=${tree_depth} --batch-size=${batch_size} --lr=${lr} --experiment=${experiment} --aug=${augmentation} --train-type=${train_type}
