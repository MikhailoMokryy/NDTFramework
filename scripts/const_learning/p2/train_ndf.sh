model="ndf"
tree_depth=8
n_tree=20
lr=1e-4
batch_size=64
epochs=40
experiment="const_learning_p2"
augmentation="gaussian_blur"
train_type="const_learning"
probability=0.2

# train model
python main.py --arch=${model} --epochs=${epochs} --tree-depth=${tree_depth} --n-tree=${n_tree} --batch-size=${batch_size} --lr=${lr} --experiment=${experiment} --aug=${augmentation} --train-type=${train_type} --apply_prob=${probability}
