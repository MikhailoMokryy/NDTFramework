model="ndf"
tree_depth=8
n_tree=20
lr=1e-4
batch_size=64
epochs=40
experiment="inc_learning"
augmentation="gaussian_blur"
train_type="inc_learning"

# train model
python main.py --arch=${model} --epochs=${epochs} --tree-depth=${tree_depth} --n-tree=${n_tree} --batch-size=${batch_size} --lr=${lr} --experiment=${experiment} --aug=${augmentation} --train-type=${train_type}
