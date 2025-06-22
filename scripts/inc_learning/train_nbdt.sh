model="nbdt"
lr=1e-1
epochs=40
experiment="inc_learning"
augmentation="gaussian_blur"
train_type="inc_learning"

# training model
python main.py --arch=${model} --lr=${lr} --epochs=${epochs} --hierarchy=induced-resnet18 --loss=SoftTreeSupLoss --experiment=${experiment} --aug=${augmentation} --train-type=${train_type}
