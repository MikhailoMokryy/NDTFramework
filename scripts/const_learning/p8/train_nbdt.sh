model="nbdt"
lr=1e-1
epochs=40
experiment="const_learning_p8"
augmentation="gaussian_blur"
train_type="const_learning"
probability=0.8

# training model
python main.py --arch=${model} --lr=${lr} --epochs=${epochs} --hierarchy=induced-resnet18 --loss=SoftTreeSupLoss --experiment=${experiment} --aug=${augmentation} --train-type=${train_type} --apply_prob=${probability}
