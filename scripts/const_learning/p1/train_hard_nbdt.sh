model="hard_nbdt"
lr=1e-1
epochs=40
augmentation="gaussian_blur"
experiment="const_learning_p1"
train_type="const_learning"
probability=0.1

# training model
python main.py --arch=${model} --lr=${lr} --epochs=${epochs} --hierarchy=induced-resnet18 --loss=HardTreeSupLoss --experiment=${experiment} --aug=${augmentation} --train-type=${train_type} --apply_prob=${probability}
