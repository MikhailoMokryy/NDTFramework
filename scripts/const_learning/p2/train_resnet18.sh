model="resnet18"
lr=1e-1
epochs=40
experiment="const_learning_p2"
augmentation="gaussian_blur"
train_type="const_learning"
probability=0.2

# training model
python main.py --arch=${model} --lr=${lr} --epochs=${epochs} --experiment=${experiment} --aug=${augmentation} --train-type=${train_type} --apply_prob=${probability}
