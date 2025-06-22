model="nbdt"
model_path="./checkpoints/ckpt-nbdt-20250208161401.pth"

# test model
python main.py --arch=${model} --model-path=${model_path} --hierarchy=induced-resnet18 --loss=SoftTreeSupLoss --eval
