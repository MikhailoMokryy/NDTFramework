model="nbdt"
model_path="./checkpoints/const_learning/nbdt/2025-04-10T11:29.pth"

# training model
python main.py --arch=${model} --hierarchy=induced-resnet18 --loss=SoftTreeSupLoss --model-path=${model_path} --eval
