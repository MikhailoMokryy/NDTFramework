model="nbdt"
model_path="./checkpoints/experiment1/nbdt/2025-03-31T21:55.pth"
lr=1e-1
epochs=40

python main.py --arch=${model} --lr=${lr} --epochs=${epochs} --hierarchy=induced-resnet18 --loss=SoftTreeSupLoss --model-path=${model_path} --eval
