model="nbdt"
model_path="./checkpoints/experiment1/nbdt/2025-04-01T14:11.pth"
lr=1e-1
epochs=40

python main.py --arch=${model} --lr=${lr} --epochs=${epochs} --hierarchy=induced-resnet18 --loss=HardTreeSupLoss --model-path=${model_path} --eval
