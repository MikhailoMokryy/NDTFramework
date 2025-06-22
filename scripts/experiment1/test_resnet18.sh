model="resnet18"
model_path="./checkpoints/experiment1/resnet18/2025-03-27T11:37.pth"
lr=1e-1
epochs=40

# training model
python main.py --arch=${model} --lr=${lr} --epochs=${epochs} --model-path=${model_path} --eval
