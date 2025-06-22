model="nbdt"
model_path="./checkpoints/inc_learning/nbdt/model-2025-05-18.pth"
augmentation="gaussian_blur"

echo "Pure:"
python main.py --arch=${model} --hierarchy=induced-resnet18 --loss=SoftTreeSupLoss --model-path=${model_path} --eval

echo
echo "Aug:"
python main.py --arch=${model} --hierarchy=induced-resnet18 --loss=SoftTreeSupLoss --model-path=${model_path} --eval --aug=${augmentation}
