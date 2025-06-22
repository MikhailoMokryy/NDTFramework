model="nbdt"
model_path="./checkpoints/const_learning_p8/nbdt/model-2025-05-04.pth"
augmentation="gaussian_blur"

echo "Pure:"
python main.py --arch=${model} --hierarchy=induced-resnet18 --loss=SoftTreeSupLoss --model-path=${model_path} --eval

echo
echo "Aug:"
python main.py --arch=${model} --hierarchy=induced-resnet18 --loss=SoftTreeSupLoss --model-path=${model_path} --eval --aug=${augmentation}
