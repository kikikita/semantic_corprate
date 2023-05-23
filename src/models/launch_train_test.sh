echo "Training"

python3 train_model.py -ml_config train_config.json

echo "Inference"

python3 test_model.py -ml_config test_config.json

echo "Script finished"