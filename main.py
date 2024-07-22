import matplotlib.pyplot as plt

from cnn import CNNPipeline
from hybird import HybridPipeline
from rnn import RNNPipeline


def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.close()


if __name__ == "__main__":
    train_path = "./dataset/train"
    val_path = "./dataset/validation"
    test_path = "./dataset/test"

    # CNN Pipeline
    print("Running CNN Pipeline...")
    cnn_pipeline = CNNPipeline()
    cnn_history, cnn_results = cnn_pipeline.run(train_path, val_path, test_path, "./models/cnn_model.h5")
    plot_training_history(cnn_history, "CNN")
    print("CNN Results:", cnn_results)
    print()

    # RNN Pipeline
    print("Running RNN Pipeline...")
    rnn_pipeline = RNNPipeline()
    rnn_history, rnn_results = rnn_pipeline.run(train_path, val_path, test_path, "./models/rnn_model.h5")
    plot_training_history(rnn_history, "RNN")
    print("RNN Results:", rnn_results)
    print()

    # Hybrid Pipeline
    print("Running Hybrid Pipeline...")
    hybrid_pipeline = HybridPipeline()
    hybrid_history, hybrid_results = hybrid_pipeline.run(train_path, val_path, test_path, "./models/hybrid_model.h5")
    plot_training_history(hybrid_history, "Hybrid")
    print("Hybrid Results:", hybrid_results)
    print()

    # Compare results
    models = ["CNN", "RNN", "Hybrid"]
    results = [cnn_results, rnn_results, hybrid_results]

    print("Model Comparison:")
    for model, result in zip(models, results):
        print(f"{model}:")
        for metric, value in result.items():
            print(f"  {metric}: {value:.4f}")
        print()

    # Find the best model based on F1 score
    best_model = max(zip(models, results), key=lambda x: x[1]['f1'])
    print(f"Best model based on F1 score: {best_model[0]}")