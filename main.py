"""End-to-end training and evaluation pipeline for ADA-COVID.

Usage:
    python main.py
"""

from ada_covid.config import SEED, get_default_config
from ada_covid.data import load_and_preprocess_datasets
from ada_covid.evaluation import evaluate_model
from ada_covid.models import build_all_models, build_inference_model
from ada_covid.training import stage2_train, train
from ada_covid.utils import print_environment_info, set_seed
from ada_covid.visualization import plot_training_history


def main() -> None:
    set_seed(SEED)
    print_environment_info()

    config = get_default_config()

    print("Loading and preprocessing datasets...")
    dataset = load_and_preprocess_datasets(config)
    config["num_classes"] = dataset["num_classes"]
    print(f"Number of classes: {dataset['num_classes']}")

    print("Building models (Stage 1: triplet embedding + adversarial domain)...")
    models = build_all_models(config)
    models["combined_model"].summary(line_length=90)

    print(f"Starting training: {config['num_iterations']} iterations...")
    history = train(config, models, dataset)
    print("Training complete.")

    plot_training_history(history)

    print("Building Stage 2 inference model...")
    inference_model = build_inference_model(models["combined_classifier"], config)
    inference_model.summary()

    print("Fine-tuning softmax head (Stage 2)...")
    stage2_train(inference_model, dataset, seed=SEED, epochs=30)

    print("Evaluating on SOURCE dataset (SARS-CoV-2 CT)...")
    evaluate_model(
        inference_model,
        dataset["source_data"],
        dataset["source_label"],
        dataset_name="Source (SARS-CoV-2 CT)",
    )

    print("Evaluating on TARGET dataset (COVID-19 CT, without retraining)...")
    evaluate_model(
        inference_model,
        dataset["target_data"],
        dataset["target_label"],
        dataset_name="Target (COVID-19 CT)",
    )

    inference_model.save("ADA_COVID_inference_model.keras")
    models["combined_classifier"].save("ADA_COVID_embedding_model.keras")
    print("Models saved.")


if __name__ == "__main__":
    main()
