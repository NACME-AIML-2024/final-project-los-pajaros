import argparse
import yaml
import datetime
import os
from models.gan_model import GanModel

def generate_experiment_id():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")

def create_experiment_folder(base_dir, experiment_id, experiment_name):
    folder_name = f"{experiment_id}_{experiment_name.replace(' ', '_')}"
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def load_hyperparameters(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description="Run experiments with unique IDs.")
    parser.add_argument("model", type=str, choices=["GanModel", "model_b"], help="Model name")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    # Load and update YAML
    config = load_hyperparameters(args.config)
    experiment_id = generate_experiment_id()
    config["experiment"]["id"] = experiment_id

    # Save updated YAML in the experiment folder
    experiment_name = config["experiment"]["name"]
    experiment_folder = create_experiment_folder("experiments", experiment_id, experiment_name)
    updated_config_path = os.path.join(experiment_folder, "hyperparameters.yaml")
    with open(updated_config_path, "w") as file:
        yaml.safe_dump(config, file)

    print(f"Running Experiment: {experiment_name} (ID: {experiment_id})")
    print(f"Results will be saved in: {experiment_folder}")

    # Initialize and run the model
    if args.model == "GanModel":
        model = GanModel(config["hyperparameters"], save_dir=experiment_folder)
    else:
        raise ValueError("Model not implemented")

    model.train()
    model.test()

if __name__ == "__main__":
    main()
