# Train a path-integrating RNN

To train a path-integrating RNN, take the following steps:

1. Set the appropriate parameters in the `config.yaml` file.

2. Run the following command to train the model:

```bash
python main.py --config config.yaml
```

3. The script will create a run ID (based on the time of the run) and save the trained model in `logs/checkpoints/<run_id>/`.

## Analyze the trained model

The notebook `analyze.ipynb` shows how to load the trained model, and provides some visualizations.








