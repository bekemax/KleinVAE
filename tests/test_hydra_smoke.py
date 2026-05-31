from pathlib import Path

from hydra import compose, initialize_config_dir

from main import train


def test_hydra_config_instantiates_and_runs_fast_dev_training(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    overrides = [
        "model=vanilla_vae",
        "callbacks=model_summary",
        "logger=csv",
        "+trainer.fast_dev_run=True",
        "trainer.max_epochs=1",
        "data.image_linear_pixel_size=4",
        "data.num_images=8",
        "data.batch_size=2",
        "data.persistence_subsample_size=2",
        "model.model.input_dim=16",
        "hidden_dims=[4]",
        "paths.root_dir=" + str(tmp_path),
        "paths.output_dir=" + str(tmp_path / "outputs"),
        "paths.work_dir=" + str(tmp_path),
        "paths.data_dir=" + str(tmp_path / "data"),
        "paths.log_dir=" + str(tmp_path / "logs"),
        "extras.enforce_tags=False",
        "extras.print_config=False",
    ]

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="main.yaml", overrides=overrides)

    metric_dict, objects = train(cfg)

    assert "val_loss" in metric_dict
    assert objects["datamodule"].train_dataset is not None
    assert objects["model"].model.latent_dim == 2
    assert len(objects["callbacks"]) == 1
    assert objects["trainer"].fast_dev_run
