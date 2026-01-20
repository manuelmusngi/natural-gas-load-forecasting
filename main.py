from src.pipeline.forecasting_pipeline import ForecastingPipeline
from src.utils.config_parser import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    data_cfg = load_config("config/data_config.yaml")
    model_cfg = load_config("config/model_config.yaml")
    train_cfg = load_config("config/training_config.yaml")

    pipeline = ForecastingPipeline(
        data_config=data_cfg,
        model_config=model_cfg,
        training_config=train_cfg,
    )

    logger.info("Starting full forecasting pipeline...")
    pipeline.run()
    logger.info("Pipeline completed.")

if __name__ == "__main__":
    main()
