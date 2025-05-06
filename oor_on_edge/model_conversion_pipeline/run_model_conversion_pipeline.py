from datetime import datetime

from oor_on_edge.model_conversion_pipeline.components.model_conversion import (
    run_model_conversion,
)
from oor_on_edge.settings.luna_logging import setup_luna_logging
from oor_on_edge.settings.settings import OOROnEdgeSettings


def main():
    settings = OOROnEdgeSettings.set_from_yaml("config.yml")
    logging_file_path = f"{settings['logging']['luna_logs_dir']}/model_conversion_pipeline/{datetime.now().strftime('%y%m%d-%H%M%S')}.txt"
    setup_luna_logging(settings["logging"], logging_file_path)

    run_model_conversion(
        pretrained_model_path=settings["detection_pipeline"]["pretrained_model_path"],
        model_name=settings["detection_pipeline"]["model_name"],
        image_size=settings["detection_pipeline"]["output_image_size"],
        model_save_path=settings["detection_pipeline"]["pretrained_model_path"],
    )


if __name__ == "__main__":
    main()
