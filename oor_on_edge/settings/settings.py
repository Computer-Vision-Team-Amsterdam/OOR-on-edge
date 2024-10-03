from cvtoolkit.settings.settings_helper import GenericSettings, Settings
from oor_on_edge.settings.settings_schema import OOROnEdgeSettingsSpec
from pydantic import BaseModel


class OOROnEdgeSettings(Settings):  # type: ignore
    @classmethod
    def set_from_yaml(
        cls, filename: str, spec: BaseModel = OOROnEdgeSettingsSpec
    ) -> "GenericSettings":
        return super().set_from_yaml(filename, spec)
