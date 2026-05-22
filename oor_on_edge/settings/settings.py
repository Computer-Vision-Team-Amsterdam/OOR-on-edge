from cvtoolkit.settings.settings_helper import GenericSettings, Settings
from pydantic import BaseModel

from oor_on_edge.settings.settings_schema import OOROnEdgeSettingsSpec


class OOROnEdgeSettings(Settings):  # type: ignore
    @classmethod
    def set_from_yaml(
        cls, filename: str, spec: BaseModel = OOROnEdgeSettingsSpec
    ) -> "GenericSettings":
        return super().set_from_yaml(filename, spec)
