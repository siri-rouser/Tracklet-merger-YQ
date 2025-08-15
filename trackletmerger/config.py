from typing import List, Tuple, Dict
from pathlib import Path

from pydantic import BaseModel, Field, conlist
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated
from visionlib.pipeline.settings import LogLevel, YamlConfigSettingsSource


class RedisConfig(BaseModel):
    host: str = 'localhost'
    port: Annotated[int, Field(ge=1, le=65536)] = 6379
    stream_id: str = 'aggregate'
    input_stream_prefix: str = 'objecttracker'
    output_stream_prefix: str = 'tracklet-merger'


Size2D = Tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]

class MergingConfig(BaseModel):
    input_stream_ids: conlist(str)
    output_stream_id: str
    matching_metric: str
    dis_thre: float
    dis_remove: float
    dis_alpha: float
    dis_beta: float
    lost_time: int
    searching_time: int
    original_img_size: Dict[str, Size2D] = Field(default_factory=dict)
    zone_data: Dict[str, Path] = Field(default_factory=dict)
    clm_path: Path = Path('cam_pair.json')
    clm_bandwidth: float = 5.0 
    frame_window: int = 2000  # in frames
    overlap_frames: int = 1000  # in frames

class SCTMergingConfig(BaseModel):
    max_frame_gap: int = 100
    max_pixel_distance: int = 500
    cosine_threshold: float = 0.1

class TrackletMergerConfig(BaseSettings):
    log_level: LogLevel = LogLevel.WARNING
    redis: RedisConfig = RedisConfig()
    prometheus_port: Annotated[int, Field(ge=1024, le=65536)] = 8000
    merging_config: MergingConfig
    sct_merging_config: SCTMergingConfig = SCTMergingConfig()
    save_directory: Path = Path('./results')

    model_config = SettingsConfigDict(env_nested_delimiter='__')

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls), file_secret_settings)