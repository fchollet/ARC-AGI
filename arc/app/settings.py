import dataclasses


@dataclasses.dataclass
class Settings:
    N: int = 100
    # folder: str = "./data/training"
    folder: str = "../ARC/data/training"
    cache_ttl: int = 3600  # Seconds until st.cache expiration
    log_level: int = 30
    grid_width: int = 10
    grid_height: int = 4
