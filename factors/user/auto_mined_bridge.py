"""Default-empty registration bridge for immutable mined-factor snapshots."""
from factor_mining.bridge import register_snapshot_from_environment


REGISTERED_MINED_FACTORS = register_snapshot_from_environment()


__all__ = ["REGISTERED_MINED_FACTORS"]
