"""Read-only health check for configured market data sources and local cache."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from core.config import load_config


def _model_dict(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def check_health(config_path: str) -> dict:
    config = load_config(config_path)
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(config_path).resolve()),
        "mysql": {},
        "ddb": {},
        "cache": {},
    }

    mysql_config = config.data.mysql
    if mysql_config is None:
        result["mysql"] = {"status": "not_configured"}
    else:
        from data.mysql_source import MySQLSource

        source = MySQLSource(_model_dict(mysql_config))
        result["mysql"]["endpoint_count"] = len(source._endpoints)
        try:
            source.engine
            result["mysql"].update({
                "status": "ok",
                "active_endpoint": source.active_endpoint_name,
            })
        except Exception as exc:
            result["mysql"].update({
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "message": str(exc),
            })

    ddb_config = config.data.ddb
    if ddb_config is None or not ddb_config.host:
        result["ddb"] = {"status": "not_configured"}
    else:
        try:
            import dolphindb  # noqa: F401
        except ImportError:
            result["ddb"] = {
                "status": "client_not_installed",
                "error_type": "ImportError",
            }
        else:
            from data.ddb_source import DDBSource

            source = DDBSource(_model_dict(ddb_config))
            connection = None
            try:
                connection = source._get_connection()
                value = connection.run("1 + 1")
                result["ddb"] = {
                    "status": "ok" if int(value) == 2 else "unexpected_response"
                }
            except Exception as exc:
                result["ddb"] = {
                    "status": "unavailable",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass

    cache_path = Path(config.data.cache.get("path", "./cache"))
    files = list(cache_path.glob("*.parquet")) if cache_path.is_dir() else []
    result["cache"] = {
        "status": "ok" if files else "empty",
        "path": str(cache_path.resolve()),
        "parquet_files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "latest_modified": (
            datetime.fromtimestamp(
                max(path.stat().st_mtime for path in files), timezone.utc
            ).isoformat()
            if files else None
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--strict", action="store_true", help="Exit non-zero if configured sources fail"
    )
    args = parser.parse_args()

    result = check_health(args.config)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")

    if args.strict:
        statuses = [result["mysql"].get("status"), result["ddb"].get("status")]
        if any(status == "unavailable" for status in statuses):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
