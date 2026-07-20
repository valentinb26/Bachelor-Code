# config/logging_config.py
import logging
from datetime import datetime
from pathlib import Path


def setup_logging(logs_dir: Path = None):
    """
    Konfiguriert das globale Logging-System.
    
    Args:
        logs_dir: Pfad zum Logs-Verzeichnis. Falls None, wird './logs' verwendet.
    """
    if logs_dir is None:
        logs_dir = Path("./logs")
    
    # Logs-Verzeichnis erstellen, falls nicht vorhanden
    logs_dir.mkdir(exist_ok=True)
    
    # 1. Basis-Konfiguration für Console-Output
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s [%(name)s] [%(threadName)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # 2. Root-Logger auf DEBUG setzen, damit er DEBUG-Events verarbeitet
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Console-Handler auf INFO beschränken (DEBUG nur in Datei)
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.INFO)
    
    # 3. Datei-Handler hinzufügen (DEBUG + alles)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = logs_dir / f"{timestamp}_all_output.log"
    
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] [%(threadName)s] %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    # Handler an Root-Logger hängen (gilt für alle Module)
    root_logger.addHandler(file_handler)
    
    root_logger.info(f"Logging konfiguriert. Logdatei: {log_file_path}")