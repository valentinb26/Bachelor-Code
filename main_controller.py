# plan_executor.py
import time
import threading
import warnings
import os
import logging

from manipulation.robot_device import DobotController
from perception.vision_detector import VisionDetector
from display.window_display import WindowDisplay
from utils.utils import load_plan_from_json, strip_instance
from config.logging_config import setup_logging
from config.config import LOGS_DIR, SLOT_POSITIONS, PLAN_PATH, OBJECT_CONFIG

# Ignoriere nervige Warnings die ich eh nicht ändern kann
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

setup_logging(LOGS_DIR)
logger = logging.getLogger(__name__)

class PlanExecutor:
    def __init__(self):
        logger.info("Starte System-Initialisierung...")
        
        # Alle eindeutigen Prompts aus der Config sammeln, um die GDINO zu geben. 
        #self.master_prompt = " . ".join(set(OBJECT_PROMPT_MAP.values()))
        prompts = {
            config["prompt"] 
            for config in OBJECT_CONFIG.values() 
            if config.get("prompt")  # Filtert leere Strings ("") und None heraus
        }
        
        # Wichtig: " . " damit die einzelnen Prompts nicht zusammenzählen!
        self.master_prompt = " . ".join(prompts)
        logger.debug(f"Master Prompt: {self.master_prompt}")
        
        # Subsysteme starten
        self.detector = VisionDetector(target_prompt=self.master_prompt)
        self.robot = DobotController()
        self.window = WindowDisplay(self.detector)
        
        logger.info("System bereit für Plan-Ausführung!")

    
    def _get_prompt_for_object(self, object_name: str):
        """Ermittelt den Suchprompt aus der Objekt-Konfiguration für einen Objekt-Namen."""
        base_name = strip_instance(object_name)
        # Holt das Config-Dict für das Objekt, standardmäßig ein leeres Dict {}
        obj_config = OBJECT_CONFIG.get(base_name, {})
        # Holt den Prompt aus der Config (Fallback auf None, falls Key nicht existiert)
        return obj_config.get("prompt")
    
    def _get_fixed_position_for_object(self, object_name: str):
        """Ermittelt die feste Position aus der Objekt-Konfiguration für einen Objekt-Namen."""
        base_name = strip_instance(object_name)
        obj_config = OBJECT_CONFIG.get(base_name, {})
        return obj_config.get("pos")  # Gibt das Tupel (x, y, z) oder None zurück


    def execute_plan(self, steps: list) -> bool:
        """Arbeitet eine Liste von Schritten nacheinander ab."""

        blacklist = ("blue_cover_", "side_cover", "bumper")

        for i, (action_name, obj, slots) in enumerate(steps, 1):
            logger.info(f"[{i:02d}] {action_name}({obj} : {slots[0] if slots else 'None'})")


            # macht gerade zu viel ärger erstmal skippen
            if any(keyword in obj for keyword in blacklist):
                logger.warning(f"Überspringe '{obj}' erstmal...")
                continue

            # Prompt nur bei pick setzen
            if action_name == "pick":
                expected_label = self._get_prompt_for_object(obj)
                if expected_label:
                    # später ggf relevant
                    #self.detector.set_prompt(expected_label)
                    logger.info(f"Suche gerade nach: {expected_label}")

            try:        
                input("      Enter zum Ausfuehren, Ctrl+C zum Abbrechen...\n") 

                if action_name == "pick":
                    success = self._handle_pick(obj)
                    
                elif action_name == "place":
                    success = self._handle_place(obj, slots[0] if slots else None)
                    
                elif action_name == "screw":
                    success = self._handle_screw(obj, slots[1] if len(slots) > 1 else None)
                    
                elif action_name in ("place_blue_cover", "place_side_cover_right", "place_side_cover_left"):
                    success = self._handle_human_action()
                    
                else:
                    logger.warning(f"Unbekannte Aktion '{action_name}', wird uebersprungen.")
                    success = True   

                if not success:
                    raise RuntimeError("Schnittstellen-Rückmeldung war negativ (False).")
            except Exception as e:
                logger.error(f"Aktion fehlgeschlagen: {type(e).__name__}: {e}")
                cont = input("      Trotzdem fortfahren? (j/n): ")
                if cont.lower() != "j":
                    logger.warning("Ausführung durch Nutzer abgebrochen.")
                    self.shutdown()
                    return False
                
            time.sleep(0.5)  # Kurze Pause zwischen den Schritten
            
        logger.info("Der gesamte Plan wurde erfolgreich ausgeführt! 🎉")
        self.shutdown()
        return True

    # ---------------------------------------------------------------------------
    # AKTIONSHANDLER (Schnittstelle zwischen Vision und Robot)
    # ---------------------------------------------------------------------------
    
    def _handle_pick(self, object_name: str) -> bool:
        """Lässt den Roboter das Objekt greifen (entweder über feste Position oder Kamera)."""
        
        # 1. Prüfen, ob eine feste Position hinterlegt ist
        fixed_pos = self._get_fixed_position_for_object(object_name)
        
        if fixed_pos is not None:
            logger.info(f"Feste Position für '{object_name}' gefunden: {fixed_pos}")
            # Nutze die x- und y-Koordinaten aus dem 3D-Tupel (x, y, z)
            robot_x, robot_y = fixed_pos[0], fixed_pos[1], fixed_pos[2]
            
        else:
            # 2. Keine feste Position -> Kamera-Erkennung nutzen
            logger.info(f"Keine feste Position für '{object_name}'. Starte Kamera-Erkennung...")
            
        # 3. Prompt ermitteln (z.B. "blue_cover_1" -> "blue_cover" -> "blue plastic cover")
        expected_label = self._get_prompt_for_object(object_name)
        
        if not expected_label:
            logger.error(f"Kein Prompt-Mapping für '{strip_instance(object_name)}' in config.py gefunden.")
            return False
            
        # 4. Kamera nach Koordinaten fragen (holt x, y in Roboter-mm)
        coords = self.detector.get_robot_coordinates_for_object(expected_label)
        if coords is None:
            return False
            
        # 5. Physischen Pick ausführen
        robot_x, robot_y = coords
        return self.robot.pick(robot_x, robot_y)

    def _handle_place(self, object_name: str, slot_name: str) -> bool:
        """Lässt den Roboter das gegriffene Objekt an einem Slot ablegen."""
        slot_pos = SLOT_POSITIONS.get(slot_name)
        return self.robot.place(object_name, slot_name, slot_pos)

    def _handle_screw(self, screw_name: str, target_name: str) -> bool:
        """Führt den Schraubvorgang an der Position des Ziel-Objekts aus."""
        return self.robot.screw(screw_name, target_name)
    
    def _handle_human_action(self, object_name:str, slot_name:str) -> bool:
        #todo:
        return False

    def shutdown(self):
        """Räumt alle Ressourcen sauber auf."""
        logger.info("Fahre Systeme herunter...")
        self.detector.shutdown()
        self.window.stop()

        try:
            self.robot.park_and_disconnect()
        except Exception as e:
            logger.error(f"Dobot-Home fehlgeschlagen: {e}")
        
        logger.info("Alle Systeme sauber heruntergefahren.")
        os._exit(1)


# ===============================================================================
# HAUPTPROGRAMM (Beispielhafte Nutzung)
# ===============================================================================
if __name__ == "__main__":
    executor = PlanExecutor()
    try:
        #steps = parse_plan(PLAN_TEXT)
        steps = load_plan_from_json(PLAN_PATH)
        logger.info(f"{len(steps)} Schritte geladen.")
        # Die Plan-Ausführung (inkl. input()-Bestätigung) läuft im Hintergrund-Thread.
        # Der Main-Thread bleibt frei für das Fenster, da cv2-Fenster nur dort laufen dürfen.
        exec_thread = threading.Thread(target=executor.execute_plan, args=(steps,), daemon=True)
        exec_thread.start()
 
        # Blockierende Fenster-Schleife im Main-Thread
        executor.window.run()

        # Blockierende Fenster-Schleife im Main-Thread. Endet über 'q' oder shutdown().
        logger.info("Fenster geschlossen, fahre System herunter")
        executor.shutdown()
 
    except KeyboardInterrupt:
        logger.info("Ctrl+C erkannt, fahre herunter...")
        executor.shutdown()
    except Exception as e:
        logger.error(f"Fehler bei der Planausführung: {e}", exc_info=True)
        executor.shutdown()