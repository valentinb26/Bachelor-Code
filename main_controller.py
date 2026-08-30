# plan_executor.py
import time
import threading
import warnings
import os
import logging

from manipulation.robot_device import DobotController
from perception.vision_detector import VisionDetector
from display.window_display import WindowDisplay
from utils.utils import load_plan_from_json, strip_instance, get_object_config
from config.logging_config import setup_logging
from config.config import LOGS_DIR, SLOT_POSITIONS, PLAN_PATH, OBJECT_CONFIG, Z_GROUND

# Ignoriere nervige Warnings die ich eh nicht ändern kann
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

setup_logging(LOGS_DIR)
logger = logging.getLogger(__name__)

class MainController:
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

    # Hilfsfunktionen um auf Objekt-Konfiguration zuzugreifen
    def _get_prompt_for_object(self, object_name: str):
        """Ermittelt den Suchprompt aus der Objekt-Konfiguration für einen Objekt-Namen."""
        return get_object_config(object_name).get("prompt")
    
    def _get_fixed_position_for_object(self, object_name: str):
        """Ermittelt die feste Position aus der Objekt-Konfiguration für einen Objekt-Namen."""
        return get_object_config(object_name).get("pos")  # Gibt das Tupel (x, y, z) oder None zurückj
    
    def _get_object_height_for_object(self, object_name: str):
        """Ermittelt die feste Position aus der Objekt-Konfiguration für einen Objekt-Namen."""
        return get_object_config(object_name).get("height")  # Gibt die Höhe in mm zurück

    # weitere Hilfsfunktionen
    def _check_side_switch(self, slots: list, last_side: str) -> str:
        """
        Prüft, ob ein Seitenwechsel vorliegt, fordert ggf. zum Drehen auf
        und gibt die aktuelle Seite zurück.
        """
        if not slots:
            return last_side

        target_slot = str(slots[0])
        current_side = None

        if "slot_right" in target_slot:
            current_side = "right"
        elif "slot_left" in target_slot:
            current_side = "left"

        # Wechsel erkennen und Nutzer auffordern
        if last_side and current_side and current_side != last_side:
            logger.warning(f"🔄 SEITENWECHSEL ERKANNT: Wechsel von '{last_side}' auf '{current_side}'!")
            input(
                f"\n=========================================================\n"
                f"  ACHTUNG: Bitte das Auto JETZT auf die andere Seite drehen!\n"
                f"  (Wechsel von {last_side.upper()} zu {current_side.upper()})\n"
                f"=========================================================\n"
                f"  Drücke ENTER, sobald das Auto gedreht wurde..."
            )

        # Gibt die neue Seite zurück (oder die alte, falls kein Side-Slot vorlag)
        return current_side if current_side else last_side
    
    def _is_human_object(self, obj: str) -> bool:
        """Prüft, ob das Objekt anhand seines Namens an den Menschen delegiert werden muss."""
        human_objects = ("blue_cover_", "side_cover", "bumper")
        return any(keyword in obj for keyword in human_objects)
    
    def _timed_robot_call(self, func, *args, **kwargs):
        """Führt eine Roboter-Aktion aus und misst die reine Ausführungszeit,
        auch wenn währenddessen eine Exception geworfen wird."""
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            self._robot_time_ms += (time.perf_counter() - t0) * 1000

# ---------------------------------------------------------------------------
# HAUPTFUNKTION
# ---------------------------------------------------------------------------

    def execute_plan(self, steps: list) -> bool:
        """Arbeitet eine Liste von Schritten nacheinander ab."""
        t_start_full = time.perf_counter()# Messung
        self._robot_time_ms = 0.0  # NEU
        
        
        last_side = None
        i = 1
        
        while i <= len(steps):
            action_name, obj, slots = steps[i - 1]
        #for i, (action_name, obj, slots) in enumerate(steps, 1):
            logger.info(f"[{i:02d}] {action_name}({obj} : {slots[0] if slots else 'None'})")

            # Seitenwechsel prüfen und Zustand aktualisieren
            last_side = self._check_side_switch(slots, last_side)

            try:        
                input("      Enter zum Ausfuehren, Ctrl+C zum Abbrechen...\n") 

                if action_name == "pick":
                    if self._is_human_object(obj):
                        success = self._handle_human_action(obj, slots[0] if slots else None)
                    else:
                        #success = self._handle_pick(obj)
                        success = self._timed_robot_call(self._handle_pick, obj)
                    
                elif action_name == "place":
                    if self._is_human_object(obj):
                        success = self._handle_human_action(obj, slots[0] if slots else None)
                    else:
                        #success = self._handle_place(obj, slots[0] if slots else None)
                        success = self._timed_robot_call(self._handle_place, obj, slots[0] if slots else None)

                    
                elif action_name == "screw":
                    #success = self._handle_screw(obj, slots[1] if len(slots) > 1 else None)
                    success = self._timed_robot_call(self._handle_screw, obj, slots[1] if len(slots) > 1 else None)
                    
                elif action_name in ("place_blue_cover", "place_side_cover_right", "place_side_cover_left"):
                    
                    success = self._handle_human_action(obj, slots[0] if slots else None)
                    
                else:
                    logger.warning(f"Unbekannte Aktion '{action_name}', wird uebersprungen.")
                    success = True   

                if not success:
                    raise RuntimeError("Schnittstellen-Rückmeldung war negativ (False).")
                
                # Nach success oder Exception: frag nach Wiederholung
                repeat = input("      Wiederholen? (j/n): ")
                if repeat.lower() == "j":
                    continue  # i bleibt gleich
                
                i += 1  # Nur wenn erfolgreich oder "j"

            except Exception as e:
                logger.error(f"Aktion fehlgeschlagen: {type(e).__name__}: {e}")
                cont = input("      Wiederholen (w) / Trotzdem fortfahren (j) / Abbrechen (n): ")
                if cont.lower() == "w":
                    continue  # i bleibt gleich, nächste Loop-Iteration
                elif cont.lower() != "j":
                    logger.warning("Ausführung durch Nutzer abgebrochen.")
                    self.shutdown()
                    return False
            
            
                
            time.sleep(0.5)  # Kurze Pause zwischen den Schritten
            
        logger.info("Der gesamte Plan wurde erfolgreich ausgeführt! 🎉")
        full_time = (time.perf_counter() - t_start_full) * 1000 # Messung
        logger.info(f"Gesamtzeit dieses Durchlaufs: {full_time:.2f} ms")
        logger.info(f"Reine Roboterzeit (pick/place/screw): {self._robot_time_ms:.2f} ms")
        self.shutdown()
        return True

# ---------------------------------------------------------------------------
# AKTIONSHANDLER (Schnittstelle zwischen Vision und Robot)
# ---------------------------------------------------------------------------
    
    def _handle_pick(self, object_name: str) -> bool:
        """Lässt den Roboter das Objekt greifen (entweder über feste Position oder Kamera)."""
        # 1. Prüfen, ob eine feste Position hinterlegt ist
        fixed_pos = self._get_fixed_position_for_object(object_name)
        
        if fixed_pos is None:
            # 2. Keine feste Position -> Kamera-Erkennung nutzen
            logger.info(f"Keine feste Position für '{object_name}'. Starte Kamera-Erkennung...")
            
            # 3. Prompt ermitteln (z.B. "blue_cover_1" -> "blue_cover" -> "blue plastic cover")
            expected_label = self._get_prompt_for_object(object_name)
            
            if not expected_label:
                logger.error(f"Kein Prompt-Mapping für '{strip_instance(object_name)}' in config.py gefunden.")
                return False
            
            logger.info(f"Suche gerade nach: {expected_label}")
                
            # 4. Kamera nach Koordinaten fragen (holt x, y in Roboter-mm)
            coords = self.detector.get_robot_coordinates_for_object(expected_label)
            if coords is None:
                return False
            
            robot_x, robot_y = coords
            robot_z = Z_GROUND + self._get_object_height_for_object(object_name)
            robot_r = 0 # Default-Rotation
            
        else:
            # Feste Position nutzen
            logger.info(f"Feste Position für '{object_name}' gefunden: {fixed_pos}")
            robot_x, robot_y, robot_z, robot_r = fixed_pos
            
        # 5. Physischen Pick ausführen
        if robot_z is None:
            return self.robot.pick(robot_x, robot_y)
        else:
            return self.robot.pick(robot_x, robot_y, robot_z, robot_r)

    def _handle_place(self, object_name: str, slot_name: str) -> bool:
        """Lässt den Roboter das gegriffene Objekt an einem Slot ablegen."""
        slot_pos = SLOT_POSITIONS.get(slot_name)    
        # Objekt-Höhe ermitteln
        obj_height = self._get_object_height_for_object(object_name)
        
        return self.robot.place(object_name, slot_name, slot_pos, object_height=obj_height)

    def _handle_screw(self, screw_name: str, target_slot: str) -> bool:
        """Führt den Schraubvorgang an der Position des Ziel-Objekts aus."""
        object_height = self._get_object_height_for_object(screw_name)
        return self.robot.screw(screw_name, object_height, target_slot)
    
    def _handle_human_action(self, object_name:str, slot_name:str) -> bool:
        # 0. Prüfen ob slot übergeben wurde
        if slot_name is not None:
            slot_pos = SLOT_POSITIONS.get(slot_name)
            robot_x, robot_y = slot_pos[:2]
        else:
            # 1. Prüfen, ob eine feste Position hinterlegt ist
            fixed_pos = self._get_fixed_position_for_object(object_name)
            
            if fixed_pos is not None:
                logger.info(f"Feste Position für '{object_name}' gefunden: {fixed_pos}")
                robot_x, robot_y = fixed_pos[0], fixed_pos[1]
                
            else:
                # 2. Keine feste Position -> Kamera-Erkennung nutzen
                logger.info(f"Keine feste Position für '{object_name}'. Starte Kamera-Erkennung...")
                
                # 3. Prompt ermitteln (z.B. "blue_cover_1" -> "blue_cover" -> "blue plastic cover")
                expected_label = self._get_prompt_for_object(object_name)
            
                if not expected_label:
                    logger.error(f"Kein Prompt-Mapping für '{strip_instance(object_name)}' in config.py gefunden.")
                    return False
                
                logger.info(f"Suche gerade nach: {expected_label}")
                
                # 4. Kamera nach Koordinaten fragen (holt x, y in Roboter-mm)
                coords = self.detector.get_robot_coordinates_for_object(expected_label, center=True)
                if coords is None:
                    return False
            
                robot_x, robot_y = coords
            
        return self.robot.human_action(robot_x, robot_y)

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
# HAUPTPROGRAMM 
# ===============================================================================
if __name__ == "__main__":
    controller = MainController()
    try:
        #steps = parse_plan(PLAN_TEXT)
        steps = load_plan_from_json(PLAN_PATH)
        logger.info(f"{len(steps)} Schritte geladen.")
        # Die Plan-Ausführung (inkl. input()-Bestätigung) läuft im Hintergrund-Thread.
        # Der Main-Thread bleibt frei für das Fenster, da cv2-Fenster nur dort laufen dürfen.
        exec_thread = threading.Thread(target=controller.execute_plan, args=(steps,), daemon=True)
        exec_thread.start()
 
        # Blockierende Fenster-Schleife im Main-Thread
        controller.window.run()

        # Blockierende Fenster-Schleife im Main-Thread. Endet über 'q' oder shutdown().
        logger.info("Fenster geschlossen, fahre System herunter")
        controller.shutdown()
 
    except KeyboardInterrupt:
        logger.info("Ctrl+C erkannt, fahre herunter...")
        controller.shutdown()
    except Exception as e:
        logger.error(f"Fehler bei der Planausführung: {e}", exc_info=True)
        controller.shutdown()