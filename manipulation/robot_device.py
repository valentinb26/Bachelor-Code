# robot_device.py
# Kapselt die Hardware-Ansteuerung des Roboterarms.
import sys
import time
import os
import logging

logger = logging.getLogger(__name__)

from pydobotplus import Dobot
from serial.tools import list_ports 
from config.config import (
    Z_GROUND, 
    Z_SAFE, 
    Z_LIFT, 
    Z_DIP,
    DOBOT_RETRIES, 
    DOBOT_TIMEOUT,
    SCREW_ROTATIONS
    )

class DobotController:
    def __init__(self):
        # Initialisiert den Dobot unter Verwendung von init_robot()
        # Setzt den Roboter in die Ausgangslage (Home)
        self.dobot = None
        self.placed_positions = {} 

        self._init_robot()


    def _init_robot(self):
        """
        Findet den seriellen Port und initialisiert den Dobot inkl. Homing.
        """
        available_ports = list_ports.comports()
        if not available_ports:
            logger.critical("Kein serieller Port für den Roboter gefunden!")
            sys.exit(1)
            
        logger.info(f"Verfügbare Ports: {[x.device for x in available_ports]}")
        port_idx = min(2, len(available_ports) - 1)
        port = available_ports[port_idx].device

        # --- Retry-Logik Start ---
        dobot = None

        for versuch in range(1, DOBOT_RETRIES + 1):
            try:
                logger.info(f"Verbindungsversuch {versuch} von {DOBOT_RETRIES} auf Port {port}...")
                dobot = Dobot(port=port)

                # Ein kurzer Testbefehl, um zu sehen, ob die Kommunikation wirklich steht
                dobot.clear_alarms()
                dobot.suck(False)

                logger.info("Verbindung zum Dobot erfolgreich hergestellt.")
                break 

            except Exception as e:
                logger.warning(f"Versuch {versuch} fehlgeschlagen: {e}")
                if versuch < DOBOT_RETRIES:
                    logger.info(f"Warte {DOBOT_TIMEOUT} Sekunden vor dem nächsten Versuch...")
                    time.sleep(DOBOT_TIMEOUT)
                else:
                    logger.critical("Dobot konnte nach mehreren Versuchen nicht erreicht werden.")
                    os._exit(1)

        _homing(dobot)
        
        logger.info("Dobot bereit.")
        self.dobot = dobot

    def park_and_disconnect(self):
        """
        Versetzt den Roboter in eine sichere Parkposition, deaktiviert alle 
        Aktuatoren (Greifer, Sauger) und schließt die serielle Verbindung.
        
        Sollte vor dem Ausschalten der Hardware oder beim Beenden des 
        Programms aufgerufen werden.
        """
        try:
            self.dobot.clear_alarms()
            self.dobot.grip(enable=False)
            self.dobot.suck(False)
            self.dobot.move_to(200,0,100, wait=True)
            self.dobot.move_to(200,0,0, wait=True)
            self.dobot.close()
            logger.info("Dobot erfolgreich geparkt und Verbindung geschlossen.")

        except Exception as e:
            logger.error(f"Fehler beim Parken und Trennen des Roboters: {e}")
        

    def pick(self, robot_x: float, robot_y: float, robot_z = Z_GROUND, robot_r = 0) -> bool:
        """
        Roboter hebt ein Objekt auf.
        """
        logger.info(f"Pick Objekt bei Position ({robot_x}, {robot_y}, {robot_z}, {robot_r})")
        try:
            self.dobot.clear_alarms()
            self.dobot.grip(enable=False)
            self.dobot.move_to(robot_x, robot_y, Z_SAFE, robot_r)
            time.sleep(0.5)
            self.dobot.move_to(robot_x, robot_y, robot_z, robot_r)
            self.dobot.grip(enable=True)
            time.sleep(0.5)
            self.dobot.move_rel(0, 0, Z_LIFT, 0)
            
            _check_alarms(self.dobot, robot_x, robot_y, robot_z)
                
            return True
        except Exception as e:
            logger.error(f"Pick-Aktion fehlgeschlagen: {e}")
        return False


    def place(self, object_name: str, slot_name: str, slot_pos: tuple, object_height: float) -> bool:
        # Holt Koordinaten aus SLOT_POSITIONS und legt das Objekt dort ab
        #coords = SLOT_POSITIONS.get(slot_name)
        if slot_pos is None:
            logger.error(f"Keine Koordinaten fuer Slot '{slot_name}' in config definiert.")
            return False

        x, y, z = slot_pos[:3]    
        # Basis-Höhe festlegen: Wenn Slot-Z = 0, dann Z_GROUND, sonst die Slot-Höhe
        base_z = z if z != 0 else Z_GROUND
        
        # Finale Zielhöhe ist die Basishöhe PLUS die Objekt-Höhe
        target_z = base_z + object_height
        
        # Sicherheitshöhe liegt immer dynamisch über dem Ablagepunkt
        safe_z = max(Z_SAFE, target_z + 20)

        try:
            logger.info(f"   -> Platziere '{object_name}' auf Slot '{slot_name}' bei ({x}, {y}, Z={target_z})")
            self.dobot.move_to(x, y, safe_z)
            time.sleep(0.5)
            self.dobot.move_to(x, y, target_z)
            self.dobot.grip(enable=False)
            time.sleep(0.5)
            self.dobot.move_rel(0, 0, Z_LIFT, 0)
            self.dobot.suck(False)
    
            # Speichere den Slot auf welchem etwas platziert wurde
            self.placed_positions[slot_name] = (x, y, target_z)
            
            _check_alarms(self.dobot, x, y, target_z)
            
            _idle(self.dobot)
            
            return True
        except Exception as e:
            logger.error(f"Place-Aktion fehlgeschlagen: {e}")
        return False
  

    def screw(self, object_name: str, object_height: float, target_slot: str) -> bool:
        # Erhält name des Objekts und den Slot auf welchem es platziert werden soll
        # Holt die Position des Slots aus self.placed_positions und führt Schraubvorgang aus
        position = self.placed_positions.get(target_slot)
        if position is None:
            logger.error(f"Keine bekannte Position fuer Slot '{target_slot}'.")
            return False

        x, y, z = position
        
        try:
            logger.info(f"   -> Schraube '{object_name}' in '{target_slot}' bei ({x:.1f}, {y:.1f}, {z:.1f})")
            # todo:
            # x - offset der schraube = mitte der radnarbe. Wenn auto immer senkrecht platziert wird
            # sollte es passen
            # BZW move_rel um x offsset wie im dobotlab! (12.5mm)
            # quasi die place aktion
            self.dobot.move_to(x, y, Z_SAFE)
            time.sleep(0.5)
            self.dobot.move_to(x, y, z + object_height)
            self.dobot.grip(enable=False)
            time.sleep(0.2)
            self.dobot.suck(False)
            time.sleep(0.5)

            # kleine schraub animation
            self.dobot.move_rel(0, 0, -10, r=45)
            for _ in range(SCREW_ROTATIONS):
                self.dobot.move_rel(r=-90)
                self.dobot.move_rel(r=90)
            self.dobot.move_rel(r=-45)
            
            _check_alarms(self.dobot, x, y, z)
            
            _idle(self.dobot)
            
            return True
        except Exception as e:
            logger.error(f"Schrauben-Aktion fehlgeschlagen: {e}")
            return False
        
    def human_action(self, robot_x: float, robot_y: float) -> bool:
        """
        Roboter zeigt eine HRC Animation an.
        """
        try:
            self.dobot.clear_alarms()

            # 1. Anfahren über das Zielbauteil
            self.dobot.move_to(robot_x, robot_y, Z_SAFE)
            time.sleep(0.1)

            # 2. Auf Zeige-Höhe absenken
            self.dobot.move_to(robot_x, robot_y, Z_SAFE)
            
            # 3. Greifer VOLL ÖFFNEN als visuelle Freigabe ("Hier zugreifen")
            self.dobot.grip(enable=False)
            time.sleep(0.2)

            # 4. Doppeltes Nicken (Aufmerksamkeit des Werkers auf das Bauteil lenken)
            for _ in range(2):
                self.dobot.move_rel(0, 0, Z_DIP, 0)
                time.sleep(0.15)
                self.dobot.move_rel(0, 0, -Z_DIP, 0)
                time.sleep(0.15)

            # 5. Kurze Denkpause für die Wahrnehmung des Werkers
            time.sleep(0.5)

            # 6. Roboter zieht sich auf Z_SAFE zurück (Mensch hat Platz zum Greifen)
            self.dobot.move_to(robot_x, robot_y, Z_SAFE)
            self.dobot.suck(False)
            
            
            _check_alarms(self.dobot, robot_x, robot_y, Z_SAFE)
            
            _idle(self.dobot)
                
            return True
        except Exception as e:
            logger.error(f"Werker-Aktion fehlgeschlagen: {e}")
        return False
  
# Interne Hilfsmethoden
  
def _homing(dobot):
    # --- Homing procedure ---
    logger.debug("User wird gefragt ob er die homing procedure ausführen will. Es wird gewartet")
    cont = input("      Homing durchführen? (j/n): ")
    
    if cont.lower() == "j":
        logger.info("Arm wird gehomed, bitte warten...")
        dobot.clear_alarms()
        dobot.grip(enable=False)
        dobot.suck(False)
        dobot.move_rel(0, 0, 100, 0)
        # damit homing auch funktioniert falls rel move ein problem hatte
        dobot.clear_alarms()
        home_cmd_index = dobot.home()
        dobot.wait_for_cmd(home_cmd_index)
        logger.debug("Homing procedure beendet.")

    # Positionier den Roboter damit er nicht im Bild ist
    _idle(dobot)
        
def _idle(dobot):
    # Roboter fährt an eine Stelle in welcher er nicht im Weg steht
    try:
        pose = dobot.get_pose()
        x, y, z, r = pose.position
        z_diff = Z_SAFE - z
        
        dobot.clear_alarms()
        dobot.grip(enable=False)
        dobot.suck(False)
        dobot.move_rel(0, 0, z_diff, 0)
        dobot.move_to(-140,-210,140,0)
    except Exception as e:
        logger.error(f"Fehler bei der Idle-Aktion: {e}")
        
def _check_alarms(dobot, x, y, z):
    """Prüft, ob das Dobot ein Alarm hatte und löscht es, falls ja."""
    alarms = dobot.get_alarms()
    
    if alarms:
        logger.error("Alarm-Status: Fehler bei der Aktion!")
        logger.error(f"Position: ({x:.1f}, {y:.1f}, {z:.1f})")
        logger.error(f"Alarm-Status: {dobot.get_alarms()}")
        dobot.clear_alarms()