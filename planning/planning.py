# planning.py
import sys
import json

import unified_planning as up
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.shortcuts import (
    GE,
    LT,
    BoolType,
    Equals,
    Fluent,
    InstantaneousAction,
    IntType,
    Not,
    Object,
    OneshotPlanner,
    PlanValidator,
    Problem,
    UserType,
)

up.shortcuts.get_environment().credits_stream = None

# ============================================================
# KONFIGURATION: 
# ============================================================
VEHICLE_TYPE = "car"  # Hier wechseln zwischen 'car' und 'monster_truck'

PLANNER = "enhsp"
NUM_SCREWS_LONG = 11
NUM_SCREWS_SHORT = 3
NUM_SLOTS_TOP = 3
NUM_SLOTS_RIGHT = 2
NUM_SLOTS_LEFT = 2
NUM_SLOTS_FRONT = 1
NUM_SLOTS_BACK = 1

# Fahrzeugspezifische Parameter-Generierung
if VEHICLE_TYPE == "car":
    NUM_WHEELS_SMALL = 4
    NUM_WHEELS_BIG = 0
    NUM_SIDE_COVERS = 0
    NUM_ADDONS = 0
    NUM_BUMPERS = 0
    
elif VEHICLE_TYPE == 'tuned_car':
    NUM_WHEELS_SMALL = 4
    NUM_WHEELS_BIG = 0
    NUM_SIDE_COVERS = 0
    NUM_ADDONS = 3
    NUM_BUMPERS = 2
    
elif VEHICLE_TYPE == "monster_truck":
    NUM_WHEELS_SMALL = 0
    NUM_WHEELS_BIG = 4
    NUM_SIDE_COVERS = 2  # 1 links, 1 rechts
    NUM_ADDONS = 3
    NUM_BUMPERS = 2
else:
    raise ValueError(
        "Unbekannter VEHICLE_TYPE. Bitte 'car' oder 'monster_truck' wählen."
    )


# ============================================================
# 0. Helper-Funktionen
# ============================================================
def define_types():
    types = {}
    types["PhysicalObject"] = UserType("PhysicalObject")

    types["Component"] = UserType("Component", types["PhysicalObject"])
    types["BlueCover"] = UserType("BlueCover", types["Component"])
    types["SideCover"] = UserType("SideCover", types["Component"])
    types["WheelSmall"] = UserType("WheelSmall", types["Component"])
    types["WheelBig"] = UserType("WheelBig", types["Component"])
    types["Addon"] = UserType("Addon", types["Component"])
    types["Bumper"] = UserType("Bumper", types["Component"])
    types["BumperFront"] = UserType("BumperFront", types["Bumper"])
    types["BumperBack"] = UserType("BumperBack", types["Bumper"])

    types["Screw"] = UserType("Screw", types["PhysicalObject"])
    types["ScrewLong"] = UserType("ScrewLong", types["Screw"])
    types["ScrewShort"] = UserType("ScrewShort", types["Screw"])

    types["Slot"] = UserType("Slot")
    types["SlotMaster"] = UserType("SlotMaster", types["Slot"])
    types["SlotTop"] = UserType("SlotTop", types["Slot"])
    types["SlotRight"] = UserType("SlotRight", types["Slot"])
    types["SlotLeft"] = UserType("SlotLeft", types["Slot"])
    types["SlotFront"] = UserType("SlotFront", types["Slot"])
    types["SlotBack"] = UserType("SlotBack", types["Slot"])
    types["SlotSideCover"] = UserType("SlotSideCover", types["Slot"])

    return types


def define_fluents(types):
    PhysicalObject = types["PhysicalObject"]
    Slot = types["Slot"]
    Component = types["Component"]
    Screw = types["Screw"]

    fluents = {}
    fluents["arm_free"] = Fluent("arm_free")
    fluents["holding"] = Fluent("holding", BoolType(), obj=PhysicalObject)
    fluents["in_storage"] = Fluent("in_storage", BoolType(), obj=PhysicalObject)
    fluents["placed"] = Fluent("placed", BoolType(), obj=PhysicalObject)

    fluents["at_slot"] = Fluent("at_slot", BoolType(), obj=PhysicalObject, s=Slot)
    fluents["slot_free"] = Fluent("slot_free", BoolType(), s=Slot)
    fluents["slot_compatible"] = Fluent(
        "slot_compatible", BoolType(), obj=PhysicalObject, s=Slot
    )

    fluents["compatible"] = Fluent("compatible", BoolType(), c=Component, scr=Screw)
    fluents["screw_count"] = Fluent("screw_count", IntType(), c=Component)

    fluents["blue_cover_placed"] = Fluent("blue_cover_placed", BoolType())
    fluents["slot_active"] = Fluent("slot_active", BoolType(), s=Slot)

    return fluents


def define_actions(types, fluents):
    PhysicalObject = types["PhysicalObject"]
    Slot = types["Slot"]
    Component = types["Component"]
    Screw = types["Screw"]
    BlueCover = types["BlueCover"]
    SideCover = types["SideCover"]

    # Subtypen für präzises Scoping holen
    SlotLeft = types["SlotLeft"]
    SlotRight = types["SlotRight"]
    SlotSideCover = types["SlotSideCover"]

    arm_free = fluents["arm_free"]
    holding = fluents["holding"]
    in_storage = fluents["in_storage"]
    slot_free = fluents["slot_free"]
    slot_compatible = fluents["slot_compatible"]
    at_slot = fluents["at_slot"]
    compatible = fluents["compatible"]
    screw_count = fluents["screw_count"]
    blue_cover_placed = fluents["blue_cover_placed"]
    slot_active = fluents["slot_active"]

    # ── pick(obj) ──
    pick = InstantaneousAction("pick", obj=PhysicalObject)
    obj = pick.parameter("obj")
    pick.add_precondition(arm_free())
    pick.add_precondition(in_storage(obj))
    pick.add_effect(arm_free(), False)
    pick.add_effect(holding(obj), True)
    pick.add_effect(in_storage(obj), False)

    # ── place_blue_cover(bc, s) ──
    place_bc = InstantaneousAction(
        "place_blue_cover", bc=BlueCover, s=types["SlotMaster"]
    )
    bc = place_bc.parameter("bc")
    s = place_bc.parameter("s")
    place_bc.add_precondition(holding(bc))
    place_bc.add_precondition(slot_free(s))

    place_bc.add_effect(holding(bc), False)
    place_bc.add_effect(arm_free(), True)
    place_bc.add_effect(at_slot(bc, s), True)
    place_bc.add_effect(slot_free(s), False)
    place_bc.add_effect(blue_cover_placed(), True)

    # ── OPTIMIERUNG: place_side_cover aufgeteilt in links und rechts wegen Grounding-Explosion ──

    # ── place_side_cover_left ──
    place_sc_left = InstantaneousAction(
        "place_side_cover_left",
        sc=SideCover,
        s1=SlotLeft,
        s2=SlotLeft,
        ns1=SlotSideCover,
        ns2=SlotSideCover,
    )
    sc = place_sc_left.parameter("sc")
    s1 = place_sc_left.parameter("s1")
    s2 = place_sc_left.parameter("s2")
    ns1 = place_sc_left.parameter("ns1")
    ns2 = place_sc_left.parameter("ns2")

    place_sc_left.add_precondition(blue_cover_placed())
    place_sc_left.add_precondition(holding(sc))
    place_sc_left.add_precondition(slot_free(s1))
    place_sc_left.add_precondition(slot_free(s2))
    place_sc_left.add_precondition(slot_active(s1))
    place_sc_left.add_precondition(slot_active(s2))
    place_sc_left.add_precondition(slot_compatible(sc, s1))
    place_sc_left.add_precondition(slot_compatible(sc, s2))
    place_sc_left.add_precondition(slot_compatible(sc, ns1))
    place_sc_left.add_precondition(slot_compatible(sc, ns2))
    place_sc_left.add_precondition(Not(Equals(s1, s2)))
    place_sc_left.add_precondition(Not(Equals(ns1, ns2)))

    place_sc_left.add_effect(holding(sc), False)
    place_sc_left.add_effect(arm_free(), True)
    place_sc_left.add_effect(at_slot(sc, s1), True)  # Slot 1 belegen
    place_sc_left.add_effect(
        at_slot(sc, s2), True
    )  # Slot 2 belegen -> Erlaubt präzises Schrauben an beiden Slots!
    place_sc_left.add_effect(slot_free(s1), False)
    place_sc_left.add_effect(slot_free(s2), False)
    place_sc_left.add_effect(slot_active(ns1), True)
    place_sc_left.add_effect(slot_active(ns2), True)
    place_sc_left.add_effect(fluents["placed"](sc), True)

    # ── place_side_cover_right ──
    place_sc_right = InstantaneousAction(
        "place_side_cover_right",
        sc=SideCover,
        s1=SlotRight,
        s2=SlotRight,
        ns1=SlotSideCover,
        ns2=SlotSideCover,
    )
    sc = place_sc_right.parameter("sc")
    s1 = place_sc_right.parameter("s1")
    s2 = place_sc_right.parameter("s2")
    ns1 = place_sc_right.parameter("ns1")
    ns2 = place_sc_right.parameter("ns2")

    place_sc_right.add_precondition(blue_cover_placed())
    place_sc_right.add_precondition(holding(sc))
    place_sc_right.add_precondition(slot_free(s1))
    place_sc_right.add_precondition(slot_free(s2))
    place_sc_right.add_precondition(slot_active(s1))
    place_sc_right.add_precondition(slot_active(s2))
    place_sc_right.add_precondition(slot_compatible(sc, s1))
    place_sc_right.add_precondition(slot_compatible(sc, s2))
    place_sc_right.add_precondition(slot_compatible(sc, ns1))
    place_sc_right.add_precondition(slot_compatible(sc, ns2))
    place_sc_right.add_precondition(Not(Equals(s1, s2)))
    place_sc_right.add_precondition(Not(Equals(ns1, ns2)))

    place_sc_right.add_effect(holding(sc), False)
    place_sc_right.add_effect(arm_free(), True)
    place_sc_right.add_effect(at_slot(sc, s1), True)
    place_sc_right.add_effect(at_slot(sc, s2), True)
    place_sc_right.add_effect(slot_free(s1), False)
    place_sc_right.add_effect(slot_free(s2), False)
    place_sc_right.add_effect(slot_active(ns1), True)
    place_sc_right.add_effect(slot_active(ns2), True)
    place_sc_right.add_effect(fluents["placed"](sc), True)

    # ── place(obj, s) ──
    place = InstantaneousAction("place", obj=Component, s=Slot)
    obj = place.parameter("obj")
    s = place.parameter("s")

    place.add_precondition(blue_cover_placed())
    place.add_precondition(holding(obj))
    place.add_precondition(slot_free(s))
    place.add_precondition(slot_active(s))
    place.add_precondition(slot_compatible(obj, s))

    place.add_effect(holding(obj), False)
    place.add_effect(arm_free(), True)
    place.add_effect(at_slot(obj, s), True)
    place.add_effect(slot_free(s), False)
    place.add_effect(fluents["placed"](obj), True)

    # ── OPTIMIERUNG: screw hat wieder s=Slot und prüft exakt via at_slot ──
    screw = InstantaneousAction("screw", scr=Screw, c=Component, s=Slot)
    scr = screw.parameter("scr")
    c = screw.parameter("c")
    s = screw.parameter("s")

    screw.add_precondition(holding(scr))
    screw.add_precondition(
        at_slot(c, s)
    )  # Prüft, ob Komponente auf genau diesem Slot liegt
    screw.add_precondition(compatible(c, scr))
    screw.add_precondition(
        slot_compatible(c, s)
    )  # KOSTET NIX & HILFT DEM GROUNDER ENORM (Statischer Filter)
    screw.add_precondition(LT(screw_count(c), 2))
    screw.add_precondition(fluents["placed"](c))

    screw.add_effect(holding(scr), False)
    screw.add_effect(arm_free(), True)
    screw.add_effect(screw_count(c), screw_count(c) + 1)

    return [pick, place_bc, place_sc_left, place_sc_right, place, screw]


def setup_component_compatibility(problem, fluents, components, screws, slots):
    for comp in components:
        for screw in screws:
            problem.set_initial_value(fluents["compatible"](comp, screw), True)
        for slot in slots:
            problem.set_initial_value(fluents["slot_compatible"](comp, slot), True)


def create_objects_by_type(type_class, count):
    import re

    type_name_lower = re.sub(r"(?<!^)(?=[A-Z])", "_", type_class.name).lower()
    if count == 0:
        return []
    if count == 1:
        return [Object(f"{type_name_lower}_", type_class)]
    return [Object(f"{type_name_lower}_{i}", type_class) for i in range(1, count + 1)]


def setup_problem():
    types = define_types()
    fluents = define_fluents(types)
    actions = define_actions(types, fluents)

    problem = Problem(f"roboter_montage_{VEHICLE_TYPE}")

    fluents_config = {
        fluents["arm_free"]: True,
        fluents["holding"]: False,
        fluents["in_storage"]: False,
        fluents["at_slot"]: False,
        fluents["slot_free"]: True,
        fluents["screw_count"]: 0,
        fluents["compatible"]: False,
        fluents["slot_compatible"]: False,
        fluents["blue_cover_placed"]: False,
        fluents["slot_active"]: False,
        fluents["placed"]: False,
    }

    for fluent, default_value in fluents_config.items():
        problem.add_fluent(fluent, default_initial_value=default_value)

    for action in actions:
        problem.add_action(action)

    # Objekte generieren
    blue_covers = create_objects_by_type(types["BlueCover"], 1)
    slots_master = create_objects_by_type(types["SlotMaster"], 1)

    side_covers = create_objects_by_type(types["SideCover"], NUM_SIDE_COVERS)
    slots_side_cover = create_objects_by_type(
        types["SlotSideCover"], NUM_SIDE_COVERS * 2
    )

    wheels_small = create_objects_by_type(types["WheelSmall"], NUM_WHEELS_SMALL)
    wheels_big = create_objects_by_type(types["WheelBig"], NUM_WHEELS_BIG)

    addons = create_objects_by_type(types["Addon"], NUM_ADDONS)
    bumpers_front = create_objects_by_type(types["BumperFront"], NUM_SLOTS_FRONT)
    bumpers_back = create_objects_by_type(types["BumperBack"], NUM_SLOTS_BACK)
    bumpers = bumpers_front + bumpers_back

    screws_long = create_objects_by_type(types["ScrewLong"], NUM_SCREWS_LONG)
    screws_short = create_objects_by_type(types["ScrewShort"], NUM_SCREWS_SHORT)

    slots_top = create_objects_by_type(types["SlotTop"], NUM_SLOTS_TOP)
    slots_right = create_objects_by_type(types["SlotRight"], NUM_SLOTS_RIGHT)
    slots_left = create_objects_by_type(types["SlotLeft"], NUM_SLOTS_LEFT)
    slots_front = create_objects_by_type(types["SlotFront"], NUM_SLOTS_FRONT)
    slots_back = create_objects_by_type(types["SlotBack"], NUM_SLOTS_BACK)

    all_objects = (
        blue_covers
        + slots_master
        + side_covers
        + slots_side_cover
        + wheels_small
        + wheels_big
        + addons
        + bumpers
        + screws_long
        + screws_short
        + slots_top
        + slots_right
        + slots_left
        + slots_front
        + slots_back
    )

    for obj in all_objects:
        problem.add_object(obj)

    problem.set_initial_value(fluents["arm_free"](), True)

    for obj in (
        blue_covers
        + side_covers
        + wheels_small
        + wheels_big
        + addons
        + bumpers
        + screws_long
        + screws_short
    ):
        problem.set_initial_value(fluents["in_storage"](obj), True)

    standard_slots = (
        slots_master + slots_top + slots_right + slots_left + slots_front + slots_back
    )
    for s in standard_slots:
        problem.set_initial_value(fluents["slot_active"](s), True)

    problem.add_goal(fluents["blue_cover_placed"]())

    for obj in wheels_small + wheels_big + addons + bumpers_front:
        problem.add_goal(Equals(fluents["screw_count"](obj), 1))

    for obj in bumpers_back + side_covers:
        problem.add_goal(Equals(fluents["screw_count"](obj), 2))

    setup_component_compatibility(problem, fluents, blue_covers, [], slots_master)
    setup_component_compatibility(
        problem, fluents, wheels_small, screws_long, slots_right + slots_left
    )
    setup_component_compatibility(
        problem, fluents, wheels_big, screws_long, slots_side_cover
    )
    setup_component_compatibility(problem, fluents, addons, screws_long, slots_top)
    setup_component_compatibility(
        problem, fluents, bumpers_front, screws_short, slots_front
    )
    setup_component_compatibility(
        problem, fluents, bumpers_back, screws_short, slots_back
    )

    if VEHICLE_TYPE == "monster_truck" and len(side_covers) >= 2:
        sc_left = side_covers[0]
        sc_right = side_covers[1]

        for s in slots_left:
            problem.set_initial_value(fluents["slot_compatible"](sc_left, s), True)
            problem.set_initial_value(
                fluents["slot_compatible"](sc_left, slots_side_cover[0]), True
            )
            problem.set_initial_value(
                fluents["slot_compatible"](sc_left, slots_side_cover[1]), True
            )

        for s in slots_right:
            problem.set_initial_value(fluents["slot_compatible"](sc_right, s), True)
            problem.set_initial_value(
                fluents["slot_compatible"](sc_right, slots_side_cover[2]), True
            )
            problem.set_initial_value(
                fluents["slot_compatible"](sc_right, slots_side_cover[3]), True
            )

        for scr in screws_long:
            problem.set_initial_value(fluents["compatible"](sc_left, scr), True)
            problem.set_initial_value(fluents["compatible"](sc_right, scr), True)


    return problem


def solve(problem, output_filename="plan.json"):
    with OneshotPlanner(name=PLANNER) as planner:
        print(f"\n[DEBUG] Starte Planer {planner.name}...")
        result = planner.solve(problem, output_stream=sys.stdout)
    
    print(f"\nPlanerstatus: {result.status}")
    
    plan_data = {
        "status": str(result.status),
        "solved": result.status == PlanGenerationResultStatus.SOLVED_SATISFICING,
        "actions": []
    }
    
    if plan_data["solved"]:
        print("\n✓ Plan gefunden! Speichere in JSON...\n")
        
        for action_instance in result.plan.actions:
            # action_instance.action.name gibt den Namen der Aktion (z.B. "move")
            # action_instance.actual_parameters gibt die UP-Objekte zurück
            action_name = action_instance.action.name
            args = [str(param) for param in action_instance.actual_parameters]
            
            plan_data["actions"].append({
            "action": action_name,
            "object": args[0] if len(args) > 0 else None,
            "slots": args[1:] if len(args) > 1 else []
        })
            
        # Als JSON-Datei abspeichern
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, indent=4, ensure_ascii=False)
    else:
        print("✗ Kein Plan gefunden. Keine JSON erstellt.")
        
    return result


def validate(problem, plan):
    if plan is not None:
        with PlanValidator(problem_kind=problem.kind) as validator:
            val = validator.validate(problem, plan)
            print(f"\nValidierungsergebnis: {val}")


def solve_and_validate(problem):
    print("=" * 65)
    print(f"Problem:      {problem.name}")
    print(f"ProblemKind:  {problem.kind}")
    print("=" * 65)
    result = solve(problem)
    if result.plan:
        validate(problem, result.plan)


if __name__ == "__main__":
    problem_instance = setup_problem()
    solve_and_validate(problem_instance)
