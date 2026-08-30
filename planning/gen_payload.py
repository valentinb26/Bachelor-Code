import json

BIG_WHEEL_COUNT = 4
SMALL_WHEEL_COUNT = 4
SIDE_COVER_COUNT = 2
SCREW_COUNT = 10
    
def generate_items(base_id: str, count: int, item_type: str, status: str = "available"):
    """Helper function to quickly create multiple identical objects."""
    return [
        {
            "id": f"{base_id}_{i+1}",
            "type": item_type,
            "status": status
        }
        for i in range(count)
    ]

def generate_payload():
    # 1. Individual or special components
    components = [
        {"id": "blue_cover", "type": "cover", "status": "available"},
        {"id": "engine", "type": "engine", "status": "available"},
        {"id": "roof", "type": "roof", "status": "available"},
        {"id": "spoiler", "type": "spoiler", "status": "available"},
    ]


    # 2. Generate multiple components automatically
    components.extend(generate_items(base_id="big_wheel", count=BIG_WHEEL_COUNT, item_type="wheel"))
    components.extend(generate_items(base_id="small_wheel", count=SMALL_WHEEL_COUNT, item_type="wheel"))
    components.extend(generate_items(base_id="side_cover", count=SIDE_COVER_COUNT, item_type="side_cover"))
    components.extend(generate_items(base_id="screw", count=SCREW_COUNT, item_type="screw"))

    # 3. Slots / Locations
    slots = [
        {"id": "blue_cover_slot", "occupied": False},
        {"id": "engine-slot", "occupied": False},
        {"id": "roof-slot", "occupied": False},
        {"id": "spoiler-slot", "occupied": False},
    ]

    # Wheel slots per side: slot_A and slot_B
    sides = ["left", "right"]
    for side in sides:
        slots.append({"id": f"slot_A_{side}", "occupied": False})
        slots.append({"id": f"slot_B_{side}", "occupied": False})

    # Big wheel slots (only created after side_covers are screwed)
    for side in sides:
        slots.append({"id": f"big_slot_1_{side}", "occupied": False, "created_by": f"side_cover_{side}"})
        slots.append({"id": f"big_slot_2_{side}", "occupied": False, "created_by": f"side_cover_{side}"})

    # 4. Assemble complete payload
    payload = {
        "task": "Assemble Monster Truck",
        "components": components,
        "slots": slots,
        "hand": {"status": "empty", "holding": None},
        "robot_position": "home"
    }

    # Output as formatted JSON
    json_output = json.dumps(payload, indent=2)
    print(json_output)
        
    return json_output
        
if __name__ == "__main__":
    payload =generate_payload()
    with open("API/experimental_payload.json", "w", encoding="utf-8") as f:
        f.write(payload)