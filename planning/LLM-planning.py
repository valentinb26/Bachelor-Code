import os
import time
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from gen_payload import generate_payload

# Lädt die Variablen aus der .env-Datei in die Umgebung
load_dotenv()

# 1. Definieren der gewünschten JSON-Ausgabe via Pydantic
class PlanStep(BaseModel):
    step_number: int
    action: str = Field(description="Name of the action, z. B. PICK, PLACE, SCREW")
    target: str = Field(description="The target object")
    location: str = Field(description="The target location")
    reasoning: str = Field(description="short explanation for this step")

class ExecutionPlan(BaseModel):
    plan_possible: bool = Field(description="Is the goal state reachable with the available objects?")
    steps: list[PlanStep]

# 2. Gemini Client initialisieren
client = genai.Client()

# 3. System-Prompt definieren
with open("/Users/v/Desktop/Coding_Workspaces/ba_code/planning/API/prompt.txt", "r", encoding="utf-8") as f:
    system_instruction = f.read()

# 4. Objektliste, Ist-Zustand und Ziel-Zustand definieren (User Payload)
user_payload = generate_payload()
#with open("/Users/v/Desktop/Coding_Workspaces/ba_code/planning/API/experimental_payload.json", "r", encoding="utf-8") as f:
#    user_payload = json.load(f)

t0 = time.time()
# 5. API-Aufruf mit Gemini 
response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=json.dumps(user_payload, ensure_ascii=False),
    #contents="follow the instructions",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        # Zwingt das Modell, exakt das Pydantic-Schema als JSON auszugeben
        response_mime_type="application/json",
        response_schema=ExecutionPlan,
        temperature=0.1,  # Niedrige Temperature für deterministische Ergebnisse
    ),
)
t1 = time.time()

# 6. Ergebnis ausgeben
print("--- GENERIERTER PLAN ---")
print(f"API-Aufruf dauerte: {t1 - t0:.2f} Sekunden")
print(response.text)

with open("API/execution_plan.json", "w", encoding="utf-8") as f:
    f.write(response.text)