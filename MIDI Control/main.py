import json
import os
import threading
import time
import re
import mido
import vgamepad as vg

from constants import XBOX_BUTTONS

# Initialize virtual gamepad
gamepad = vg.VX360Gamepad()

# --- Load and Parse Configuration ---
with open('config.json', 'r') as f:
    config = json.load(f)

# Global Constants from Config
BUFFER_DURATION = config.get("buffer_ms", 15) / 1000.0
PAN_SPEED = config.get("pan_speed", 400)
TOGGLE_TRIGGER = config.get("inventory_toggle_trigger", "").lower().strip()
CLOSE_INV_TRIGGER = config.get("close_inventory_trigger", "").lower().strip()
PEDAL_TRIGGER = config.get("sustain_pedal_trigger", "").lower().strip()
EXIT_TRIGGER = config.get("exit_trigger", "").lower().strip()

# State tracking for inventory layer split
inventory_mode = False

# Track active direction states for BOTH thumbsticks globally
panning_states = {
    "look_right": False,
    "look_left": False,
    "look_up": False,
    "look_down": False
}

movement_states = {
    "move_forward": False,
    "move_back": False,
    "move_left": False,
    "move_right": False
}

# Unified Master Panning & Movement Thread Loop
def master_controller_engine():
    while True:
        # 1. Right Stick Calculations (Looking)
        rx_axis = 0.0
        ry_axis = 0.0
        if panning_states["look_right"]: rx_axis += 1.0
        if panning_states["look_left"]:  rx_axis -= 1.0
        if panning_states["look_down"]:  ry_axis -= 1.0  # Inverted standard gamepad Y
        if panning_states["look_up"]:    ry_axis += 1.0
        
        # 2. Left Stick Calculations (Walking)
        lx_axis = 0.0
        ly_axis = 0.0
        if movement_states["move_forward"]: ly_axis += 1.0
        if movement_states["move_back"]:    ly_axis -= 1.0
        if movement_states["move_left"]:    lx_axis -= 1.0
        if movement_states["move_right"]:   lx_axis += 1.0
        
        # Push values to both joysticks
        gamepad.right_joystick_float(x_value_float=rx_axis, y_value_float=ry_axis)
        gamepad.left_joystick_float(x_value_float=lx_axis, y_value_float=ly_axis)
        gamepad.update()
        
        time.sleep(0.01)

# Launch our unified gamepad engine
threading.Thread(target=master_controller_engine, daemon=True).start()

class SmartKeyboard:
    def press(self, key_str):
        global inventory_mode
        key_str = key_str.lower().strip()
        
        # 1. Handle Hard Exit Hook
        if key_str == "exit_program" or key_str == EXIT_TRIGGER:
            print("\nExit Triggered! Releasing all inputs and shutting down...")
            gamepad.reset()
            gamepad.update()
            os._exit(0)
            
        # 2. Intercept inventory toggle natively inside the chain
        if key_str == TOGGLE_TRIGGER:
            # If the menu is already open, use the close button instead
            target_button = CLOSE_INV_TRIGGER if inventory_mode else TOGGLE_TRIGGER
            inventory_mode = not inventory_mode            
            print(f"🔄 Inventory Mode Swapped! Active: {inventory_mode} (Sending: {target_button})")
            
            if target_button in XBOX_BUTTONS:
                gamepad.press_button(button=XBOX_BUTTONS[target_button])
                gamepad.update()
        
        # 3. Handle Analog Controller Movement 
        elif key_str in movement_states:
            movement_states[key_str] = True
            
        # 4. Handle Analog Controller Looking
        elif key_str in panning_states:
            panning_states[key_str] = True
            
        # 5. Handle Analog Triggers
        elif key_str == "left_trigger":
            gamepad.left_trigger_float(value_float=1.0)
            gamepad.update()
        elif key_str == "right_trigger":
            gamepad.right_trigger_float(value_float=1.0)
            gamepad.update()
            
        # 6. Handle Virtual Controller Buttons
        elif key_str in XBOX_BUTTONS:
            gamepad.press_button(button=XBOX_BUTTONS[key_str])
            gamepad.update()
            
    def release(self, key_str):
        key_str = key_str.lower().strip()
        
        if key_str in ["exit_program", EXIT_TRIGGER]:
            return
            
        # FIX: The release method must check if it was the toggle key,
        # and release BOTH possible buttons just to guarantee nothing gets stuck!
        if key_str == TOGGLE_TRIGGER:
            if TOGGLE_TRIGGER in XBOX_BUTTONS:
                gamepad.release_button(button=XBOX_BUTTONS[TOGGLE_TRIGGER])
            if CLOSE_INV_TRIGGER in XBOX_BUTTONS:
                gamepad.release_button(button=XBOX_BUTTONS[CLOSE_INV_TRIGGER])
            gamepad.update()
            
        elif key_str in movement_states:
            movement_states[key_str] = False
            
        elif key_str in panning_states:
            panning_states[key_str] = False
            
        elif key_str == "left_trigger":
            gamepad.left_trigger_float(value_float=0.0)
            gamepad.update()
        elif key_str == "right_trigger":
            gamepad.right_trigger_float(value_float=0.0)
            gamepad.update()
            
        elif key_str in XBOX_BUTTONS:
            gamepad.release_button(button=XBOX_BUTTONS[key_str])
            gamepad.update()

keyboard = SmartKeyboard()

# --- MIDI Parsing Helper ---
def note_name_to_midi(note_str):
    note_str = note_str.strip().upper()
    match = re.match(r"^([A-G][#B]?)(-?\d+)$", note_str)
    if not match:
        raise ValueError(f"Invalid note format: {note_str}")
    
    name, octave = match.groups()
    flat_map = {"BB": "A#", "DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#"}
    if name in flat_map:
        name = flat_map[name]
        
    note_mapping = {
        "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
        "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11
    }
    return (int(octave) + 1) * 12 + note_mapping[name]

# Build live runtime dictionaries
SINGLE_NOTES = {}
for note_name, bind in config.get("single_notes", {}).items():
    midi_num = note_name_to_midi(note_name)
    SINGLE_NOTES[str(midi_num)] = bind

INVENTORY_NOTES = {}
for note_name, bind in config.get("inventory_notes", {}).items():
    midi_num = note_name_to_midi(note_name)
    INVENTORY_NOTES[str(midi_num)] = bind

CHORDS = {}
for chord_names, bind in config.get("chords", {}).items():
    midi_nums = [note_name_to_midi(n) for n in chord_names.split(",")]
    chord_key = ",".join(map(str, sorted(midi_nums)))
    CHORDS[chord_key] = bind

# State tracking
pending_notes = set()
active_keys_pressed = set() 
buffer_timer = None

def process_buffer():
    global pending_notes, buffer_timer, active_keys_pressed
    
    notes_to_process = sorted(list(pending_notes))
    pending_notes.clear()
    buffer_timer = None
    
    if not notes_to_process:
        return

    chord_key = ",".join(map(str, notes_to_process))
    
    if chord_key in CHORDS:
        virtual_key = CHORDS[chord_key]
        keyboard.press(virtual_key)
        active_keys_pressed.add((chord_key, virtual_key))
    else:
        for note in notes_to_process:
            note_str = str(note)
            current_layer = INVENTORY_NOTES if (inventory_mode and note_str in INVENTORY_NOTES) else SINGLE_NOTES
            
            if note_str in current_layer:
                virtual_key = current_layer[note_str]
                keyboard.press(virtual_key)
                active_keys_pressed.add((note_str, virtual_key))

def handle_note_off(note):
    global active_keys_pressed
    note_str = str(note)
    
    to_remove = set()
    for trigger, virtual_key in active_keys_pressed.copy():
        if note_str == trigger or note_str in trigger.split(','):
            if virtual_key != PEDAL_TRIGGER:
                keyboard.release(virtual_key)
            to_remove.add((trigger, virtual_key))
            
    active_keys_pressed -= to_remove

def handle_note_on(note):
    global buffer_timer
    pending_notes.add(note)
    
    if buffer_timer is None:
        buffer_timer = threading.Timer(BUFFER_DURATION, process_buffer)
        buffer_timer.start()

# Main MIDI Loop
device_name = mido.get_input_names()[0] 

print(f"Listening on {device_name}...")
with mido.open_input(device_name) as inport:
    for msg in inport:
        if msg.type == 'note_on' and msg.velocity > 0:
            handle_note_on(msg.note)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            handle_note_off(msg.note)
        elif msg.type == 'control_change' and msg.control == 64:
            if PEDAL_TRIGGER:
                if msg.value >= 64:
                    keyboard.press(PEDAL_TRIGGER)
                else:
                    keyboard.release(PEDAL_TRIGGER)