import mido

# 1. List all available devices so you can see what your PC detects
devices = mido.get_input_names()

if not devices:
    print("❌ No MIDI input devices found! Check your USB connection.")
else:
    print("Available devices:")
    for i, name in enumerate(devices):
        print(f"  [{i}] {name}")
        
    # 2. Pick the first device found
    device_name = devices[0]
    print(f"\n🚀 Opening: {device_name}")
    print("Press keys on your keyboard... (Press Ctrl+C to exit)\n")
    
    # 3. Listen and print everything
    try:
        with mido.open_input(device_name) as inport:
            for msg in inport:
                # This will print the note number, velocity, and message type
                print(msg)
    except KeyboardInterrupt:
        print("\nStopping test script.")