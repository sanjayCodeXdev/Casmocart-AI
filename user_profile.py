import json
import os
import time
import numpy as np

PROFILES_FILE = "profiles.json"

def calculate_similarity(sig1, sig2):
    """Calculates similarity, handles mismatched signature lengths safely."""
    s1 = np.array(sig1)
    s2 = np.array(sig2)
    if s1.shape != s2.shape:
        return 999.0 # Incompatible signature format
    return float(np.linalg.norm(s1 - s2))

def load_all_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_all_profiles(profiles):
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=4)

def identify_user(current_signature):
    """Attempts to find a matching face signature in the database."""
    if current_signature is None:
        return None, None
        
    profiles = load_all_profiles()
    min_dist = 0.15 # EXTREMELY strict threshold to prevent false matches
    recognized_name = None

    print(f"DEBUG: Checking signature {current_signature}")
    for name, data in profiles.items():
        if not data: continue # Skip corrupt entries
        
        # Check if signature exists and is valid
        sig = data.get("face_signature")
        if sig and isinstance(sig, list):
            if len(sig) == len(current_signature):
                dist = calculate_similarity(current_signature, sig)
                print(f"DEBUG: Distance to {name} = {dist:.4f}")
                if dist < min_dist:
                    min_dist = dist
                    recognized_name = name
    
    if recognized_name:
        print(f"DEBUG: Recognized as {recognized_name}")
    else:
        print("DEBUG: No match found within threshold.")

    return recognized_name, profiles.get(recognized_name) if recognized_name else None

def register_user(name, signature, skin_data):
    """Adds a new user or updates an existing one."""
    profiles = load_all_profiles()
    
    # Basic Interview if new
    if name not in profiles:
        print(f"\n[NEW] Registering New Profile: {name}")
        age = input("Enter your age: ")
        concerns = input("Primary skin concerns: ")
        budget = input("Monthly budget (Low/Mid/High): ")
        profiles[name] = {
            "user_info": {"name": name, "age": age, "concerns": concerns, "budget": budget},
            "history": []
        }
    else:
        print(f"\n[UPDATE] Updating Existing Profile: {name}")
    
    return update_user_scan(name, signature, skin_data, profiles)

def update_user_scan(name, signature, skin_data, profiles=None):
    """Updates the latest scan data for a user."""
    if profiles is None:
        profiles = load_all_profiles()
        
    if name not in profiles:
        return None

    profiles[name]["face_signature"] = signature
    profiles[name]["latest_scan"] = {
        "timestamp": time.ctime(),
        "interpretation": skin_data['interpretation'],
        "raw_data": skin_data['regions']
    }

    save_all_profiles(profiles)
    return profiles[name]
