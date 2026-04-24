from face_detect2 import run_face_scanner
from supervisor import supervisor_decision
from user_profile import identify_user, register_user, update_user_scan

def run_pipeline():
    print("\n" + "="*40)
    print("Welcome to AI For Her - Facial ID Skincare")
    print("="*40)

    # Step 1: Face Scan & Identification
    print("\n[STEP 1] Scanning your face...")
    skin_data = run_face_scanner()
    
    if not skin_data:
        print("⚠️ Scan failed. Cannot proceed without face identification.")
        return

    # Try to recognize the face
    signature = skin_data.get('signature')
    name, user_profile = identify_user(signature)

    if name:
        print(f"\n[WELCOME BACK] {name.upper()}!")
        print(f"I remember you. Your last scan was: {user_profile['latest_scan']['timestamp']}")
        
        # Ask if they want to update their profile or just proceed
        choice = input("Would you like to update your scan data? (y/n): ").lower()
        if choice == 'y':
            user_profile = update_user_scan(name, signature, skin_data)
            print(f"Profile for {name} updated successfully.")
    else:
        print("\n[NEW USER] I don't recognize this face yet.")
        new_name = input("Please enter your name to create a new profile: ")
        user_profile = register_user(new_name, signature, skin_data)

    # Step 2: User Query
    print("\n" + "="*40)
    user_query = input(f"How can I help you today, {user_profile['user_info']['name']}? ")

    # Injecting Profile Info for Personalization
    name = user_profile['user_info']['name']
    concerns = user_profile['user_info']['concerns']
    age = user_profile['user_info']['age']
    context_query = f"User: {name}, Age: {age}, Concerns: {concerns}. Analysis: {skin_data['interpretation']}. Query: {user_query}"

    # Step 3: Analysis
    print("\n[STEP 2] Running Multi-Agent Analysis...")
    result = supervisor_decision(context_query, skin_data=skin_data)

    # Step 4: Final Output
    print("\n" + "="*50)
    print(f" CONSULTATION FOR {name.upper()} ")
    print("="*50)
    print(result)
    print("="*50 + "\n")

if __name__ == "__main__":
    run_pipeline()
