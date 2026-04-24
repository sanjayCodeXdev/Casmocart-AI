from config import call_openrouter

def product_agent(user_input):
    messages = [
        {"role": "system", "content": 
         "You are a Senior Cosmetic Chemist and Safety Auditor. Your task is to provide a deep analysis of skin care/cosmetic products based on their ingredients or name."
         "Identify potential irritants (like denatured alcohol, synthetic fragrances), endocrine disruptors (parabens, phthalates), and beneficial ingredients (ceramides, niacinamide)."
         "Provide: 1. Ingredient Deep Dive, 2. Safety Score (0-100), 3. Skin Type Compatibility, and 4. Final Verdict (Safe/Caution/Avoid)."},
        
        {"role": "user", "content": user_input}
    ]

    response = call_openrouter(messages)
    return {
        "agent": "Product Agent",
        "response": response
    }