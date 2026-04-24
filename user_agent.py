from config import call_openrouter

def user_agent(user_input):
    messages = [
        {"role": "system", "content": 
         "You are a Board-Certified Dermatologist AI. Your goal is to analyze user skin concerns, types, and goals to suggest specific products or active ingredients."
         "Identify the user's skin type (Oily, Dry, Combination, Sensitive), specific concerns (Acne, Aging, Hyperpigmentation), and suggest a curated list of product types (Cleanser, Serum, Moisturizer, SPF) with specific ingredient recommendations (e.g., 'Look for Niacinamide' or 'Avoid heavy oils')."
         "Suggest 2-3 specific, well-known, high-quality product examples that fit their budget and needs."},
        
        {"role": "user", "content": user_input}
    ]

    response = call_openrouter(messages)
    return {
        "agent": "User Requirement Agent",
        "response": response
    }