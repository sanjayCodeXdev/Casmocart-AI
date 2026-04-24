from concurrent.futures import ThreadPoolExecutor
from product_agent import product_agent
from user_agent import user_agent
from config import call_openrouter

def synthesize_response(user_query, product_data=None, user_data=None, skin_data=None):
    """Combines information from multiple agents into a single proper reply."""
    skin_context = f"\nDetected Skin Analysis: {skin_data['interpretation']}" if skin_data else ""
    
    synthesis_prompt = [
        {"role": "system", "content": 
         "You are the Casmocart AI Lead Assistant. Your job is to take raw analysis from specialized agents and turn it into a beautiful, empathetic, and professional response for the user."
         "Use Markdown for formatting. Ensure the reply is concise but comprehensive."
         "If there is product analysis, include the safety score and key highlights."
         "If there is dermatologist advice, include the specific product suggestions and skin type analysis."
         f"Context from Scan: {skin_context}"},
        {"role": "user", "content": f"User Query: {user_query}\n\nProduct Analysis: {product_data}\n\nUser Analysis: {user_data}"}
    ]
    return call_openrouter(synthesis_prompt)

def supervisor_decision(user_input, skin_data=None):
    print(f"DEBUG: Supervisor analyzing query...")
    
    # Inject skin data into user query context if available
    contextual_input = user_input
    if skin_data:
        contextual_input = f"[Skin Analysis: {skin_data['interpretation']}] User Query: {user_input}"

    routing_prompt = [
        {"role": "system", "content":
         "You are a supervisor AI. Decide which agent should handle the request."
         "PRODUCT: User asks about a specific product or its ingredients."
         "USER: User asks for recommendations or mentions their skin type/concerns/budget."
         "BOTH: Request involves both specific product analysis and personal skin advice."
         "Return ONLY one word: PRODUCT or USER or BOTH."},
        {"role": "user", "content": contextual_input}
    ]

    decision = call_openrouter(routing_prompt).strip().upper()
    print(f"DEBUG: Routing to {decision}")

    if decision == "PRODUCT":
        result = product_agent(user_input)
        return synthesize_response(user_input, product_data=result["response"], skin_data=skin_data)

    elif decision == "USER":
        result = user_agent(contextual_input) # Pass context to user agent
        return synthesize_response(user_input, user_data=result["response"], skin_data=skin_data)

    elif "BOTH" in decision or (("PRODUCT" in decision) and ("USER" in decision)):
        # Run in parallel for speed
        with ThreadPoolExecutor() as executor:
            future_product = executor.submit(product_agent, user_input)
            future_user = executor.submit(user_agent, contextual_input)
            
            product_result = future_product.result()
            user_result = future_user.result()

        return synthesize_response(user_input, product_data=product_result["response"], user_data=user_result["response"], skin_data=skin_data)

    else:
        return synthesize_response(user_input, user_data="The supervisor could not classify but here is a general analysis based on your query.", skin_data=skin_data)


# -------------------------
# RUN SYSTEM
# -------------------------

if __name__ == "__main__":
    print("\n====================================")
    print(" AI For Her - Premium Multi-Agent ")
    print("====================================\n")

    user_query = input("Enter your query: ")

    print("\n Thinking...")
    result = supervisor_decision(user_query)

    print("\n" + "="*40)
    print("FINAL RESPONSE")
    print("="*40)
    print(result)
    print("="*40 + "\n")