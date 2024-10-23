import spacy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import language_tool_python
import openai
import re
from config import OPENAI_API_KEY
from langdetect import detect, LangDetectException

models = {
    "en": spacy.load("en_core_web_sm"),  # Inglés
    "es": spacy.load("es_core_news_md"),  # Español
    "fr": spacy.load("fr_core_news_sm"),  # Francés
}

# Historial de interacciones, mantendrá las últimas tres interacciones
history = []

def init():
    print("Initializing the application...")
    openai.api_key = OPENAI_API_KEY
    
    print("Loading NLP model...")
    global nlp
    nlp = spacy.load("es_core_news_md")

    print("Training classifier...")
    prompts = [
        # Spanish prompts
        "Hola", 
        "Explícame cómo funciona la mecánica cuántica", 
        "komo estas?",

        # English prompts
        "Hello", 
        "Explain to me how quantum mechanics works", 
        "how r u?",
    ]
    
    # Corresponding labels: 0 = not optimizable, 1 = optimizable
    labels = [
        0, 1, 1,  # Spanish
        0, 1, 1,  # English
    ]

    # Extract features for each prompt
    X = np.array([extract_features(prompt) for prompt in prompts])

    # Train the classifier
    global clf
    clf = RandomForestClassifier()
    clf.fit(X, labels)
    print("Classifier trained.")
    
def contains_important_content(prompt):
    # Detecta cualquier texto dentro de los delimitadores < >
    has_protected_parts = bool(re.search(r"<[^>]+>", prompt))
    return has_protected_parts


def detect_grammatical_errors(text, language='es'):
    tool = language_tool_python.LanguageTool(language)
    errors = tool.check(text)
    error_list = []
    for error in errors:
        error_list.append({
            'message': error.message,
            'suggested_replacements': error.replacements,
            'found_error': error.context,
            'start_index': error.offset,
            'end_index': error.offset + error.errorLength
        })
    return error_list

def detect_language(prompt):
    try:
        lang_code = detect(prompt)  # Detecta el idioma
        if lang_code not in models:
            print(f"Idioma detectado '{lang_code}' no soportado, usando 'es' por defecto.")
            return "en"
        return lang_code
    except LangDetectException:
        return "em"  # Valor predeterminado en caso de error (español)


def count_adjectives(prompt):
    language = detect_language(prompt)
    nlp = models.get(language)

    if not nlp:
        print(f"No hay modelo disponible para el idioma '{language}'. Saltando el conteo de adjetivos.")
        return 0

    doc = nlp(prompt)
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    return len(adjectives)

def extract_features(prompt):
    doc = nlp(prompt)
    prompt_length = len(prompt)
    num_tokens = len(doc)
    num_verbs = len([token for token in doc if token.pos_ == "VERB"])
    num_nouns = len([token for token in doc if token.pos_ == "NOUN"])
    num_adjectives = count_adjectives(prompt)
    num_grammatical_errors = len(detect_grammatical_errors(prompt))
    return [prompt_length, num_tokens, num_grammatical_errors, num_verbs, num_nouns, num_adjectives]

def is_optimizable(prompt):
    features = np.array(extract_features(prompt)).reshape(1, -1)
    return clf.predict(features)[0]

def optimize_prompt_with_openai(prompt, grammatical_errors, protected_parts, is_opt):
    last_role = get_last_role()  # Obtener el rol anterior del historial
    
    if not is_opt:
        print("The prompt is not optimizable, only grammatical errors will be corrected.")
        
        if grammatical_errors:
            for error in grammatical_errors:
                prompt = prompt[:error['start_index']] + (error['suggested_replacements'][0] if error['suggested_replacements'] else '') + prompt[error['end_index']:]
            return prompt, last_role  # Usamos el último rol si el prompt no es optimizable
        else:
            print("The prompt has no grammatical errors, returning it as is.")
            return prompt, last_role
    
    print("The prompt is optimizable, proceeding with optimization.")
    
    # Proteger los fragmentos que están entre < >
    if protected_parts:
        for i, part in enumerate(protected_parts):
            prompt = prompt.replace(part, f"[IMPORTANT_CONTENT_{i}]")
    
    # Instrucciones claras a OpenAI para devolver el rol en la respuesta
    prompt_optimization = f"""
    Your goal is to fully understand the intention of the user's question and optimize its vocabulary to offer a detailed and specialized response.
    
    - Clarify and improve the structure of the user's prompt by adding necessary details and steps while maintaining its core intent.
    - You must also **identify the role** of the person answering the prompt based on the context (e.g., "You are a senior developer", "You are a professional chef").
    
    **Query to Optimize:**
    {prompt}
    
    **Errors Detected (for reference):**
    {"No significant errors detected." if not grammatical_errors else "".join([f"Error in '{error['found_error']}': {error['message']}" for error in grammatical_errors])}
    
    **Optimized Query and Role**:
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that optimizes user queries based on their content."},
            {"role": "user", "content": prompt_optimization}
        ],
        max_tokens=300,
        temperature=0.7
    )
    
    # Procesamos la respuesta de OpenAI
    optimized_text = response['choices'][0]['message']['content'].strip()

    # Restauramos los fragmentos protegidos en el texto optimizado
    if protected_parts:
        for i, part in enumerate(protected_parts):
            optimized_text = optimized_text.replace(f"[IMPORTANT_CONTENT_{i}]", part)
    
    # Extraer el rol de la respuesta si está presente
    role_match = re.search(r"Role:\s*(.+)", optimized_text)
    if role_match:
        role = role_match.group(1).strip()
        optimized_text = re.sub(r"Role:\s*.+", "", optimized_text).strip()  # Remove role from the final optimized text
    else:
        role = last_role  # If no role is found, use the last role

    return optimized_text, role

def get_last_role():
    # Si hay interacciones previas en el historial, devolvemos el último rol
    if history:
        last_interaction = history[-1]
        return last_interaction.get("role", "No role")
    return "No role"



def process_prompt(prompt):
    print(f"\nProcessing the prompt: {prompt}")
    
    # Detección de fragmentos protegidos usando delimitadores < >
    protected_parts = re.findall(r"<[^>]+>", prompt)
    
    # Remover los delimitadores < > de los fragmentos protegidos
    protected_parts = [part[1:-1] for part in protected_parts]
    
    # Detect if the prompt is optimizable
    is_opt = is_optimizable(prompt)
    
    # Detect grammatical errors
    grammatical_errors = detect_grammatical_errors(prompt)
    
    # Optimize the prompt using OpenAI, or correct grammatical errors if not optimizable
    optimized_prompt, role = optimize_prompt_with_openai(prompt, grammatical_errors, protected_parts, is_opt)
    
    # Store the interaction in the history
    history.append({"prompt": prompt, "optimized_prompt": optimized_prompt, "role": role})
    if len(history) > 3:
        history.pop(0)  # Keep only the last 3 interactions
    
    # Structured JSON output
    output = {
        "optimized_prompt": optimized_prompt,
        "role": role,
        "history": history[-3:]  # Return last 3 interactions
    }
    
    return output




# Main loop to receive multiple prompts
if __name__ == "__main__":
    init()  # Initialize the model and classifier

    print("\nApplication started. Type 'exit' to quit.")
    
    while True:
        prompt = input("\nEnter a prompt: ")
        if prompt.lower() == "exit":
            print("Exiting the application...")
            break
        result = process_prompt(prompt)
        print(result)  # Output the structured result as JSON
