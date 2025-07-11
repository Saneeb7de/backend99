# utils.py - Prompts and Utilities for Medical Transcription
import asyncio
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# --- Gemini API Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=GOOGLE_API_KEY) # type: ignore

# --- PROMPTS ---
VERBATIM_TRANSCRIPTION_PROMPT = """
You are an expert AI for transcribing healthcare conversations in a mix of Malayalam and English.

Your goal is to transcribe the provided audio accurately. The audio may be in multiple parts, but you must treat it as one continuous conversation.

**INSTRUCTIONS:**

1. **Identify Speaker Roles:** First, listen to the audio to understand the context (e.g., is it a clinic, a pharmacy, or a reception desk?). Then, assign one of the following roles to each speaker:
• `Doctor`
• `Patient` 
• `Pharmacist`
• `Customer`
• `Receptionist`
• `Attender`
• `Guardian`
• `Nurse`
• `Lab Technician`

2. **Label Speakers:**
• If a speaker's name is mentioned, use their role and name (e.g., `Doctor Priya:`, `Pharmacist Anil:`)
• If no name is used, just use the role you identified (e.g., `Patient:`, `Customer:`)
• For anyone else, use `Other:`

3. **Transcribe Accurately:** 
• Be extremely precise with all medical and pharmaceutical terms
• Transcribe the mix of English and Malayalam exactly as it is spoken
• Include proper punctuation and paragraph breaks for readability
• Note any unclear audio with [unclear] tags
• Preserve the natural flow of conversation

4. **Medical Context Awareness:**
• Pay special attention to medication names, dosages, and instructions
• Accurately capture symptoms, diagnoses, and medical advice
• Note any numbers, dates, or time references precisely

**EXAMPLE of a pharmacy interaction:**

Pharmacist: Good morning, how can I help you?
Customer: എനിക്ക് ഡോക്ടർ തന്ന ഈ prescription-ലെ മരുന്ന് വേണം. (I need the medicine from this prescription the doctor gave me.)
Pharmacist: Okay, let me check. This is Atorvastatin 10mg. You need to take one tablet after dinner daily.
Customer: ഇത് എത്ര ദിവസം കഴിക്കണം? (How many days should I take this?)
Pharmacist: Doctor has prescribed this for 30 days. Complete the full course.
Customer: Okay, thank you.
"""

def get_medical_data_extraction_prompt(verbatim_transcript):
    """
    Returns the medical data extraction prompt with the verbatim transcript embedded.
    """
    return f"""
You are an expert clinical data analyst with two key responsibilities:
1.  **Correcting Errors**: You must identify and correct likely spelling or speech-to-text errors in the transcript (e.g., correct "Parasite Amol" to "Paracetamol").
2.  **Extracting Data**: You must meticulously extract the corrected information into a structured JSON object.

First, carefully read the definitions and rules for each category:
- "final_english_text": A cleaned, coherent summary of the conversation, containing the *corrected* information.
- "Symptoms": Patient-reported issues (e.g., "headache", "shortness of breath when walking").
- "Medicine Names": Specific prescribed drug names.
    - **RULE 1**: Correct any misspellings or ASR errors (e.g., "Atorvasatin" becomes "Atorvastatin").
    - **RULE 2**: Do NOT include dosage, strength, or frequency.
    - **RULE 3 (CRITICAL)**: Do NOT include generic forms like 'tablet', 'syrup', 'injection', 'capsule', or 'ointment'.
- "Dosage & Frequency": The amount and timing of a dose (e.g., "500mg", "twice a day").
- "Diseases / Conditions": Diagnosed or potential medical conditions (e.g., "Hypertension").
- "Medical Procedures / Tests": Any ordered medical tests (e.g., "Blood test", "ECG").
- "Duration": How long a treatment or symptom lasts (e.g., "for 3 days").
- "Doctor's Instructions": Specific non-medication advice (e.g., "get plenty of rest").

---
Here is an example of how to perform the task correctly:

EXAMPLE INPUT TEXT (contains errors):
"Patient has a fever and it was mentioned he is taking Dolo and also Parasite Amol. And a bottle of Cof-Ex syrup. We need an ECG."

EXAMPLE JSON OUTPUT (shows correction and filtering):
{{
  "final_english_text": "The patient has a fever and is taking Dolo and Paracetamol. He is also taking Cof-Ex syrup. An ECG is required.",
  "extracted_terms": {{
    "Symptoms": ["fever"],
    "Medicine Names": ["Dolo", "Paracetamol", "Cof-Ex"],
    "Dosage & Frequency": [],
    "Diseases / Conditions": [],
    "Medical Procedures / Tests": ["ECG"],
    "Duration": [],
    "Doctor's Instructions": []
  }},
  "summary": {{
    "medications_discussed": "Dolo, Paracetamol, Cof-Ex syrup",
    "important_instructions": "Patient should get an ECG",
    "follow_up_actions": "Schedule ECG appointment"
  }}
}}
---

CRITICAL: Your response must be ONLY a valid JSON object. Do not include markdown code blocks or any other text.

Text to process:
\"\"\"{verbatim_transcript}\"\"\"
"""

# --- UTILITY FUNCTIONS ---
def process_medical_data_with_gemini(verbatim_transcript, model):
    """
    Process verbatim transcript with Gemini API to extract medical data.
    
    Args:
        verbatim_transcript (str): The verbatim transcription text
        model: Configured Gemini model instance
        
    Returns:
        str: JSON string containing medical data extraction
        
    Raises:
        ValueError: If JSON parsing fails or API error occurs
    """
    try:
        medical_prompt = get_medical_data_extraction_prompt(verbatim_transcript)
        medical_response = model.generate_content(medical_prompt)
        medical_data_raw = medical_response.text.strip()
        
        # Clean up the response - remove markdown code blocks if present
        medical_data_clean = medical_data_raw.replace('```json\n', '').replace('\n```', '').replace('```json', '').replace('```', '').strip()
        
        # Validate that it's proper JSON
        try:
            medical_data_parsed = json.loads(medical_data_clean)
            return json.dumps(medical_data_parsed, ensure_ascii=False)  # This preserves Unicode characters
        except json.JSONDecodeError as e:
            print(f"⚠️ WARNING: Medical data is not valid JSON, using raw response: {e}")
            return medical_data_clean
            
    except Exception as e:
        raise ValueError(f"An unexpected error occurred with the Gemini API: {e}")

# def combine_transcription_data(verbatim_transcript, medical_data):
#     """
#     Combine verbatim transcription and medical data into a single JSON structure.
    
#     Args:
#         verbatim_transcript (str): The verbatim transcription
#         medical_data (str): The medical data extraction JSON
        
#     Returns:
#         str: Combined JSON string
#     """
#     combined_data = {
#         "verbatim_transcription": verbatim_transcript,
#         "medical_data": medical_data
#     }
#     return json.dumps(combined_data, ensure_ascii=False, indent=2)

def create_gemini_model(temperature=0.3):
    """
    Create and configure a Gemini model instance.
    
    Args:
        temperature (float): Generation temperature (default: 0.3)
        
    Returns:
        GenerativeModel: Configured Gemini model
    """
    generation_config = {"temperature": temperature}
    return genai.GenerativeModel( # type: ignore
        model_name="gemini-1.5-flash-latest", 
        generation_config=generation_config # type: ignore
    )

def cleanup_uploaded_files(uploaded_files_list):
    """
    Clean up uploaded files from Gemini API.
    
    Args:
        uploaded_files_list (list): List of uploaded file objects
    """
    for uploaded_file in uploaded_files_list:
        try:
            genai.delete_file(uploaded_file.name) # type: ignore
        except Exception as e:
            print(f"Warning: Could not delete file {uploaded_file.name}: {e}")


async def process_medical_data_with_gemini_async(verbatim_transcript, model):
    """
    Asynchronous version of process_medical_data_with_gemini.
    """
    try:
        medical_prompt = get_medical_data_extraction_prompt(verbatim_transcript)
        medical_response = await model.generate_content_async(medical_prompt)
        medical_data_raw = medical_response.text.strip()
        
        medical_data_clean = medical_data_raw.replace('```json\n', '').replace('\n```', '').replace('```json', '').replace('```', '').strip()
        
        try:
            # FIX: Return the parsed Python dictionary directly, NOT a JSON string.
            medical_data_parsed = json.loads(medical_data_clean)
            return medical_data_parsed 
        except json.JSONDecodeError as e:
            print(f"❌ CRITICAL ERROR: Gemini response could not be parsed as JSON: {e}")
            print(f"Raw Gemini Response was: {medical_data_clean}")
            # Return a structured error object instead of a raw string
            return {"error": "Failed to parse medical data from AI model."}
            
    except Exception as e:
        raise ValueError(f"An unexpected error occurred with the Gemini API: {e}")
    

def combine_transcription_data(verbatim_transcript, medical_data_dict):
    """
    Combine verbatim transcription and medical data into a single JSON structure.
    
    Args:
        verbatim_transcript (str): The verbatim transcription
        medical_data_dict (dict): The medical data extraction as a Python dictionary
        
    Returns:
        str: Combined JSON string
    """
    # This function now correctly receives a dictionary and creates one final, clean JSON string.
    combined_data = {
        "verbatim_transcription": verbatim_transcript,
        "medical_data": medical_data_dict  # This is now a dict, not a string
    }
    return json.dumps(combined_data, ensure_ascii=False, indent=2)