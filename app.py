import os
import asyncio
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
import json
import aiohttp
from dotenv import load_dotenv
from transformers import pipeline
import torch
import cv2
import numpy as np
from PIL import Image
import pytesseract
import io
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()
HuggingFace_token = os.getenv("HUGGINGFACE_TOKEN")
app = FastAPI(title="AI Medical Prescription Verification", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    print("Loading Hugging Face NER model...")
    print("Hugging Face NER model loaded.")
except ImportError:
    print("Hugging Face transformers not found. Using AI-only extraction.")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-3.5-turbo"

def simple_drug_extraction_ai_only(text: str) -> List[str]:
    """Very simple extraction that just splits words and filters"""
    words = text.replace(',', ' ').replace('.', ' ').split()
    potential_drugs = []
    for word in words:
        word = word.strip('.,();:!?')
        if len(word) > 3 and word.isalpha() and word[0].isupper():
            potential_drugs.append(word)
    
    return potential_drugs[:20]

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Simple image preprocessing for better OCR results"""
    try:
        # Convert PIL image to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_cv = img_array
        
        # Apply basic preprocessing
        # 1. Resize if image is too small
        height, width = img_cv.shape
        if height < 300 or width < 300:
            scale_factor = max(300/height, 300/width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img_cv = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 2. Denoise
        img_cv = cv2.medianBlur(img_cv, 3)
        
        # 3. Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_cv = clahe.apply(img_cv)
        
        # 4. Threshold to get better black and white image
        _, img_cv = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to PIL Image
        return Image.fromarray(img_cv)
    
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return image  # Return original if preprocessing fails

def extract_text_from_image(image_file) -> str:
    """Extract text from image using OCR - split processing for multiple lines"""
    try:
        # Read image
        image = Image.open(io.BytesIO(image_file))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array for processing
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Split image into top and bottom halves
        mid_point = height // 2
        top_half = image.crop((0, 0, width, mid_point + 50))  # Add overlap
        bottom_half = image.crop((0, mid_point - 50, width, height))  # Add overlap
        
        # Process each half separately
        text_parts = []
        
        for img_part in [top_half, bottom_half]:
            text = pytesseract.image_to_string(img_part, config=r'--oem 3 --psm 7')
            if text.strip():
                text_parts.append(text.strip())
        
        # Also try full image with PSM 6
        full_text = pytesseract.image_to_string(image, config=r'--oem 3 --psm 6')
        if full_text.strip():
            text_parts.append(full_text.strip())
        
        # Combine and clean results
        all_text = ' '.join(text_parts)
        
        return all_text
    
    except Exception as e:
        print(f"OCR extraction error: {e}")
        return ""

class DrugInteractionRequest(BaseModel):
    drugs: List[str]
    patient_age: int
    medical_conditions: List[str] = []

class DrugExtractionRequest(BaseModel):
    medical_text: str

class InteractionResponse(BaseModel):
    interactions: List[Dict[str, Any]]
    alternatives: List[Dict[str, Any]]
    safety_warnings: List[str]
    age_specific_warnings: List[str]

FDA_API_BASE = "https://api.fda.gov/drug"

async def get_drug_info_fda(drug_name: str) -> Dict:
    """Get drug information from FDA API"""
    try:
        url = f"{FDA_API_BASE}/label.json"
        params = {
            "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
            "limit": 1
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("results"):
                        return data["results"][0]
    except Exception as e:
        print(f"FDA API error for {drug_name}: {e}")
    
    return {}

async def call_openrouter_api(messages: List[Dict], temperature: float = 0.2, max_tokens: int = 2000) -> str:
    """Call OpenRouter API for chat completions"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Medical Prescription Verification"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenRouter API error {response.status}: {error_text}")
                    
    except Exception as e:
        print(f"OpenRouter API error: {e}")
        raise e

async def analyze_drug_interactions_openrouter(drugs: List[str], patient_age: int, 
                                             medical_conditions: List[str]) -> Dict:
    """Analyze drug interactions using OpenRouter API"""
    try:
        drug_list = ", ".join(drugs)
        conditions_text = ", ".join(medical_conditions) if medical_conditions else "None reported"
        
        prompt = f"""
        As a medical AI assistant, analyze the following drug combination for potential interactions:

        Drugs: {drug_list}
        Patient Age: {patient_age} years
        Medical Conditions: {conditions_text}

        Please provide a detailed analysis including:
        1. Drug-to-drug interactions (severity: mild/moderate/severe)
        2. Age-specific considerations for {patient_age}-year-old patient
        3. Alternative medication suggestions for any problematic drugs
        4. Safety warnings and precautions

        Format your response as JSON with the following structure:
        {{
            "interactions": [
                {{
                    "drug1": "drug name",
                    "drug2": "drug name", 
                    "severity": "mild/moderate/severe",
                    "description": "interaction description",
                    "mechanism": "how the interaction occurs"
                }}
            ],
            "alternatives": [
                {{
                    "original_drug": "drug name",
                    "alternative": "alternative drug name",
                    "reason": "why this alternative is better",
                    "age_appropriate": true/false
                }}
            ],
            "safety_warnings": ["warning1", "warning2"],
            "age_specific_warnings": ["age warning1", "age warning2"]
        }}
        """

        messages = [
            {
                "role": "system", 
                "content": "You are a specialized medical AI that provides drug interaction analysis. Always prioritize patient safety and provide evidence-based recommendations. Always respond with valid JSON format."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        response_content = await call_openrouter_api(messages, temperature=0.2, max_tokens=2000)
        
        # Parse JSON response
        content = response_content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        
        result = json.loads(content)
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return {
            "interactions": [],
            "alternatives": [],
            "safety_warnings": ["Unable to parse AI response. Please consult a healthcare professional."],
            "age_specific_warnings": []
        }
    except Exception as e:
        print(f"OpenRouter API error: {e}")
        return {
            "interactions": [],
            "alternatives": [],
            "safety_warnings": [f"AI analysis unavailable: {str(e)}"],
            "age_specific_warnings": []
        }

@app.get("/")
async def root():
    return {"message": "AI Medical Prescription Verification API", "status": "running"}

@app.post("/analyze_interactions", response_model=InteractionResponse)
async def analyze_drug_interactions(request: DrugInteractionRequest):
    """Analyze drug interactions and provide alternatives"""
    if not request.drugs:
        raise HTTPException(status_code=400, detail="No drugs provided")
    
    if request.patient_age <= 0:
        raise HTTPException(status_code=400, detail="Invalid patient age")
    
    try:
        drug_info_tasks = [get_drug_info_fda(drug) for drug in request.drugs]
        drug_info_results = await asyncio.gather(*drug_info_tasks, return_exceptions=True)
        
        analysis = await analyze_drug_interactions_openrouter(
            request.drugs, 
            request.patient_age, 
            request.medical_conditions
        )
        
        return InteractionResponse(
            interactions=analysis.get("interactions", []),
            alternatives=analysis.get("alternatives", []),
            safety_warnings=analysis.get("safety_warnings", []),
            age_specific_warnings=analysis.get("age_specific_warnings", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/extract_drugs")
async def extract_drugs_from_text(request: DrugExtractionRequest):
    """Extract drug names from medical text using AI only"""
    if not request.medical_text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        if OPENROUTER_API_KEY:
            try:
                extraction_prompt = f"""
                From the following medical text, identify and extract ALL pharmaceutical drug names, medications, and treatments mentioned:
                
                Text: {request.medical_text}
                
                Please return only actual prescription medications, over-the-counter drugs, or medical treatments as a JSON array.
                Include both brand names and generic names if mentioned.
                Format: ["drug1", "drug2", "drug3"]
                """
                
                messages = [
                    {
                        "role": "system", 
                        "content": "You are a medical text analyzer specialized in identifying pharmaceutical drugs and medications. Extract all valid drug names, including prescription medications, over-the-counter drugs, and medical treatments. Always respond with a valid JSON array."
                    },
                    {
                        "role": "user", 
                        "content": extraction_prompt
                    }
                ]
                
                response_content = await call_openrouter_api(messages, temperature=0.1, max_tokens=500)
                
                content = response_content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                
                extracted_drugs = json.loads(content)
                
                return {
                    "extracted_drugs": extracted_drugs,
                    "nlp_candidates": [],
                    "text_processed": len(request.medical_text),
                    "extraction_method": "AI-only"
                }
            except Exception as ai_error:
                print(f"AI extraction error: {ai_error}")
                return {
                    "extracted_drugs": [],
                    "nlp_candidates": [],
                    "text_processed": len(request.medical_text),
                    "error": f"AI extraction failed: {str(ai_error)}",
                    "note": "Please enter drug names manually for analysis"
                }
        else:
            return {
                "extracted_drugs": [],
                "nlp_candidates": [],
                "text_processed": len(request.medical_text),
                "error": "No API key configured",
                "note": "Please enter drug names manually for analysis"
            }
        
    except Exception as e:
        return {
            "extracted_drugs": [],
            "nlp_candidates": [],
            "text_processed": len(request.medical_text),
            "error": f"Extraction failed: {str(e)}"
        }

@app.post("/extract_from_image")
async def extract_drugs_from_image(file: UploadFile = File(...)):
    """Extract text from prescription image and then extract drug names"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Extract text from image using OCR
        extracted_text = extract_text_from_image(contents)
        
        if not extracted_text.strip():
            return {
                "extracted_text": "",
                "extracted_drugs": [],
                "error": "No text could be extracted from the image",
                "note": "Try uploading a clearer image or enter text manually"
            }
        
        # Use AI to extract drug names from the OCR text
        if OPENROUTER_API_KEY:
            try:
                extraction_prompt = f"""
                From the following text extracted from a medical prescription image, identify and extract ALL pharmaceutical drug names, medications, and treatments mentioned:
                
                Text: {extracted_text}
                
                Please return only actual prescription medications, over-the-counter drugs, or medical treatments as a JSON array.
                Include both brand names and generic names if mentioned.
                Be forgiving of OCR errors and try to recognize common drug names even if slightly misspelled.
                Format: ["drug1", "drug2", "drug3"]
                """
                
                messages = [
                    {
                        "role": "system", 
                        "content": "You are a medical text analyzer specialized in identifying pharmaceutical drugs from OCR-extracted prescription text. Extract all valid drug names, even if there are minor spelling errors from OCR. Always respond with a valid JSON array."
                    },
                    {
                        "role": "user", 
                        "content": extraction_prompt
                    }
                ]
                
                response_content = await call_openrouter_api(messages, temperature=0.1, max_tokens=500)
                
                content = response_content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                
                extracted_drugs = json.loads(content)
                
                return {
                    "extracted_text": extracted_text,
                    "extracted_drugs": extracted_drugs,
                    "text_processed": len(extracted_text),
                    "extraction_method": "OCR + AI"
                }
                
            except Exception as ai_error:
                print(f"AI extraction error: {ai_error}")
                return {
                    "extracted_text": extracted_text,
                    "extracted_drugs": [],
                    "error": f"AI drug extraction failed: {str(ai_error)}",
                    "note": "OCR succeeded but drug extraction failed"
                }
        else:
            return {
                "extracted_text": extracted_text,
                "extracted_drugs": [],
                "error": "No API key configured for drug extraction",
                "note": "OCR succeeded but need API key for drug extraction"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if Tesseract is available
    tesseract_available = True
    try:
        pytesseract.get_tesseract_version()
    except:
        tesseract_available = False
    
    return {
        "status": "healthy",
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "extraction_method": "AI-only",
        "api_provider": "openrouter",
        "ocr_available": tesseract_available
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)