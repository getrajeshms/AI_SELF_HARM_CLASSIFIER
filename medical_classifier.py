import os
import json
import logging
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel

class ClassificationResult(BaseModel):
    category: str
    confidence: float
    reasoning: str

class MedicalClassifier:
    """Medical classification system using Gemini AI or OpenAI for self-harm description categorization"""
    
    def __init__(self):
        self.gemini_client = None
        self.openai_client = None
        self.current_provider = "Gemini"
        
        self.category_structure = {
            "A. Poisoning/Toxic Ingestion": {
                "Poisoning - Pharmaceutical/Medication": ["Tablet/Capsule Overdose", "Liquid Medication Overdose", "Injectable Drug Overdose", "Multiple Drug Overdose"],
                "Poisoning - Pesticides/Agricultural Chemicals": ["Organophosphate Compounds", "Insecticide (non-OP)", "Herbicides", "Fungicides"],
                "Poisoning - Household/Industrial Chemicals": ["Rat Poison (Rodenticides)", "Corrosive Substances (Acids/Alkalis)", "Cleaning Agents", "Petroleum Products/Kerosene"],
                "Poisoning - Plant/Natural Toxins": ["Plant-based Poisons", "Mushroom/Fungal Toxins"],
                "Poisoning - Unknown/Unspecified Compound": []
            },
            "B. Asphyxiation Methods": {
                "Hanging": [],
                "Drowning/Submersion": [],
                "Suffocation/Smothering": [],
                "Gas Inhalation": ["Carbon Monoxide", "Other Toxic Gases", "Chemical Vapor Inhalation"]
            },
            "C. Sharp Object Injury": {
                "Self-Cutting (Wrist/Forearm)": [],
                "Self-Stabbing (Chest/Abdomen/Neck)": [],
                "Self-Injury by Glass": [],
                "Self-Injury by Knife/Blade": [],
                "Self-Injury by Other Sharp Objects": []
            },
            "D. Burns/Thermal Injury": {
                "Self-Immolation (Fire)": [],
                "Chemical Burns (Self-Inflicted)": [],
                "Hot Liquid/Steam Burns": [],
                "Contact Burns (Hot Objects)": []
            },
            "E. Impact/Blunt Trauma": {
                "Jumping from Height": [],
                "Jumping/Lying Before Moving Vehicle": ["Railway/Train", "Motor Vehicle"],
                "Self-Inflicted Blunt Trauma": [],
                "Motor Vehicle Crash (Intentional)": []
            },
            "F. Firearm Injury": {
                "Self-Inflicted Gunshot Wound": ["Handgun", "Rifle/Shotgun", "Other Firearms"]
            },
            "G. Electrocution": {
                "Intentional Electrical Injury": []
            },
            "H. Other Specified Methods": {
                "Explosive Material": [],
                "Strangulation (Self)": [],
                "Exposure to Extremes (Cold/Heat)": [],
                "Intentional Starvation/Dehydration": []
            },
            "I. Multiple Methods": {
                "Combined Methods": []
            },
            "J. Unclear/Undetermined Method": {
                "Other/Unclear Method": [],
                "Undetermined Intent": []
            }
        }
        
        # Flatten categories for validation
        self.all_subcategories = []
        self.category_to_main = {}
        for main_cat, subcats in self.category_structure.items():
            for subcat in subcats.keys():
                self.all_subcategories.append(subcat)
                self.category_to_main[subcat] = main_cat
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize both Gemini and OpenAI clients"""
        # Initialize Gemini client
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if gemini_api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=gemini_api_key)
                logging.info("Gemini client initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Gemini client: {e}")
                self.gemini_client = None
        
        # Initialize OpenAI client
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_api_key)
                logging.info("OpenAI client initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
    
    def set_provider(self, provider: str):
        """Set the current AI provider"""
        if provider in ["Gemini", "OpenAI"]:
            self.current_provider = provider
        else:
            raise ValueError("Provider must be either 'Gemini' or 'OpenAI'")
    
    def get_current_provider(self) -> str:
        """Get the current AI provider"""
        return self.current_provider
    
    def set_api_key(self, provider: str, api_key: str):
        """Set API key for a specific provider"""
        if provider == "Gemini" and api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=api_key)
                logging.info("Gemini client initialized with provided API key")
            except Exception as e:
                logging.error(f"Failed to initialize Gemini client with provided key: {e}")
                self.gemini_client = None
        elif provider == "OpenAI" and api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_key)
                logging.info("OpenAI client initialized with provided API key")
            except Exception as e:
                logging.error(f"Failed to initialize OpenAI client with provided key: {e}")
                self.openai_client = None
    
    def check_api_key(self) -> bool:
        """Check if the current provider's API key is properly configured"""
        if self.current_provider == "Gemini":
            return self.gemini_client is not None
        elif self.current_provider == "OpenAI":
            return self.openai_client is not None
        return False
    
    def check_env_key(self) -> bool:
        """Check if the current provider has an environment variable API key"""
        if self.current_provider == "Gemini":
            return bool(os.environ.get("GEMINI_API_KEY"))
        elif self.current_provider == "OpenAI":
            return bool(os.environ.get("OPENAI_API_KEY"))
        return False
    
    def classify_description(self, description: str) -> Dict[str, Any]:
        """
        Classify a self-harm description into predefined medical categories
        
        Args:
            description: The self-harm description text to classify
            
        Returns:
            Dictionary containing category, confidence, and reasoning
        """
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        
        if self.current_provider == "Gemini":
            return self._classify_with_gemini(description)
        elif self.current_provider == "OpenAI":
            return self._classify_with_openai(description)
        else:
            raise ValueError(f"Unknown provider: {self.current_provider}")
    
    def _classify_with_gemini(self, description: str) -> Dict[str, Any]:
        """Classify using Gemini API"""
        if not self.gemini_client:
            raise Exception("Gemini client not initialized. Please check your GEMINI_API_KEY.")
        
        from google.genai import types
        
        # Create the system prompt with categories and examples
        system_prompt = self._create_system_prompt()
        
        # Create the user prompt
        user_prompt = f"""
        Please classify the following self-harm description into one of the predefined medical categories.
        
        Description to classify: "{description}"
        
        Provide your response as JSON with the following structure:
        {{
            "category": "exact category name from the predefined list",
            "confidence": 0.95,
            "reasoning": "brief explanation of why this classification was chosen"
        }}
        """
        
        try:
            # Add retry logic for rate limiting
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.gemini_client.models.generate_content(
                        model="gemini-2.5-flash",  # the newest Gemini model is "gemini-2.5-flash", do not change this unless explicitly requested by the user
                        contents=[
                            types.Content(role="user", parts=[types.Part(text=user_prompt)])
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            response_schema=ClassificationResult,
                            temperature=0.1  # Low temperature for consistent medical classifications
                        ),
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "rate limit" in str(e).lower():
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 10  # Exponential backoff: 10s, 20s, 30s
                            logging.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}")
                            time.sleep(wait_time)
                            continue
                        else:
                            # Final attempt failed, return rate limit error
                            return {
                                "category": "Other/Unclear",
                                "confidence": 0.0,
                                "reasoning": "Rate limit exceeded. Please wait a moment and try again, or consider upgrading your API plan for higher limits."
                            }
                    else:
                        raise e  # Re-raise non-rate-limit errors
            
            if not response.text:
                raise Exception("Empty response from Gemini API")
            
            # Parse the JSON response
            result_data = json.loads(response.text)
            
            # Validate and clean the result
            return self._validate_and_clean_result(result_data)
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return {
                "category": "Other/Unclear",
                "confidence": 0.0,
                "reasoning": f"JSON parsing error: {str(e)}"
            }
        except Exception as e:
            logging.error(f"Gemini classification failed: {e}")
            return {
                "category": "Other/Unclear", 
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}"
            }
    
    def _classify_with_openai(self, description: str) -> Dict[str, Any]:
        """Classify using OpenAI API"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized. Please check your OPENAI_API_KEY.")
        
        # Create the system prompt with categories and examples
        system_prompt = self._create_system_prompt()
        
        # Create the user prompt
        user_prompt = f"""
        Please classify the following self-harm description into one of the predefined medical categories.
        
        Description to classify: "{description}"
        
        Provide your response as JSON with the following structure:
        {{
            "category": "exact category name from the predefined list",
            "confidence": 0.95,
            "reasoning": "brief explanation of why this classification was chosen"
        }}
        """
        
        try:
            # Add retry logic for rate limiting
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        max_completion_tokens=512
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    if "rate_limit_exceeded" in str(e).lower() or "429" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 10  # Exponential backoff: 10s, 20s, 30s
                            logging.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}")
                            time.sleep(wait_time)
                            continue
                        else:
                            # Final attempt failed, return rate limit error
                            return {
                                "category": "Other/Unclear",
                                "confidence": 0.0,
                                "reasoning": "Rate limit exceeded. Please wait a moment and try again, or consider upgrading your API plan for higher limits."
                            }
                    else:
                        raise e  # Re-raise non-rate-limit errors
            
            if not response.choices[0].message.content:
                raise Exception("Empty response from OpenAI API")
            
            # Parse the JSON response
            result_data = json.loads(response.choices[0].message.content)
            
            # Validate and clean the result
            return self._validate_and_clean_result(result_data)
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return {
                "category": "Other/Unclear",
                "confidence": 0.0,
                "reasoning": f"JSON parsing error: {str(e)}"
            }
        except Exception as e:
            logging.error(f"OpenAI classification failed: {e}")
            return {
                "category": "Other/Unclear", 
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}"
            }
    
    def _validate_and_clean_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the classification result"""
        subcategory = result_data.get("category", "Other/Unclear Method")
        
        # Validate the subcategory is in our predefined list
        if subcategory not in self.all_subcategories:
            # Default to Other/Unclear Method
            subcategory = "Other/Unclear Method"
            result_data["reasoning"] += " (Category adjusted to predefined list)"
        
        # Get the main category
        main_category = self.category_to_main.get(subcategory, "J. Unclear/Undetermined Method")
        
        # Update result with both main and subcategory
        result_data["category"] = subcategory
        result_data["main_category"] = main_category
        
        # Ensure confidence is between 0 and 1
        result_data["confidence"] = max(0.0, min(1.0, float(result_data["confidence"])))
        
        # Ensure reasoning is a string
        result_data["reasoning"] = str(result_data["reasoning"])
        
        return result_data
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt with medical categories and guidelines"""
        categories_text = ""
        for main_cat, subcats in self.category_structure.items():
            categories_text += f"\n**{main_cat}**\n"
            for subcat, details in subcats.items():
                if details:
                    categories_text += f"  - {subcat}: {', '.join(details)}\n"
                else:
                    categories_text += f"  - {subcat}\n"
        
        return f"""
        You are a Chief Medical Officer specializing in medico-legal cases and ICD-10 classification. Your task is to classify self-harm descriptions into specific medical categories for forensic analysis.

        **COMPREHENSIVE ICD-10 BASED CLASSIFICATION CATEGORIES:**
        {categories_text}

        **CLASSIFICATION GUIDELINES:**
        1. You must classify each description into exactly ONE subcategory from the list above
        2. Choose the MOST SPECIFIC subcategory that matches the description
        3. For poisoning cases, identify the specific substance type (pharmaceutical, pesticide, household chemical, etc.)
        4. For trauma cases, identify the mechanism (asphyxiation, sharp object, blunt trauma, burns, etc.)
        5. If multiple methods are clearly described, use "Combined Methods" from category I
        6. If the description is unclear or intent is uncertain, use appropriate category from J
        7. Base classification on medical terminology, mechanism of harm, and clinical presentation
        8. Provide confidence score between 0.0 and 1.0 based on certainty

        **SPECIFIC CLASSIFICATION RULES:**
        
        **Poisoning/Toxic Ingestion:**
        - Pharmaceutical: tablets, capsules, syrups, injections (medical drugs)
        - Pesticides: organophosphates, insecticides, herbicides, fungicides
        - Household/Industrial: rat poison, acids, alkalis, cleaning agents, kerosene
        - Plant/Natural: plant-based toxins, mushrooms
        - Unknown: when substance is not specified or cannot be determined
        
        **Asphyxiation:**
        - Hanging: ligature around neck, suspension
        - Drowning: submersion in water/liquid
        - Suffocation: blocking airways, smothering
        - Gas Inhalation: CO, toxic gases, chemical vapors
        
        **Sharp Object Injury:**
        - Cutting: superficial cuts on wrists/forearms
        - Stabbing: penetrating wounds to chest/abdomen/neck
        - Specify object: glass, knife/blade, or other sharp objects
        
        **Burns/Thermal:**
        - Fire: self-immolation, burning
        - Chemical: corrosive burns
        - Hot Liquid/Steam: scalding
        - Contact: touching hot objects
        
        **Impact/Blunt Trauma:**
        - Jumping from height
        - Railway/train or motor vehicle impact
        - Self-inflicted blunt force
        - Intentional vehicle crash
        
        **Other Methods:**
        - Firearm: specify if handgun, rifle, shotgun, or other
        - Electrocution: intentional electrical injury
        - Strangulation, explosive, exposure, starvation

        **CONFIDENCE SCORING GUIDELINES:**
        - 0.9-1.0: Very clear indicators, specific method/substance mentioned with medical terminology
        - 0.7-0.9: Clear classification with good supporting evidence and clinical details
        - 0.5-0.7: Reasonable classification but some ambiguity in description
        - 0.3-0.5: Uncertain classification due to vague or limited information
        - 0.0-0.3: Very uncertain, minimal supporting evidence, unclear description

        **OUTPUT FORMAT:**
        Respond with valid JSON containing:
        - "category": The exact subcategory name from the list (e.g., "Hanging", "Poisoning - Pharmaceutical/Medication")
        - "confidence": A number between 0.0 and 1.0
        - "reasoning": Brief medical explanation for the classification choice
        """
    
    def batch_classify(self, descriptions: list) -> list:
        """
        Classify multiple descriptions in batch
        
        Args:
            descriptions: List of self-harm description strings
            
        Returns:
            List of classification results
        """
        results = []
        for i, description in enumerate(descriptions):
            try:
                result = self.classify_description(description)
                results.append(result)
                
                # Add delay between batch requests
                if i < len(descriptions) - 1:
                    time.sleep(2)  # 2 second delay between requests
                    
            except Exception as e:
                results.append({
                    "category": "Other/Unclear",
                    "confidence": 0.0,
                    "reasoning": f"Batch processing error: {str(e)}"
                })
        return results
    
    def get_categories(self) -> Dict[str, str]:
        """Get the predefined medical categories and their examples"""
        return self.categories.copy()
    
    def validate_classification(self, category: str) -> bool:
        """Validate if a category is in the predefined list"""
        return category in self.categories
