# Medical Classification System

## Overview

This is a complete medical classification system designed for medico-legal analysis. It uses Google's Gemini AI or OpenAI to automatically categorize self-harm descriptions into predefined medical categories.

## Key Features

### 🤖 AI-Powered Classification
- **Dual AI Support**: Choose between Gemini or OpenAI for classification
- **Flexible API Key Setup**: Enter API keys directly in the app or use environment variables
- **10 Predefined Categories**: Specialized medical categories for self-harm descriptions
- **Confidence Scoring**: Detailed confidence scores (0-1) with reasoning for each classification
- **Smart Rate Limiting**: Automatic retry logic with exponential backoff

### 📊 User-Friendly Interface
- **Upload & Classify**: Upload CSV files with column selection or enter text manually
- **Review & Edit**: Interactive review interface with manual override capabilities
- **Export Results**: Download complete results and summary reports

### 🛡️ Production Features
- **Error Handling**: Comprehensive error handling with detailed feedback
- **Data Validation**: Input validation and cleaning
- **Serial Number Support**: Track classifications with original serial numbers
- **Quality Insights**: Data quality analysis and recommendations

## Comprehensive ICD-10 Based Classification Categories

The system uses a hierarchical two-level categorization based on ICD-10 classification:

### Main Categories:

**A. Poisoning/Toxic Ingestion**
- Pharmaceutical/Medication (tablets, capsules, liquids, injectables)
- Pesticides/Agricultural Chemicals (organophosphates, insecticides, herbicides)
- Household/Industrial Chemicals (rat poison, corrosives, cleaning agents, petroleum)
- Plant/Natural Toxins
- Unknown/Unspecified Compound

**B. Asphyxiation Methods**
- Hanging, Drowning/Submersion, Suffocation/Smothering
- Gas Inhalation (CO, toxic gases, chemical vapors)

**C. Sharp Object Injury**
- Self-Cutting (wrist/forearm), Self-Stabbing (chest/abdomen/neck)
- Injury by Glass, Knife/Blade, or Other Sharp Objects

**D. Burns/Thermal Injury**
- Self-Immolation (fire), Chemical Burns, Hot Liquid/Steam, Contact Burns

**E. Impact/Blunt Trauma**
- Jumping from Height, Jumping/Lying Before Moving Vehicle
- Self-Inflicted Blunt Trauma, Motor Vehicle Crash (Intentional)

**F. Firearm Injury**
- Self-Inflicted Gunshot Wound (handgun, rifle/shotgun, other)

**G. Electrocution**
- Intentional Electrical Injury

**H. Other Specified Methods**
- Explosive Material, Strangulation, Exposure to Extremes, Starvation/Dehydration

**I. Multiple Methods**
- Combined Methods (when multiple methods are used)

**J. Unclear/Undetermined Method**
- Other/Unclear Method, Undetermined Intent

## Setup Instructions

### 1. API Key Configuration
You need at least one API key from either provider. You can provide it in **two ways**:

#### Option A: Enter API Key in the App (Recommended)
1. Get your API key from one of these providers:
   - **Gemini**: [Google AI Studio](https://aistudio.google.com/app/apikey) (Free tier available)
   - **OpenAI**: [OpenAI Platform](https://platform.openai.com/api-keys) (Paid service)
2. Open the app and enter your API key in the sidebar under "AI Configuration"
3. The key will be securely stored in your session (password-masked input)

#### Option B: Use Environment Variables
1. Set environment variable before running the app:
   - For Gemini: `GEMINI_API_KEY=your_api_key_here`
   - For OpenAI: `OPENAI_API_KEY=your_api_key_here`
2. The app will automatically detect and use these keys

### 2. Running the Application
```bash
streamlit run app.py --server.port 5000
