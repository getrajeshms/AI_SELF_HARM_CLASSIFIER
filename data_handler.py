import pandas as pd
import csv
import io
from typing import List, Dict, Any, Optional
import logging

class DataHandler:
    """Handle data import, export, and validation for medical classification system"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.txt']
    
    def read_csv_file(self, file_content: bytes, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Read CSV file content and return as DataFrame
        
        Args:
            file_content: Raw bytes content of the CSV file
            encoding: Text encoding to use
            
        Returns:
            pandas DataFrame containing the CSV data
        """
        try:
            # Try to decode the content
            text_content = file_content.decode(encoding)
            
            # Create StringIO object for pandas
            csv_io = io.StringIO(text_content)
            
            # Read CSV with various delimiter detection
            df = pd.read_csv(csv_io)
            
            if df.empty:
                raise ValueError("CSV file is empty")
            
            logging.info(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except UnicodeDecodeError:
            # Try different encodings
            for alt_encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    text_content = file_content.decode(alt_encoding)
                    csv_io = io.StringIO(text_content)
                    df = pd.read_csv(csv_io)
                    logging.info(f"Successfully read CSV with encoding {alt_encoding}")
                    return df
                except:
                    continue
            raise ValueError("Unable to decode CSV file with supported encodings")
        
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            raise ValueError(f"Failed to read CSV file: {str(e)}")
    
    def parse_text_input(self, text_content: str) -> List[str]:
        """
        Parse manual text input into list of descriptions
        
        Args:
            text_content: Raw text input with descriptions
            
        Returns:
            List of cleaned description strings
        """
        if not text_content or not text_content.strip():
            return []
        
        # Split by lines and clean each description
        descriptions = []
        for line in text_content.split('\n'):
            cleaned_line = line.strip()
            if cleaned_line:  # Skip empty lines
                descriptions.append(cleaned_line)
        
        logging.info(f"Parsed {len(descriptions)} descriptions from text input")
        return descriptions
    
    def validate_descriptions(self, descriptions: List[str]) -> Dict[str, Any]:
        """
        Validate a list of descriptions for processing
        
        Args:
            descriptions: List of description strings
            
        Returns:
            Dictionary with validation results
        """
        if not descriptions:
            return {
                "valid": False,
                "error": "No descriptions provided",
                "valid_count": 0,
                "invalid_descriptions": []
            }
        
        valid_descriptions = []
        invalid_descriptions = []
        
        for i, desc in enumerate(descriptions):
            if not desc or not desc.strip():
                invalid_descriptions.append(f"Row {i+1}: Empty description")
            elif len(desc.strip()) < 3:
                invalid_descriptions.append(f"Row {i+1}: Description too short (minimum 3 characters)")
            elif len(desc.strip()) > 1000:
                invalid_descriptions.append(f"Row {i+1}: Description too long (maximum 1000 characters)")
            else:
                valid_descriptions.append(desc.strip())
        
        return {
            "valid": len(valid_descriptions) > 0,
            "valid_count": len(valid_descriptions),
            "invalid_count": len(invalid_descriptions),
            "invalid_descriptions": invalid_descriptions,
            "cleaned_descriptions": valid_descriptions
        }
    
    def export_to_csv(self, data: List[Dict[str, Any]], filename: str = "classification_results.csv") -> str:
        """
        Export classification results to CSV format
        
        Args:
            data: List of classification result dictionaries
            filename: Output filename
            
        Returns:
            CSV content as string
        """
        if not data:
            raise ValueError("No data to export")
        
        # Create DataFrame from results
        df = pd.DataFrame(data)
        
        # Create CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_ALL)
        csv_content = csv_buffer.getvalue()
        
        logging.info(f"Exported {len(data)} records to CSV")
        return csv_content
    
    def create_summary_report(self, classifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary report of classification results
        
        Args:
            classifications: List of classification results
            
        Returns:
            Dictionary containing summary statistics
        """
        if not classifications:
            return {"error": "No classifications to summarize"}
        
        df = pd.DataFrame(classifications)
        
        # Basic statistics
        total_count = len(df)
        successful_count = len(df[df['Status'].isin(['AI Classified', 'Manual Override'])])
        failed_count = len(df[df['Status'] == 'Failed'])
        rate_limited_count = len(df[df['Status'] == 'Rate Limited'])
        manual_override_count = len(df[df['Status'] == 'Manual Override'])
        
        # Confidence statistics (only for successful AI classifications)
        ai_classified_df = df[df['Status'] == 'AI Classified']
        if len(ai_classified_df) > 0:
            avg_confidence = ai_classified_df['Confidence'].mean()
            min_confidence = ai_classified_df['Confidence'].min()
            max_confidence = ai_classified_df['Confidence'].max()
            high_confidence_count = len(ai_classified_df[ai_classified_df['Confidence'] >= 0.8])
        else:
            avg_confidence = min_confidence = max_confidence = high_confidence_count = 0
        
        # Category distribution (final main categories after manual overrides)
        final_main_categories = []
        final_subcategories = []
        for _, row in df.iterrows():
            if row.get('Manual Override Main', ''):
                final_main_categories.append(row['Manual Override Main'])
                final_subcategories.append(row.get('Manual Override Sub', 'Unknown'))
            elif row['Status'] in ['AI Classified']:
                final_main_categories.append(row.get('Main Category', 'J. Unclear/Undetermined Method'))
                final_subcategories.append(row.get('Subcategory', 'Unknown'))
            else:
                final_main_categories.append('Failed/Unclassified')
                final_subcategories.append('Failed/Unclassified')
        
        main_category_distribution = pd.Series(final_main_categories).value_counts().to_dict()
        subcategory_distribution = pd.Series(final_subcategories).value_counts().to_dict()
        
        return {
            "total_processed": total_count,
            "successful_classifications": successful_count,
            "failed_classifications": failed_count,
            "rate_limited_classifications": rate_limited_count,
            "manual_overrides": manual_override_count,
            "success_rate": round((successful_count / total_count) * 100, 2) if total_count > 0 else 0,
            "confidence_stats": {
                "average": round(avg_confidence, 3) if avg_confidence else 0,
                "minimum": round(min_confidence, 3) if min_confidence else 0,
                "maximum": round(max_confidence, 3) if max_confidence else 0,
                "high_confidence_count": high_confidence_count,
                "high_confidence_percentage": round((high_confidence_count / len(ai_classified_df)) * 100, 2) if len(ai_classified_df) > 0 else 0
            },
            "category_distribution": main_category_distribution,
            "subcategory_distribution": subcategory_distribution
        }
    
    def detect_csv_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect potential description columns in a CSV file
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Dictionary with column suggestions
        """
        text_columns = []
        potential_description_columns = []
        potential_slno_columns = []
        
        for column in df.columns:
            # Check if column contains text data
            sample_data = df[column].dropna().head(10)
            if len(sample_data) > 0:
                # Check for potential serial number columns
                if any(keyword in column.lower() for keyword in ['sl', 'serial', 'number', 'no', 'id', 'index']):
                    potential_slno_columns.append(column)
                
                # Check if it's likely text (not purely numeric)
                text_ratio = sum(isinstance(val, str) and len(str(val)) > 10 for val in sample_data) / len(sample_data)
                if text_ratio > 0.5:
                    text_columns.append(column)
                    
                    # Look for keywords that suggest medical descriptions
                    sample_text = ' '.join(str(val).lower() for val in sample_data)
                    medical_keywords = ['poison', 'hanging', 'tablet', 'burn', 'injury', 'consumption', 'chemical', 'overdose', 'harm', 'attempt']
                    if any(keyword in sample_text for keyword in medical_keywords):
                        potential_description_columns.append(column)
        
        return {
            "text_columns": text_columns,
            "suggested_description_columns": potential_description_columns if potential_description_columns else text_columns,
            "potential_slno_columns": potential_slno_columns
        }
    
    def clean_description_text(self, text: str) -> str:
        """
        Clean and normalize description text
        
        Args:
            text: Raw description text
            
        Returns:
            Cleaned description text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace and normalize
        cleaned = ' '.join(text.strip().split())
        
        # Remove common non-informative prefixes/suffixes
        prefixes_to_remove = ['description:', 'details:', 'case:', 'patient:', 'subject:']
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        return cleaned
    
    def validate_export_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate data before export
        
        Args:
            data: List of classification results
            
        Returns:
            Validation result with any issues found
        """
        if not data:
            return {"valid": False, "errors": ["No data to export"]}
        
        errors = []
        warnings = []
        
        required_fields = ["Original Description", "AI Category", "Confidence", "Status"]
        
        for i, record in enumerate(data):
            for field in required_fields:
                if field not in record or record[field] is None:
                    errors.append(f"Record {i+1}: Missing required field '{field}'")
            
            # Check confidence values
            if "Confidence" in record:
                try:
                    conf = float(record["Confidence"])
                    if not 0.0 <= conf <= 1.0:
                        warnings.append(f"Record {i+1}: Confidence {conf} is outside valid range [0.0, 1.0]")
                except (ValueError, TypeError):
                    errors.append(f"Record {i+1}: Invalid confidence value")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "record_count": len(data)
        }
