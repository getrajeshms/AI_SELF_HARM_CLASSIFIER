import streamlit as st
import pandas as pd
import io
import json
import time
from typing import List, Dict, Optional
from medical_classifier import MedicalClassifier
from data_handler import DataHandler

# Page configuration
st.set_page_config(
    page_title="Medical Classification System",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'classifications' not in st.session_state:
    st.session_state.classifications = []
if 'original_data' not in st.session_state:
    st.session_state.original_data = []
if 'classifier' not in st.session_state:
    st.session_state.classifier = MedicalClassifier()
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = DataHandler()
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ''
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ''

def main():
    st.title("🏥 Medical Classification System")
    st.markdown("### AI-Powered Self-Harm Description Categorization for Medico-Legal Analysis")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Upload & Classify", "Review & Edit", "Export Results"])
    
    # AI Provider selection in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ AI Configuration")
    ai_provider = st.sidebar.selectbox(
        "Select AI Provider:",
        ["Gemini", "OpenAI"],
        help="Choose between Gemini or OpenAI for classification"
    )
    st.session_state.classifier.set_provider(ai_provider)
    
    # API Key Input
    st.sidebar.markdown("**API Key Setup:**")
    
    if ai_provider == "Gemini":
        gemini_key = st.sidebar.text_input(
            "Enter Gemini API Key:",
            type="password",
            value=st.session_state.gemini_api_key,
            help="Get your free API key from https://aistudio.google.com/app/apikey",
            placeholder="Enter your Gemini API key here..."
        )
        if gemini_key and gemini_key != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = gemini_key
            st.session_state.classifier.set_api_key("Gemini", gemini_key)
        elif gemini_key:
            st.session_state.gemini_api_key = gemini_key
            st.session_state.classifier.set_api_key("Gemini", gemini_key)
    else:
        openai_key = st.sidebar.text_input(
            "Enter OpenAI API Key:",
            type="password",
            value=st.session_state.openai_api_key,
            help="Get your API key from https://platform.openai.com/api-keys",
            placeholder="Enter your OpenAI API key here..."
        )
        if openai_key and openai_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = openai_key
            st.session_state.classifier.set_api_key("OpenAI", openai_key)
        elif openai_key:
            st.session_state.openai_api_key = openai_key
            st.session_state.classifier.set_api_key("OpenAI", openai_key)
    
    # Show API key status
    api_status = st.session_state.classifier.check_api_key()
    has_user_key = (ai_provider == "Gemini" and st.session_state.gemini_api_key) or \
                   (ai_provider == "OpenAI" and st.session_state.openai_api_key)
    has_env_key = st.session_state.classifier.check_env_key()
    
    if has_user_key:
        st.sidebar.success(f"✅ {ai_provider} API Key: Active (User Provided)")
    elif has_env_key and api_status:
        st.sidebar.success(f"✅ {ai_provider} API Key: Active (Environment)")
    else:
        st.sidebar.warning(f"⚠️ {ai_provider} API Key: Not Set")
        st.sidebar.info(f"Enter your API key above to get started")
    
    if page == "Upload & Classify":
        upload_and_classify_page()
    elif page == "Review & Edit":
        review_and_edit_page()
    elif page == "Export Results":
        export_results_page()

def upload_and_classify_page():
    st.header("📁 Upload Data and Classify")
    
    # Display predefined categories
    with st.expander("📋 View Comprehensive ICD-10 Based Classification Categories"):
        category_data = []
        for main_cat, subcats in st.session_state.classifier.category_structure.items():
            for subcat, details in subcats.items():
                detail_text = ", ".join(details) if details else "-"
                category_data.append({
                    "Main Category": main_cat,
                    "Subcategory": subcat,
                    "Specific Types": detail_text
                })
        categories_df = pd.DataFrame(category_data)
        st.dataframe(categories_df, use_container_width=True, height=400)
    
    # File upload options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
                
                # Show preview of first few rows
                st.write("**File Preview:**")
                st.dataframe(df.head(), use_container_width=True)
                
                # Select column containing descriptions
                description_column = st.selectbox("Select column containing self-harm descriptions:", df.columns)
                
                # Optional: Select serial number column
                slno_column = st.selectbox(
                    "Select serial number column (optional):", 
                    ["None"] + list(df.columns),
                    help="This will be included in the results for reference"
                )
                
                if st.button("🔍 Process CSV Data", type="primary"):
                    # Prepare data with optional serial numbers
                    descriptions = df[description_column].dropna().tolist()
                    slnos = None
                    if slno_column != "None":
                        slnos = df[slno_column].tolist()[:len(descriptions)]
                    
                    process_data(descriptions, "CSV Upload", slnos)
                    
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
    
    with col2:
        st.subheader("✏️ Manual Text Input")
        
        # Demo button
        if st.button("📋 Load Sample Data", type="secondary"):
            sample_descriptions = [
                "consumption of unknown poison",
                "insecticide consumption in agricultural field",
                "aluminum phosphide tablets ingestion",
                "consumption of unknown tablets approximately 5-6 tablets",
                "attempted hanging with rope",
                "burns injury to both hands",
                "gas cylinder injury with possible suicidal intent"
            ]
            st.session_state.sample_text = '\n'.join(sample_descriptions)
        
        manual_text = st.text_area(
            "Enter self-harm descriptions (one per line):", 
            value=st.session_state.get('sample_text', ''),
            height=200,
            help="Enter each description on a new line"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔍 Process Manual Input", type="primary"):
                if manual_text.strip():
                    descriptions = [line.strip() for line in manual_text.split('\n') if line.strip()]
                    process_data(descriptions, "Manual Input")
                else:
                    st.warning("⚠️ Please enter some text to process.")
        
        with col_b:
            if st.button("🔄 Clear Text"):
                st.session_state.sample_text = ''
                st.rerun()

def process_data(descriptions: List[str], source: str, slnos: Optional[List] = None):
    """Process the uploaded descriptions using AI classification"""
    if not descriptions:
        st.error("❌ No valid descriptions found to process.")
        return
    
    # Check API key
    if not st.session_state.classifier.check_api_key():
        provider = st.session_state.classifier.get_current_provider()
        st.error(f"❌ {provider} API key not found. Please set the {provider.upper()}_API_KEY environment variable.")
        return
    
    # Show rate limit warning for large batches
    if len(descriptions) > 10:
        st.warning("⚠️ Large batch detected. Processing will include automatic delays to avoid rate limits.")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    rate_limit_text = st.empty()
    
    # Store original data
    st.session_state.original_data = descriptions
    st.session_state.classifications = []
    
    for i, description in enumerate(descriptions):
        status_text.text(f"Processing description {i+1} of {len(descriptions)}...")
        progress_bar.progress((i + 1) / len(descriptions))
        
        # Add delay between requests to avoid rate limits
        if i > 0 and i % 4 == 0:  # After every 4 requests
            rate_limit_text.text("⏳ Pausing to respect API rate limits...")
            time.sleep(12)  # Wait 12 seconds
            rate_limit_text.empty()
        
        try:
            result = st.session_state.classifier.classify_description(description)
            
            # Prepare the classification entry
            classification_entry = {
                "Original Description": description,
                "Main Category": result.get("main_category", "J. Unclear/Undetermined Method"),
                "Subcategory": result["category"],
                "Confidence": result["confidence"],
                "Reasoning": result["reasoning"],
                "Manual Override Main": "",
                "Manual Override Sub": "",
                "Status": "AI Classified"
            }
            
            # Add serial number if provided
            if slnos and i < len(slnos):
                classification_entry["Serial Number"] = slnos[i]
            
            # Check if it's a rate limit message
            if "Rate limit exceeded" in result["reasoning"] or "rate limit" in result["reasoning"].lower():
                classification_entry["Status"] = "Rate Limited"
            
            st.session_state.classifications.append(classification_entry)
            
        except Exception as e:
            classification_entry = {
                "Original Description": description,
                "Main Category": "J. Unclear/Undetermined Method",
                "Subcategory": "Other/Unclear Method",
                "Confidence": 0.0,
                "Reasoning": f"Classification failed: {str(e)}",
                "Manual Override Main": "",
                "Manual Override Sub": "",
                "Status": "Failed"
            }
            
            if slnos and i < len(slnos):
                classification_entry["Serial Number"] = slnos[i]
                
            st.session_state.classifications.append(classification_entry)
    
    status_text.text("✅ Processing complete!")
    rate_limit_text.empty()
    progress_bar.progress(1.0)
    
    # Show results summary
    successful = len([c for c in st.session_state.classifications if c["Status"] == "AI Classified"])
    rate_limited = len([c for c in st.session_state.classifications if c["Status"] == "Rate Limited"])
    failed = len([c for c in st.session_state.classifications if c["Status"] == "Failed"])
    
    if successful > 0:
        st.success(f"🎉 Successfully processed {successful} descriptions from {source}")
    if rate_limited > 0:
        st.warning(f"⚠️ {rate_limited} descriptions hit rate limits. You can retry these later.")
    if failed > 0:
        st.error(f"❌ {failed} descriptions failed to process.")
    
    st.info("💡 Go to 'Review & Edit' page to review and modify classifications.")

def review_and_edit_page():
    st.header("✏️ Review and Edit Classifications")
    
    if not st.session_state.classifications:
        st.warning("⚠️ No classifications available. Please upload and process data first.")
        return
    
    st.subheader("📊 Classification Summary")
    df = pd.DataFrame(st.session_state.classifications)
    
    # Summary statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Classifications", len(df))
    with col2:
        successful = len(df[df["Status"] == "AI Classified"])
        st.metric("Successful", successful)
    with col3:
        failed = len(df[df["Status"] == "Failed"])
        st.metric("Failed", failed)
    with col4:
        rate_limited = len(df[df["Status"] == "Rate Limited"])
        st.metric("Rate Limited", rate_limited)
    with col5:
        successful_df = df[df["Status"] == "AI Classified"]
        avg_confidence = successful_df["Confidence"].mean() if len(successful_df) > 0 else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Add retry button for rate limited items
    if rate_limited > 0:
        if st.button("🔄 Retry Rate Limited Classifications", type="secondary"):
            retry_rate_limited_classifications()
    
    # Category distribution
    if successful > 0:
        # Get final main categories (considering manual overrides)
        final_main_categories = []
        for _, row in df.iterrows():
            if row.get('Manual Override Main', ''):
                final_main_categories.append(row['Manual Override Main'])
            elif row['Status'] == 'AI Classified':
                final_main_categories.append(row['Main Category'])
        
        if final_main_categories:
            category_counts = pd.Series(final_main_categories).value_counts()
            st.subheader("📈 Main Category Distribution")
            st.bar_chart(category_counts)
    
    # Editable data table
    st.subheader("🔍 Review Individual Classifications")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Filter by Status:", ["All", "AI Classified", "Failed", "Rate Limited", "Manual Override"])
    with col2:
        confidence_threshold = st.slider("Minimum Confidence:", 0.0, 1.0, 0.0, 0.1)
    
    # Apply filters
    filtered_df = df.copy()
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["Status"] == status_filter]
    if confidence_threshold > 0:
        filtered_df = filtered_df[filtered_df["Confidence"] >= confidence_threshold]
    
    st.write(f"Showing {len(filtered_df)} of {len(df)} classifications")
    
    # Display and edit classifications
    for i, (idx, row) in enumerate(filtered_df.iterrows()):
        with st.expander(f"Description {i + 1}: {row['Original Description'][:80]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Description:**")
                st.write(row["Original Description"])
                
                # Show serial number if available
                if "Serial Number" in row and pd.notna(row["Serial Number"]):
                    st.write(f"**Serial Number:** {row['Serial Number']}")
                
                st.write("**AI Classification:**")
                st.write(f"Main Category: {row['Main Category']}")
                st.write(f"Subcategory: {row['Subcategory']}")
                st.write(f"Confidence: {row['Confidence']:.2f}")
                st.write(f"Status: {row['Status']}")
                
                if row['Reasoning']:
                    with st.expander("View AI Reasoning"):
                        st.write(row['Reasoning'])
            
            with col2:
                st.write("**Manual Override:**")
                
                # Main category selection
                main_categories = list(st.session_state.classifier.category_structure.keys())
                current_main_override = row.get("Manual Override Main", "")
                main_default_index = 0
                if current_main_override and current_main_override in main_categories:
                    main_default_index = main_categories.index(current_main_override) + 1
                
                override_main = st.selectbox(
                    "Main Category:",
                    [""] + main_categories,
                    key=f"override_main_{idx}",
                    index=main_default_index
                )
                
                # Subcategory selection based on main category
                if override_main:
                    subcategories = list(st.session_state.classifier.category_structure[override_main].keys())
                    current_sub_override = row.get("Manual Override Sub", "")
                    sub_default_index = 0
                    if current_sub_override and current_sub_override in subcategories:
                        sub_default_index = subcategories.index(current_sub_override) + 1
                    
                    override_sub = st.selectbox(
                        "Subcategory:",
                        [""] + subcategories,
                        key=f"override_sub_{idx}",
                        index=sub_default_index
                    )
                else:
                    override_sub = ""
                
                if st.button(f"Apply Override", key=f"apply_{idx}"):
                    if override_main and override_sub:
                        st.session_state.classifications[idx]["Manual Override Main"] = override_main
                        st.session_state.classifications[idx]["Manual Override Sub"] = override_sub
                        st.session_state.classifications[idx]["Status"] = "Manual Override"
                        st.success("✅ Override applied!")
                        st.rerun()
                    elif override_main or override_sub:
                        st.warning("⚠️ Please select both main category and subcategory")

def retry_rate_limited_classifications():
    """Retry classifications that were rate limited"""
    if not st.session_state.classifier.check_api_key():
        provider = st.session_state.classifier.get_current_provider()
        st.error(f"❌ {provider} API key not found. Cannot retry classifications.")
        return
    
    rate_limited_indices = []
    for i, classification in enumerate(st.session_state.classifications):
        if classification["Status"] == "Rate Limited":
            rate_limited_indices.append(i)
    
    if not rate_limited_indices:
        st.info("ℹ️ No rate limited classifications to retry.")
        return
    
    st.info(f"🔄 Retrying {len(rate_limited_indices)} rate limited classifications...")
    progress_bar = st.progress(0)
    
    for i, idx in enumerate(rate_limited_indices):
        progress_bar.progress((i + 1) / len(rate_limited_indices))
        
        description = st.session_state.classifications[idx]["Original Description"]
        
        try:
            result = st.session_state.classifier.classify_description(description)
            
            st.session_state.classifications[idx].update({
                "Main Category": result.get("main_category", "J. Unclear/Undetermined Method"),
                "Subcategory": result["category"],
                "Confidence": result["confidence"],
                "Reasoning": result["reasoning"],
                "Status": "AI Classified"
            })
            
            # Add delay between retries
            if i < len(rate_limited_indices) - 1:
                time.sleep(2)
                
        except Exception as e:
            st.session_state.classifications[idx].update({
                "Main Category": "J. Unclear/Undetermined Method",
                "Subcategory": "Other/Unclear Method",
                "Confidence": 0.0,
                "Reasoning": f"Retry failed: {str(e)}",
                "Status": "Failed"
            })
    
    st.success("✅ Retry complete! Refresh the page to see updated results.")
    progress_bar.progress(1.0)

def export_results_page():
    st.header("📤 Export Results")
    
    if not st.session_state.classifications:
        st.warning("⚠️ No classifications available to export. Please process some data first.")
        return
    
    df = pd.DataFrame(st.session_state.classifications)
    
    # Summary Report
    st.subheader("📊 Final Summary Report")
    summary = st.session_state.data_handler.create_summary_report(st.session_state.classifications)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Processed", summary["total_processed"])
        st.metric("Success Rate", f"{summary['success_rate']}%")
    with col2:
        st.metric("Successful Classifications", summary["successful_classifications"])
        st.metric("Manual Overrides", summary["manual_overrides"])
    with col3:
        st.metric("Average Confidence", f"{summary['confidence_stats']['average']:.2f}")
        st.metric("High Confidence (≥0.8)", summary['confidence_stats']['high_confidence_count'])
    
    # Final Category Distribution
    st.subheader("📈 Final Category Distribution")
    if summary["category_distribution"]:
        category_df = pd.DataFrame(
            list(summary["category_distribution"].items()),
            columns=["Category", "Count"]
        )
        st.bar_chart(category_df.set_index("Category"))
        st.dataframe(category_df, use_container_width=True)
    
    # Export Options
    st.subheader("💾 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Complete Results (CSV)**")
        st.write("Includes all classifications with AI reasoning and manual overrides")
        
        # Prepare final results with resolved categories
        export_data = []
        for classification in st.session_state.classifications:
            final_main = classification.get("Manual Override Main", "") or classification["Main Category"]
            final_sub = classification.get("Manual Override Sub", "") or classification["Subcategory"]
            
            export_entry = {
                "Original Description": classification["Original Description"],
                "Final Main Category": final_main,
                "Final Subcategory": final_sub,
                "AI Main Category": classification["Main Category"],
                "AI Subcategory": classification["Subcategory"],
                "Confidence Score": classification["Confidence"],
                "Manual Override Main": classification.get("Manual Override Main", ""),
                "Manual Override Sub": classification.get("Manual Override Sub", ""),
                "Status": classification["Status"],
                "AI Reasoning": classification["Reasoning"]
            }
            
            # Add serial number if available
            if "Serial Number" in classification:
                export_entry["Serial Number"] = classification["Serial Number"]
            
            export_data.append(export_entry)
        
        export_df = pd.DataFrame(export_data)
        csv_data = export_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Complete Results",
            data=csv_data,
            file_name="medical_classification_results.csv",
            mime="text/csv",
            type="primary"
        )
    
    with col2:
        st.write("**Summary Report (JSON)**")
        st.write("Statistical summary with category distribution")
        
        summary_json = json.dumps(summary, indent=2)
        
        st.download_button(
            label="📥 Download Summary Report",
            data=summary_json,
            file_name="classification_summary.json",
            mime="application/json"
        )
    
    # Preview of export data
    st.subheader("👀 Export Preview")
    st.write("**First 10 rows of export data:**")
    st.dataframe(export_df.head(10), use_container_width=True)
    
    # Data quality insights
    st.subheader("🔍 Data Quality Insights")
    
    quality_col1, quality_col2 = st.columns(2)
    
    with quality_col1:
        st.write("**Confidence Distribution:**")
        confidence_ranges = {
            "Very High (0.9-1.0)": len(df[df["Confidence"] >= 0.9]),
            "High (0.8-0.9)": len(df[(df["Confidence"] >= 0.8) & (df["Confidence"] < 0.9)]),
            "Medium (0.6-0.8)": len(df[(df["Confidence"] >= 0.6) & (df["Confidence"] < 0.8)]),
            "Low (<0.6)": len(df[df["Confidence"] < 0.6])
        }
        for range_name, count in confidence_ranges.items():
            st.write(f"- {range_name}: {count}")
    
    with quality_col2:
        st.write("**Manual Review Recommendations:**")
        low_confidence = len(df[df["Confidence"] < 0.7])
        failed_count = len(df[df["Status"] == "Failed"])
        
        if low_confidence > 0:
            st.write(f"- {low_confidence} classifications with confidence < 0.7 may need review")
        if failed_count > 0:
            st.write(f"- {failed_count} failed classifications need attention")
        if low_confidence == 0 and failed_count == 0:
            st.write("- All classifications appear to be high quality ✅")

if __name__ == "__main__":
    main()
