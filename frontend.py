import streamlit as st
import requests
import json
import pandas as pd
from typing import List, Dict
from PIL import Image
import io

st.set_page_config(
    page_title="NeuroCare AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8000"

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def analyze_drug_interactions(drugs: List[str], age: int, conditions: List[str] = []):
    """Call backend API for drug interaction analysis"""
    try:
        payload = {
            "drugs": drugs,
            "patient_age": age,
            "medical_conditions": conditions
        }
        response = requests.post(f"{API_BASE_URL}/analyze_interactions", 
                               json=payload, timeout=30)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def extract_drugs_from_text(text: str):
    """Call backend API for drug extraction from text"""
    try:
        payload = {"medical_text": text}
        response = requests.post(f"{API_BASE_URL}/extract_drugs", 
                               json=payload, timeout=30)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def extract_drugs_from_image(image_file):
    """Call backend API for drug extraction from prescription image"""
    try:
        files = {"file": ("prescription.jpg", image_file, "image/jpeg")}
        response = requests.post(f"{API_BASE_URL}/extract_from_image", 
                               files=files, timeout=60)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def display_interactions(interactions: List[Dict]):
    """Display drug interactions in a formatted way"""
    if not interactions:
        st.success("✅ No significant drug interactions detected")
        return
    
    st.subheader("🚨 Drug Interactions Found")
    
    for i, interaction in enumerate(interactions):
        severity = interaction.get('severity', 'unknown').upper()
        
        with st.container():
            if severity == 'SEVERE':
                st.error(f"**🔴 SEVERE INTERACTION #{i+1}**")
                border_color = "red"
            elif severity == 'MODERATE':
                st.warning(f"**🟡 MODERATE INTERACTION #{i+1}**")
                border_color = "orange"
            else:
                st.info(f"**🟢 MILD INTERACTION #{i+1}**")
                border_color = "green"
            
            with st.expander(f"Details for {interaction.get('drug1', 'N/A')} ↔ {interaction.get('drug2', 'N/A')}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**💊 Drugs:** {interaction.get('drug1', 'N/A')} ↔ {interaction.get('drug2', 'N/A')}")
                    st.markdown(f"**⚠️ Severity:** {severity}")
                
                with col2:
                    st.markdown(f"**📋 Description:** {interaction.get('description', 'N/A')}")
                    if interaction.get('mechanism'):
                        st.markdown(f"**⚙️ Mechanism:** {interaction.get('mechanism')}")

def display_alternatives(alternatives: List[Dict]):
    """Display alternative medication suggestions"""
    if not alternatives:
        st.info("💡 No alternative medications suggested at this time")
        return
    
    st.subheader("💡 Alternative Medication Suggestions")
    
    for i, alt in enumerate(alternatives):
        with st.expander(f"Alternative #{i+1}: {alt.get('alternative', 'N/A')}", expanded=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"**🚫 Original:** {alt.get('original_drug', 'N/A')}")
                st.markdown(f"**✅ Alternative:** {alt.get('alternative', 'N/A')}")
            
            with col2:
                st.markdown(f"**🎯 Reason:** {alt.get('reason', 'N/A')}")
                age_appropriate = alt.get('age_appropriate', True)
                st.markdown(f"**👤 Age Appropriate:** {'✅ Yes' if age_appropriate else '❌ No'}")

def display_warnings(warnings: List[str], warning_type: str):
    """Display safety warnings"""
    if not warnings:
        return
    
    st.subheader(f"⚠️ {warning_type}")
    for i, warning in enumerate(warnings):
        with st.container():
            st.warning(f"**Warning #{i+1}:** {warning}")

def perform_interaction_analysis(drugs, patient_age, medical_conditions):
    """Shared function to perform interaction analysis"""
    with st.spinner("🔍 Analyzing drug interactions..."):
        result, error = analyze_drug_interactions(drugs, patient_age, medical_conditions)
    
    if error:
        st.error(f"❌ Analysis failed: {error}")
    else:
        st.markdown("---")
        st.markdown("# 📊 Analysis Results")
        
        display_interactions(result.get('interactions', []))
        
        col1, col2 = st.columns(2)
        with col1:
            display_alternatives(result.get('alternatives', []))
        
        with col2:
            display_warnings(result.get('safety_warnings', []), "General Safety Warnings")
            display_warnings(result.get('age_specific_warnings', []), 
                           f"Age-Specific Warnings ({patient_age} years old)")

# Main UI
def main():
    st.title("💊 NeuroCare - AI Medical Prescription Verification")
    st.markdown("### 🔬 Analyze drug interactions and get safe alternative recommendations")
    
    if not check_backend_health():
        st.error("🔴 Backend server is not running. Please start the FastAPI backend first.")
        st.code("python app.py", language="bash")
        return
    
    st.success("🟢 Connected to backend server")
    
    # Initialize session state for extracted drugs
    if 'extracted_drugs' not in st.session_state:
        st.session_state.extracted_drugs = []
    if 'switch_to_analysis' not in st.session_state:
        st.session_state.switch_to_analysis = False
    
    with st.sidebar:
        st.markdown("# 🏥 Medical Dashboard")
        
        with st.container():
            st.markdown("## 👤 Patient Information")
            patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=30)
            
            if patient_age < 18:
                st.info("👶 Pediatric Patient")
            elif patient_age >= 65:
                st.info("👴 Geriatric Patient")
            else:
                st.info("👨 Adult Patient")
        
        with st.container():
            st.markdown("## 🏥 Medical Conditions")
            conditions_text = st.text_area("Enter medical conditions (one per line)", 
                                         placeholder="Diabetes\nHypertension\nAsthma",
                                         height=100)
            medical_conditions = [c.strip() for c in conditions_text.split('\n') if c.strip()]
            
            if medical_conditions:
                st.success(f"📋 {len(medical_conditions)} condition(s) noted")
        
        with st.container():
            st.markdown("## 💊 Common Drug Classes")
            with st.expander("🫀 Cardiovascular"):
                st.markdown("""
                - **ACE Inhibitors**: Lisinopril, Enalapril
                - **Beta Blockers**: Metoprolol, Propranolol
                - **Diuretics**: Furosemide, HCTZ
                - **Statins**: Atorvastatin, Simvastatin
                """)
            
            with st.expander("🧠 Neurological"):
                st.markdown("""
                - **Anticonvulsants**: Phenytoin, Carbamazepine
                - **Antidepressants**: Sertraline, Fluoxetine
                - **Pain Relief**: Gabapentin, Pregabalin
                """)
            
            with st.expander("🍭 Endocrine"):
                st.markdown("""
                - **Diabetes**: Metformin, Insulin
                - **Thyroid**: Levothyroxine, Methimazole
                - **Steroids**: Prednisone, Hydrocortisone
                """)
        
        with st.container():
            st.markdown("## ⚠️ Safety Reminders")
            st.warning("Always consult healthcare providers before making medication changes")
            st.info("This tool is for educational purposes only")
            st.error("In case of emergency, call 108 immediately")
        
        with st.container():
            st.markdown("## 📊 Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Patient Age", f"{patient_age} yrs")
            with col2:
                st.metric("Conditions", len(medical_conditions))
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Drug Interaction Analysis", "📝 Extract Drugs from Text/Image"])
    
    # If we need to switch to analysis tab
    if st.session_state.switch_to_analysis:
        st.session_state.switch_to_analysis = False
        # Force switch to tab1 by setting extracted drugs and showing results
    
    with tab1:
        st.markdown("# 🔬 Drug Interaction Analysis")
        st.markdown("**Comprehensive analysis of potential drug interactions and safety concerns**")
        
        # Check if we have extracted drugs from the other tab
        if st.session_state.extracted_drugs:
            with st.container():
                st.success(f"🎯 **Imported {len(st.session_state.extracted_drugs)} drug(s) from extraction:**")
                col1, col2 = st.columns([3, 1])
                with col1:
                    drugs_display = ", ".join(st.session_state.extracted_drugs)
                    st.markdown(f"💊 **{drugs_display}**")
                with col2:
                    if st.button("🔄 Clear Imported"):
                        st.session_state.extracted_drugs = []
                        st.rerun()
                
                if len(st.session_state.extracted_drugs) >= 2:
                    analyze_imported = st.button("🔬 Analyze Imported Drugs", type="primary", use_container_width=True)
                    if analyze_imported:
                        perform_interaction_analysis(st.session_state.extracted_drugs, patient_age, medical_conditions)
                        return
                
                st.divider()
        
        with st.container():
            st.markdown("### 💊 Drug Input Method")
            input_method = st.radio("How would you like to input drugs?", 
                                   ["✍️ Manual Entry", "📁 Upload List"],
                                   horizontal=True)
        
        drugs = []
        
        if input_method == "✍️ Manual Entry":
            with st.container():
                st.markdown("### 📝 Enter Drug Names")
                col1, col2 = st.columns([4, 1])
                with col1:
                    drug_input = st.text_input("Enter drug names (separate with commas)", 
                                             placeholder="Aspirin, Warfarin, Metformin, Lisinopril")
                with col2:
                    st.markdown("**Format:**")
                    st.caption("Drug1, Drug2, Drug3")
                    
                if drug_input:
                    drugs = [drug.strip() for drug in drug_input.split(',') if drug.strip()]
                    with st.container():
                        st.success(f"**✅ Ready to analyze {len(drugs)} drug(s):**")
                        for drug in drugs:
                            st.markdown(f"💊 **{drug}**")
        
        else:
            with st.container():
                st.markdown("### 📁 Upload Drug List")
                uploaded_file = st.file_uploader("Upload a text file with drug names", 
                                               type=['txt'], 
                                               help="📋 One drug name per line")
                if uploaded_file:
                    content = uploaded_file.read().decode('utf-8')
                    drugs = [line.strip() for line in content.split('\n') if line.strip()]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**✅ Loaded {len(drugs)} drugs successfully**")
                    with col2:
                        st.info(f"**Preview:** {', '.join(drugs[:3])}" + 
                               (f" and {len(drugs)-3} more..." if len(drugs) > 3 else ""))
        
        if drugs and len(drugs) >= 2:
            st.divider()
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    analyze_btn = st.button("🔬 Analyze Drug Interactions", type="primary", use_container_width=True)
                with col2:
                    st.metric("Drugs", len(drugs))
                with col3:
                    st.metric("Patient Age", f"{patient_age} yrs")
                
                if analyze_btn:
                    perform_interaction_analysis(drugs, patient_age, medical_conditions)
        
        elif drugs and len(drugs) == 1:
            st.info("ℹ️ Please enter at least 2 drugs to analyze interactions")
        else:
            st.info("💡 Please enter drug names to begin analysis")
    
    with tab2:
        st.markdown("# 📝 Extract Drugs from Medical Text or Prescription Image")
        st.markdown("**Use AI to automatically extract drug names from prescription notes, medical text, or prescription images**")
        
        # Input method selection
        with st.container():
            st.markdown("### 📥 Input Method")
            extraction_method = st.radio("Choose extraction method:", 
                                       ["📝 Text Input", "📷 Image Upload"], 
                                       horizontal=True)
        
        extracted_drugs = []
        
        if extraction_method == "📝 Text Input":
            with st.container():
                st.markdown("### 📋 Medical Text Input")
                col1, col2 = st.columns([3, 1])
                with col1:
                    medical_text = st.text_area("Enter medical text", 
                                               placeholder="Patient prescribed Lisinopril 10mg daily for hypertension. Also taking Metformin 500mg twice daily for diabetes. Consider adding Aspirin 81mg for cardiac protection...",
                                               height=150)
                with col2:
                    st.markdown("**💡 Tips:**")
                    st.caption("• Include dosages")
                    st.caption("• Mention conditions")
                    st.caption("• Add frequency info")
                    st.caption("• Use medical terminology")
            
            if medical_text:
                char_count = len(medical_text)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Characters", char_count)
                with col2:
                    if char_count > 20:
                        st.success("✅ Ready for extraction")
                    else:
                        st.warning("⚠️ Add more text")
                
                if st.button("🔍 Extract Drug Names from Text", type="primary", use_container_width=True):
                    with st.spinner("🤖 Extracting drug names using AI..."):
                        result, error = extract_drugs_from_text(medical_text)
                    
                    if error:
                        st.error(f"❌ Extraction failed: {error}")
                    else:
                        extracted_drugs = result.get('extracted_drugs', [])
                        nlp_candidates = result.get('nlp_candidates', [])
                        
                        st.markdown("---")
                        st.markdown("# 🎯 Extraction Results")
                        
                        if extracted_drugs:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### 🎯 Validated Drugs")
                                st.success(f"Found {len(extracted_drugs)} validated drug(s)")
                                for drug in extracted_drugs:
                                    st.markdown(f"💊 **{drug}**")
                            
                            with col2:
                                st.markdown("### 🤖 NLP Candidates")
                                if nlp_candidates:
                                    st.info(f"Found {len(nlp_candidates)} candidate(s)")
                                    for candidate in nlp_candidates:
                                        st.markdown(f"🔍 *{candidate}*")
                                else:
                                    st.info("No additional candidates found")
                        else:
                            st.warning("⚠️ No drugs found in the text. Try adding more specific medication names.")
            else:
                st.info("💡 Enter medical text above to extract drug names")
        
        else:  # Image Upload
            with st.container():
                st.markdown("### 📷 Prescription Image Upload")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    uploaded_image = st.file_uploader(
                        "Upload prescription image", 
                        type=['png', 'jpg', 'jpeg'],
                        help="📸 Upload a clear image of a prescription or medical document"
                    )
                
                with col2:
                    st.markdown("**💡 Image Tips:**")
                    st.caption("• Use good lighting")
                    st.caption("• Avoid shadows")
                    st.caption("• Keep text straight")
                    st.caption("• High resolution")
                    st.caption("• Clear handwriting")
                
                if uploaded_image is not None:
                    # Display the uploaded image
                    image = Image.open(uploaded_image)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.image(image, caption="Uploaded Prescription", use_column_width=True)
                    
                    with col2:
                        st.success("✅ Image uploaded successfully")
                        st.metric("Image Size", f"{image.size[0]}x{image.size[1]}")
                        st.metric("Format", image.format)
                    
                    if st.button("🔍 Extract Drugs from Image", type="primary", use_container_width=True):
                        with st.spinner("📷 Processing image and extracting drugs..."):
                            # Convert PIL image to bytes
                            img_byte_arr = io.BytesIO()
                            image.save(img_byte_arr, format='JPEG')
                            img_byte_arr.seek(0)
                            
                            result, error = extract_drugs_from_image(img_byte_arr)
                        
                        if error:
                            st.error(f"❌ Image processing failed: {error}")
                        else:
                            extracted_text = result.get('extracted_text', '')
                            extracted_drugs = result.get('extracted_drugs', [])
                            
                            st.markdown("---")
                            st.markdown("# 📷 Image Processing Results")
                            
                            # Show extracted text
                            with st.expander("📝 Extracted Text (OCR)", expanded=False):
                                if extracted_text:
                                    st.text_area("Raw OCR Text:", extracted_text, height=100, disabled=True)
                                else:
                                    st.warning("No text could be extracted from the image")
                            
                            # Show extracted drugs
                            if extracted_drugs:
                                st.markdown("### 🎯 Extracted Drugs from Image")
                                st.success(f"Found {len(extracted_drugs)} drug(s)")
                                for drug in extracted_drugs:
                                    st.markdown(f"💊 **{drug}**")
                            else:
                                st.warning("⚠️ No drugs found in the image. Try a clearer image or manual entry.")
                else:
                    st.info("📷 Upload a prescription image above to extract drug names")
        
        # Common section for both text and image extraction
        if extracted_drugs and len(extracted_drugs) >= 1:
            st.divider()
            st.markdown("### 🔬 Continue to Interaction Analysis?")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                continue_analysis = st.button("🔬 Analyze These Drugs for Interactions", type="secondary", use_container_width=True)
            with col2:
                st.metric("Extracted", len(extracted_drugs))
            with col3:
                st.metric("Patient Age", f"{patient_age} yrs")
            
            if continue_analysis:
                if len(extracted_drugs) >= 2:
                    # Store extracted drugs in session state and switch tabs
                    st.session_state.extracted_drugs = extracted_drugs
                    st.session_state.switch_to_analysis = True
                    
                    # Perform analysis directly
                    st.markdown("---")
                    st.markdown("# 📊 Drug Interaction Analysis Results")
                    perform_interaction_analysis(extracted_drugs, patient_age, medical_conditions)
                else:
                    st.warning("⚠️ Need at least 2 drugs for interaction analysis. Only found 1 drug.")
                    st.info("💡 You can manually add more drugs in the Drug Interaction Analysis tab.")
    
    st.divider()

if __name__ == "__main__":
    main()