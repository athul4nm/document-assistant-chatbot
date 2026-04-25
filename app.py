import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gc
import io
import os
import urllib.parse
from docx import Document

def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# 1. Page Configuration
st.set_page_config(page_title="Docu.Ai", page_icon="😎", layout="centered")
st.title("📜docu.Ai")
st.write("Powered by fine-tuned Mistral model!")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 2. Model Loading (Cached so it only loads once)
@st.cache_resource
def load_model():
    try:
        clear_gpu_memory()

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        
        st.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        st.info("Loading model with 4-bit quantization...")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model WITHOUT device_map first
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,  # This is critical!
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Manually move to GPU
        if torch.cuda.is_available():
            model = model.to('cuda')
            st.success(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ Model loaded on CPU")
        
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

tokenizer, model = load_model()

# 3. User Interface
doc_type = st.selectbox("Document Type:", ["Professional Email", "Resume", "Leave Letter", "Complaint Letter", "Cover Letter"])
user_input = st.text_area("What should the document be about?", height=150)
generate_btn = st.button("🚀 Generate Document")

# 4. Generation Logic
if "generated_doc" not in st.session_state:
    st.session_state.generated_doc = ""

if generate_btn and user_input:
    with st.spinner("Generating high-quality document..."):
        instruction = f"Write a {doc_type.lower()} based on the following request."
        prompt = f"<s>[INST] {instruction}\n{user_input} [/INST]"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )

            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_document = full_response.split("[/INST]")[-1].strip()
            
            # Save the result to Streamlit's memory
            st.session_state.generated_doc = final_document

        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            
        finally:
            torch.cuda.empty_cache()
            gc.collect()

# 5. Editing & Exporting Interface
if st.session_state.generated_doc:
    st.markdown("---")
    st.subheader("✏️ Review and Edit")
    
    # Text area that allows the user to edit the generated document
    edited_doc = st.text_area(
        "Make any changes below. The text will automatically updated for downloading.", 
        value=st.session_state.generated_doc, 
        height=300
    )
    
    # Create two columns for our action buttons
# Create 3 columns. The first two are small for the buttons, the third acts as empty space.
    col1, col2, col3 = st.columns([3.5, 3.5, 7])
    
    with col1:
        # Generate Word Document in memory
        def create_word_file(text):
            doc = Document()
            doc.add_paragraph(text)
            bio = io.BytesIO()
            doc.save(bio)
            return bio.getvalue()

        word_file = create_word_file(edited_doc)
        
        st.download_button(
            label="Download as Word(.docx)",
            data=word_file,
            file_name=f"{doc_type.replace(' ', '_')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        
    with col2:
        # URL-encode the subject and body for the web link
        email_subject = urllib.parse.quote(f"Generated {doc_type}")
        email_body = urllib.parse.quote(edited_doc)
        
        # Create the specific Gmail Compose URL
        gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&su={email_subject}&body={email_body}"
        
        # Create a clickable HTML button that opens Gmail
        st.markdown(
            f'<a href="{gmail_url}" target="_blank" style="display: inline-block; padding: 0.35em 1em; margin-top: 0.1em; color: white; background-color: #D44638; text-decoration: none; border-radius: 4px; font-weight: bold;">Send via Gmail📫</a>', 
            unsafe_allow_html=True
        )
