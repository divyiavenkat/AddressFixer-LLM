import streamlit as st
from transformers import TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

# Load model & tokenizer
checkpoint = "models/llama3_sft_sfttrainer_MA/checkpoint-2500"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = checkpoint,
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
    device_map = "auto"
)

tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
model.eval()

def parse_and_correct_address(raw_address):
    task1_prompt = f"""
    Parse the following address into a structured JSON with these fields:
    AddNum_Pre, Add_Number, AddNum_Suf, St_PreDir, St_Name, St_PosTyp, St_PosDir,
    Building, Floor, Unit, Room, Uninc_Comm, Inc_Muni, County, State, Zip_Code.
    Address: {raw_address}
    """

    prompt1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": task1_prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs1 = tokenizer(prompt1, return_tensors="pt").to(model.device)
    output1 = model.generate(**inputs1, max_new_tokens=512)
    parsed_text = tokenizer.decode(output1[0], skip_special_tokens=True)
    parsed_json = parsed_text.split("System:")[-1].strip()

    task2_prompt = f"""
    Fix the formatting, structure, correct any existing entities, or predict/add new values
    to the appropriate entities of this Address JSON. Expand common abbreviations (like st→street),
    correct obvious errors, generate new values, and standardize capitalization.
    Keep empty fields as empty strings. Do not return anything other than corrected Address JSON.
    Address JSON: {parsed_json}
    """

    prompt2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": task2_prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)
    output2 = model.generate(**inputs2, max_new_tokens=512)
    corrected_text = tokenizer.decode(output2[0], skip_special_tokens=True)
    corrected_json = corrected_text.split("System:")[-1].strip()

    return parsed_json, corrected_json

# Streamlit UI
st.title("📍 Address Corrector")

user_input = st.text_area(
    "Enter an address:"
    # placeholder="e.g. 386 Barnabye Strret Southeest, Sprngfeild Hights, Sprinfield, Hampdin Cnty, MA 01190"
)

if st.button("Parse & Correct"):
    with st.spinner("Thinking..."):
        parsed, corrected = parse_and_correct_address(user_input)

        # st.subheader("Parsed Address")
        # st.text(parsed)

        st.subheader("Corrected Address")
        st.text(corrected)
