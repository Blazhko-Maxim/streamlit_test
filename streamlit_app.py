import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import requests
import json
import torch
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

# Load CLIP model
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

# Connect to Google Sheet
@st.cache_resource
def load_sheet():
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('F:\Rmt\Style\\feed\google_creds.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open("StyleApp test feed").sheet1
    return sheet

sheet = load_sheet()

# --- UI ---
st.title("Outfit Duplicate Checker")

image_url = st.text_input("Image Path (URL)")
gender = st.text_input("Gender")
style = st.text_input("Style")
credits = st.text_input("Credits")
credits_link = st.text_input("Credits Link")

submit = st.button("Submit")

# --- Helper ---
def get_image_embedding(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        return features[0].numpy()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def fetch_existing_embeddings():
    records = sheet.get_all_records()
    embeddings = []
    urls = []
    for row in records:
        try:
            emb = np.array(json.loads(row.get("Embedding")))
            embeddings.append(emb)
            urls.append(row.get("Image URL"))
        except:
            continue
    return urls, np.array(embeddings), records

def clicked():
    st.session_state["submit_anyway"] = True

if st.session_state.get("submit_anyway", False):
    st.session_state["submit_anyway"] = False
    new_row = [
        image_url,
        gender,
        style,
        credits,
        credits_link,
        json.dumps(st.session_state["similar_urls"]),
        json.dumps(st.session_state["new_emb"])
    ]
    sheet.append_row(new_row)
    st.success("Image submitted successfully!")
    st.stop()

# --- Submission ---
if submit and image_url:
    with st.spinner("Checking for duplicates..."):
        new_emb = get_image_embedding(image_url)
        if new_emb is None:
            st.stop()

        urls, embeddings, records = fetch_existing_embeddings()

        similar_urls = []
        if len(embeddings) > 0:
            sims = cosine_similarity([new_emb], embeddings)[0]
            similar_urls = [urls[i] for i, sim in enumerate(sims) if sim > 0.9]

        if similar_urls:
            st.session_state["similar_urls"] = similar_urls
            st.session_state["new_emb"] = new_emb.tolist()
            st.warning("This image looks very similar to others already in the sheet:")
            for link in similar_urls:
                st.markdown(f"- [{link}]({link})")

            # Create two explicit buttons
            col1, col2 = st.columns(2)
            with col1:
                submit_anyway = st.button("Submit Anyway", on_click=clicked)
            with col2:
                cancel_submit = st.button("Cancel")

        else:
            new_row = [
                image_url,
                gender,
                style,
                credits,
                credits_link,
                json.dumps(similar_urls),
                json.dumps(new_emb.tolist())
            ]
            sheet.append_row(new_row)
            st.success("Image submitted successfully!")