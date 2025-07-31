import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import requests
import json
import pandas as pd
import torch
import os
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
    creds = {
        "type": "service_account",
        "project_id": os.environ.get("GOOGLE_PROJECT_ID"),
        "private_key_id": os.environ.get("GOOGLE_PRIVATE_KEY_ID"),
        "private_key": os.environ.get("GOOGLE_PRIVATE_KEY"),
        "client_email": os.environ.get("GOOGLE_CLIENT_EMAIL"),
        "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.environ.get("GOOGLE_CLIENT_X509_CERT_URL"),
        "universe_domain": "googleapis.com"
    }
    with open('google_creds.json', 'w') as file:
        json.dump(creds, file)
    creds = ServiceAccountCredentials.from_json_keyfile_name('google_creds.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open("StyleApp test feed").sheet1
    return sheet

sheet = load_sheet()

def get_stats():
    values = sheet.get_values("B:C", "COLUMNS")
    df = pd.DataFrame({'Gender': values[0][1:], 'Style': values[1][1:]})
    df.sort_values(['Gender', 'Style'], inplace=True)
    stats = df.groupby(['Gender', 'Style']).size().reset_index(name='Count')
    st.title("Statistics")
    st.table(stats)
    st.button("Back")
    st.stop()


# --- UI ---
st.title("Outfit Duplicate Checker")
st.button("Get Stats", on_click=get_stats)

image_url = st.text_input("Image Path (URL)")
gender = st.selectbox("Gender", ["Male", "Female"])
if gender == "Male":
    styles = [
        "Casual", "Grunge", "Soft Aesthetics", "Old Money", "Vintage and Retro", 
        "Minimalism", "Sporty", "Streetwear", "Workwear", "Y2K"
    ]
else:
    styles = [
        "Casual", "Chic and Classy", "Bohemian", "Elegant", "Gothic Aesthetics", "Artsy and Electric", 
        "Minimalistic", "Romantic", "Sporty", "Streetwear", "Edgy", "Vintage and Retro"
    ]
style =  st.selectbox("Style", styles)
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
    columns = [c[1:] for c in sheet.batch_get(["A:A", "G:G"])]
    embeddings = []
    urls = []
    for url_row, emb_row in zip(columns[0], columns[1]):
        if url_row and emb_row:
            emb = np.array(json.loads(emb_row[0]))
            embeddings.append(emb)
            urls.append(url_row[0])
    return urls, np.array(embeddings)

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

        urls, embeddings = fetch_existing_embeddings()

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