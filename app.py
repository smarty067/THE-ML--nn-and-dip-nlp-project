import os
import re
import av
import pytesseract
import torch
import gradio as gr
from PIL import Image
from pypdf import PdfReader
from pdf2image import convert_from_path
from openai import OpenAI
from supabase import create_client
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz

# ==========================================
# 1. CLIENT & DATABASE SETUP
# ==========================================
# Ensure OpenRouter & Supabase variables are fetched from environment
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)
MODEL = "qwen/qwen-2.5-7b-instruct"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://eqkqflcanfbeiwykikuj.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_KEY else None
except Exception:
    supabase = None

# ==========================================
# 2. MODEL INITIALIZATION & WORKAROUNDS
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
whisper_model = WhisperModel("small", device=device, compute_type=compute_type)

# Monkey patch for Python 3.13 av compatibility
original_av_open = av.open
def patched_av_open(*args, **kwargs):
    kwargs.pop('metadata_errors', None)
    return original_av_open(*args, **kwargs)
av.open = patched_av_open

# ==========================================
# 3. CORE LOGIC SERVICES
# ==========================================
class LanguageService:
    @staticmethod
    def detect_language(text):
        if not text:
            return "English"
        if re.search(r'[ऀ-ॿ]', text):
            return "Hindi"
        elif re.search(r'[ಀ-೿]', text):
            return "Kannada"
        return "English"

def safe_ocr_file(file_path):
    if not file_path:
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in [".jpg", ".jpeg", ".png", ".webp"]:
            image = Image.open(file_path).convert("RGB")
            return pytesseract.image_to_string(image, lang="eng").strip()
        elif ext == ".pdf":
            text = ""
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text: text += page_text + "\n"
            except Exception:
                pass
            if len(text.strip()) > 30:
                return text.strip()
            pages = convert_from_path(file_path)
            for page in pages:
                text += pytesseract.image_to_string(page, lang="eng") + "\n"
            return text.strip()
    except Exception as e:
        return f"[OCR failed: {str(e)}]"
    return ""

def transcribe_mic_audio(audio_path):
    if not audio_path:
        return ""
    try:
        segments, _ = whisper_model.transcribe(audio_path, beam_size=5)
        return " ".join(segment.text for segment in segments).strip()
    except Exception as e:
        return f"[Transcription Error: {str(e)}]"

def match_all_medicines(medicine_names):
    if not supabase:
        return []
    try:
        db_meds = supabase.table("medicines").select("id, brand_name, composition, active_ingredients, strength, dosage_form").execute().data
    except Exception:
        return []

    choices = {}
    for med in db_meds:
        brand = med.get("brand_name")
        comp = med.get("composition")
        if brand: choices[brand] = med
        if comp: choices[comp] = med

    matched_records = []
    for name in medicine_names:
        match = process.extractOne(name, choices.keys(), scorer=fuzz.WRatio)
        if match and match[1] >= 70:
            matched_records.append({
                "ocr_name": name,
                "matched_name": match[0],
                "confidence": match[1],
                "record": choices[match[0]]
            })
        else:
            matched_records.append({
                "ocr_name": name,
                "matched_name": None,
                "confidence": 0,
                "record": None
            })
    return matched_records

def build_rag_context(matched_records):
    context = ""
    for item in matched_records:
        if item["record"]:
            r = item["record"]
            context += f"Verified Brand: {r.get('brand_name')}\nComposition: {r.get('composition')}\nStrength: {r.get('strength')}\n\n"
        else:
            context += f"OCR: {item['ocr_name']} (No database match found)\n\n"
    return context

# ==========================================
# 4. BOT BACKENDS
# ==========================================
def analyze_prescription_pipeline(file):
    if not file:
        return "", "⚠️ Upload prescription to proceed.", ""
    text = safe_ocr_file(file)
    try:
        prompt = f"Extract list of drug names from this prescription. Return ONLY raw lines with drug names:\n{text}"
        res = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}])
        names = [n.strip() for n in res.choices[0].message.content.split("\n") if n.strip()]
        matched = match_all_medicines(names)
        rag = build_rag_context(matched)
        md_display = "### 💊 Database Matches\n\n"
        for m in matched:
            if m["matched_name"]:
                md_display += f"✅ **{m['ocr_name']}** -> **{m['matched_name']}** ({m['confidence']:.0f}%)\n"
            else:
                md_display += f"⚠️ **{m['ocr_name']}** (Manual verification required)\n"
        return text, md_display, rag
    except Exception as e:
        return text, f"Error: {str(e)}", ""

def chat_general(msg, history, selected_lang):
    if not msg: return history, ""
    if history is None: history = []
    lang = selected_lang if selected_lang != "Auto-Detect" else LanguageService.detect_language(msg)
    sys_prompt = f"You are a neutral Clinical AI Health Assistant. Help the user in {lang}. Suggest healthy Indian foods (Idli, Dosa, Ragi, Oats, Dal, Curd). Include medical disclaimers."
    messages = [{"role":"system","content":sys_prompt}]
    for item in history:
        messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})
    messages.append({"role":"user","content":msg})
    try:
        res = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
        ans = res.choices[0].message.content
    except Exception as e:
        ans = f"Connection Error: {str(e)}"
    history.append({"role":"user","content":msg})
    history.append({"role":"assistant","content":ans})
    return history, ""

def chat_prescription(msg, history, ocr_text, rag_ctx, selected_lang):
    if not msg: return history, ""
    if history is None: history = []
    lang = selected_lang if selected_lang != "Auto-Detect" else LanguageService.detect_language(msg)
    sys_prompt = f"You are a neutral Clinical AI Health Assistant answering about a prescription. Respond in {lang}.\nPrescription:\n{ocr_text}\nDatabase:\n{rag_ctx}"
    messages = [{"role":"system","content":sys_prompt}]
    for item in history:
        messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})
    messages.append({"role":"user","content":msg})
    try:
        res = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
        ans = res.choices[0].message.content
    except Exception as e:
        ans = f"Connection Error: {str(e)}"
    history.append({"role":"user","content":msg})
    history.append({"role":"assistant","content":ans})
    return history, ""

def chat_xray(msg, history, file, selected_lang):
    if not msg: return history, ""
    if history is None: history = []
    lang = selected_lang if selected_lang != "Auto-Detect" else LanguageService.detect_language(msg)
    sys_prompt = f"You are a clinical radiologist simulator assisting with an X-ray. Focus on structural alignment, bone integrity, density, and joint spacing checks. Language: {lang}. Never provide absolute diagnosis."
    if file:
        sys_prompt += f" [Image Attached: {os.path.basename(file)}]"
    messages = [{"role":"system","content":sys_prompt}]
    for item in history:
        messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})
    messages.append({"role":"user","content":msg})
    try:
        res = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
        ans = res.choices[0].message.content
    except Exception as e:
        ans = f"Connection Error: {str(e)}"
    history.append({"role":"user","content":msg})
    history.append({"role":"assistant","content":ans})
    return history, ""

def chat_report(msg, history, file, selected_lang):
    if not msg: return history, ""
    if history is None: history = []
    lang = selected_lang if selected_lang != "Auto-Detect" else LanguageService.detect_language(msg)
    extracted = safe_ocr_file(file) if file else "No report uploaded."
    sys_prompt = f"You are a clinical lab report analyzer. Reference ranges/abnormal values of: {extracted}. Propose healthy localized Indian food choices (Ragi, Idli, Dosa, Curd, Dal, Oats). Language: {lang}. Always include strict disclaimers."
    messages = [{"role":"system","content":sys_prompt}]
    for item in history:
        messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})
    messages.append({"role":"user","content":msg})
    try:
        res = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
        ans = res.choices[0].message.content
    except Exception as e:
        ans = f"Connection Error: {str(e)}"
    history.append({"role":"user","content":msg})
    history.append({"role":"assistant","content":ans})
    return history, ""

def chat_symptoms(msg, history, selected_lang):
    if not msg: return history, ""
    if history is None: history = []
    lang = selected_lang if selected_lang != "Auto-Detect" else LanguageService.detect_language(msg)
    sys_prompt = f"You are a clinical educational symptom triage assistant. Classify severity, advise localized lifestyle or warm/light Indian diet steps (e.g. Curd Rice, Moong Dal Khichdi), and display prominent clinical liability disclaimers. Language: {lang}."
    messages = [{"role":"system","content":sys_prompt}]
    for item in history:
        messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})
    messages.append({"role":"user","content":msg})
    try:
        res = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
        ans = res.choices[0].message.content
    except Exception as e:
        ans = f"Connection Error: {str(e)}"
    history.append({"role":"user","content":msg})
    history.append({"role":"assistant","content":ans})
    return history, ""

# ==========================================
# 5. GRADIO INTERFACE LAYOUT
# ==========================================
with gr.Blocks(title="AI Health Assistant") as demo:
    gr.HTML("<div style='text-align:center; padding:1.5rem;'><h1>🏥 AI Health Assistant</h1><h3>Grounded Diagnostics & Interactive Multilingual Care Portal</h3></div>")

    with gr.Row():
        with gr.Column(scale=3):
            lang_select = gr.Radio(choices=["Auto-Detect", "English", "Hindi", "Kannada"], value="Auto-Detect", label="Select System Language Overrides")
        with gr.Column(scale=2):
            gr.Markdown("🚨 **Emergency Warning:** If you are experiencing immediate life-threatening symptoms, go to the nearest emergency clinic.")

    with gr.Tabs():
        with gr.TabItem("🏠 AI Health Chatbot"):
            chatbot_gen = gr.Chatbot(label="Conversational Assistant", height=350)
            with gr.Accordion("🎤 Voice Input", open=False):
                mic_gen = gr.Audio(sources=["microphone"], type="filepath", label="Record Question")
                transcribe_gen_btn = gr.Button("Convert Voice")
            with gr.Row():
                msg_gen = gr.Textbox(placeholder="Ask about healthy lifestyles...", lines=2, scale=4)
                send_gen = gr.Button("Send", variant="primary", scale=1)

        with gr.TabItem("📄 Prescription Scanner"):
            with gr.Row():
                with gr.Column():
                    prescription_file = gr.File(label="Prescription Document")
                    analyze_btn = gr.Button("Scan & Analyze", variant="primary")
                with gr.Column():
                    prescription_output = gr.Textbox(label="Extracted Text Output", lines=6)
                    medicine_results = gr.Markdown("Matches show here.")
                    rag_context_state = gr.State("")
            prescription_chatbot = gr.Chatbot(label="Prescription Chat Assistant", height=300)
            with gr.Accordion("🎤 Voice Input", open=False):
                mic_rx = gr.Audio(sources=["microphone"], type="filepath")
                transcribe_rx_btn = gr.Button("Convert Voice")
            with gr.Row():
                prescription_msg = gr.Textbox(placeholder="Ask about dosage...", lines=2, scale=4)
                prescription_send = gr.Button("Send", variant="primary")

        with gr.TabItem("🩻 X-Ray Analysis Simulator"):
            xray_file = gr.File(label="X-Ray Scan")
            xray_chatbot = gr.Chatbot(label="X-Ray Simulator Chat", height=350)
            with gr.Accordion("🎤 Voice Input", open=False):
                mic_xray = gr.Audio(sources=["microphone"], type="filepath")
                transcribe_xray_btn = gr.Button("Convert Voice")
            with gr.Row():
                xray_msg = gr.Textbox(placeholder="Ask about alignment...", lines=2, scale=4)
                xray_send = gr.Button("Simulate", variant="primary")

        with gr.TabItem("🔬 Medical Report Analyzer"):
            report_file = gr.File(label="Lab Report")
            report_chatbot = gr.Chatbot(label="Medical Report Chat", height=350)
            with gr.Accordion("🎤 Voice Input", open=False):
                mic_report = gr.Audio(sources=["microphone"], type="filepath")
                transcribe_report_btn = gr.Button("Convert Voice")
            with gr.Row():
                report_msg = gr.Textbox(placeholder="Ask to examine ranges...", lines=2, scale=4)
                report_send = gr.Button("Analyze", variant="primary")

        with gr.TabItem("🤒 Symptom Checker"):
            symptom_chatbot = gr.Chatbot(label="Symptom Triage Chat", height=350)
            with gr.Accordion("🎤 Voice Input", open=False):
                mic_symptom = gr.Audio(sources=["microphone"], type="filepath")
                transcribe_symptom_btn = gr.Button("Convert Voice")
            with gr.Row():
                symptom_msg = gr.Textbox(placeholder="Describe symptoms...", lines=2, scale=4)
                symptom_send = gr.Button("Run Triage", variant="primary")

    # Voice transcription handlers
    transcribe_gen_btn.click(fn=transcribe_mic_audio, inputs=[mic_gen], outputs=[msg_gen])
    transcribe_rx_btn.click(fn=transcribe_mic_audio, inputs=[mic_rx], outputs=[prescription_msg])
    transcribe_xray_btn.click(fn=transcribe_mic_audio, inputs=[mic_xray], outputs=[xray_msg])
    transcribe_report_btn.click(fn=transcribe_mic_audio, inputs=[mic_report], outputs=[report_msg])
    transcribe_symptom_btn.click(fn=transcribe_mic_audio, inputs=[mic_symptom], outputs=[symptom_msg])

    # Event bindings
    send_gen.click(fn=chat_general, inputs=[msg_gen, chatbot_gen, lang_select], outputs=[chatbot_gen, msg_gen])
    analyze_btn.click(fn=analyze_prescription_pipeline, inputs=[prescription_file], outputs=[prescription_output, medicine_results, rag_context_state])
    prescription_send.click(fn=chat_prescription, inputs=[prescription_msg, prescription_chatbot, prescription_output, rag_context_state, lang_select], outputs=[prescription_chatbot, prescription_msg])
    xray_send.click(fn=chat_xray, inputs=[xray_msg, xray_chatbot, xray_file, lang_select], outputs=[xray_chatbot, xray_msg])
    report_send.click(fn=chat_report, inputs=[report_msg, report_chatbot, report_file, lang_select], outputs=[report_chatbot, report_msg])
    symptom_send.click(fn=chat_symptoms, inputs=[symptom_msg, symptom_chatbot, lang_select], outputs=[symptom_chatbot, symptom_msg])

if __name__ == '__main__':
    # Bind to host 0.0.0.0 and port 10000 for deployment environment compatibility
    demo.launch(server_name='0.0.0.0', server_port=10000, share=False)
