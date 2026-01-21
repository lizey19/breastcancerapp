import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import sqlite3
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
import os

st.image("image.jpg", use_container_width=True)
st.markdown("<h2 style='text-align: center; color: pink;'>🎀 Early Breast Cancer Detection System</h2>", unsafe_allow_html=True)


def T(en, ur, lang="English"):
    return ur if lang == "اردو" else en

# Language selection — now safe
# --------------------------------------------------
lang = st.selectbox("🌐 Select Language | زبان منتخب کریں", ["English", "اردو"])
# --------------------------------------------------
# PAGE STYLE
# --------------------------------------------------
st.set_page_config(page_title="Her Health Solution", page_icon="🎀", layout="wide")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg,#ffe6f2 0%,#fff0f6 50%,#ffffff 100%);
    font-family: "Segoe UI", sans-serif;
}
[data-testid="stSidebar"]{background:#ffd6eb;}
h1,h2,h3,h4{color:#b30059!important;}
.stTabs [data-baseweb="tab-list"]{background-color:#ffebf2;border-radius:10px;}
.stTabs [data-baseweb="tab"]{color:#b30059;}
.stTabs [aria-selected="true"]{background-color:#ffb6d1!important;color:white!important;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
APP_NAME = "Her Health Solution"
st.markdown(f"""
<div style='text-align:center; margin-top:18px; margin-bottom:12px;'>
  <h1 style='font-size:3rem;color:#b30059;font-weight:800;'>{APP_NAME}</h1>
  <p style='font-size:1.1rem;color:#cc0066;'>{T("Breast Cancer Detection & Awareness System","بریسٹ کینسر کی تشخیص اور آگاہی کا نظام", lang)}</p>
</div>
""", unsafe_allow_html=True)
# --------------------------------------------------
# PRIVACY INFO
# --------------------------------------------------
st.info(T(
    "🔒 Your information stays private and is stored only on this device. No data is shared online.",
    "🔒 آپ کی معلومات صرف اسی ڈیوائس پر محفوظ رہتی ہیں۔ کوئی ڈیٹا آن لائن شیئر نہیں کیا جاتا۔",
    lang
))

# DATABASE — simplified path
# --------------------------------------------------
DB_PATH = "breast_cancer_records.db"
EXCEL_PATH = "patient_records.xlsx"

def init_database():
    conn=sqlite3.connect(DB_PATH)
    c=conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patient_records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT, age INTEGER, bmi REAL, gender TEXT, contact TEXT,
        medical_history TEXT, doctor_name TEXT, date TEXT, diagnosis TEXT,
        confidence REAL, model_used TEXT,
        radius_worst REAL, texture_mean REAL, smoothness_mean REAL, 
        concave_points_mean REAL, area_worst REAL, timestamp TEXT)''')
    conn.commit()
    conn.close()

def clear_all_data():
    if os.path.exists(DB_PATH): os.remove(DB_PATH)
    if os.path.exists(EXCEL_PATH): os.remove(EXCEL_PATH)
    init_database()

def save_to_database(info, feats, result):
    conn=sqlite3.connect(DB_PATH)
    c=conn.cursor()
    diag="Benign" if result['prediction']==0 else "Malignant"
    conf=float(result['confidence'])
    c.execute('''INSERT INTO patient_records(
        patient_name,age,bmi,gender,contact,medical_history,doctor_name,date,
        diagnosis,confidence,model_used,radius_worst,texture_mean,
        smoothness_mean,concave_points_mean,area_worst,timestamp)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (info['name'],info['age'],info['bmi'],info['gender'],info['contact'],
         info['medical_history'],info['doctor'],info['date'],diag,conf,
         result['model_used'],*feats,datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# Initialize database
init_database()

# --------------------------------------------------
# PDF REPORT
# --------------------------------------------------
def generate_pdf_report(info, feats, result):
    buf=BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4)
    els=[]
    styles=getSampleStyleSheet()
    els.append(Paragraph("🎗️ Breast Cancer Report",styles['Title']))
    data=[[k,str(v)] for k,v in info.items()]
    t=Table(data)
    t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    els+=[Spacer(1,12),t,Spacer(1,12)]
    diag="BENIGN" if result['prediction']==0 else "MALIGNANT"
    color=colors.green if diag=="BENIGN" else colors.red
    t2=Table([["Diagnosis",diag],["Confidence",f"{result['confidence']:.2f}%"]])
    t2.setStyle(TableStyle([('TEXTCOLOR',(1,0),(1,0),color),('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    els.append(t2)
    doc.build(els)
    buf.seek(0)
    return buf

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
@st.cache_resource
def train_models(data):
    df=pd.read_csv(data)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    rename_map = {}
    for col in df.columns:
        if "concave" in col and "mean" in col:
            rename_map[col] = "concave_points_mean"
    df = df.rename(columns=rename_map)
    expected = ['diagnosis','radius_worst','texture_mean','smoothness_mean','concave_points_mean','area_worst']
    existing = [c for c in expected if c in df.columns]
    if 'diagnosis' not in existing:
        st.error("❌ Dataset must include a 'diagnosis' column (with M/B labels).")
        st.stop()
    df = df[existing]
    X=df.drop('diagnosis',axis=1)
    y=(df['diagnosis'].str.upper().isin(['M','MALIGNANT'])).astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    sc=StandardScaler()
    XtrS=sc.fit_transform(Xtr)
    XteS=sc.transform(Xte)
    rf=RandomForestClassifier(n_estimators=150,max_depth=10,random_state=42)
    gb=GradientBoostingClassifier(n_estimators=150,random_state=42)
    svm=SVC(probability=True,random_state=42)
    lr=LogisticRegression(max_iter=1000,random_state=42)
    ens=VotingClassifier(estimators=[('rf',rf),('gb',gb),('svm',svm),('lr',lr)],voting='soft')
    models={'Random Forest':rf,'Gradient Boosting':gb,'SVM':svm,'Logistic Regression':lr,'Ensemble':ens}
    for m in models.values(): m.fit(XtrS,ytr)
    acc={n:m.score(XteS,yte) for n,m in models.items()}
    return models,sc,acc,X.columns.tolist()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

mode = st.sidebar.radio("Choose Mode", ["👩‍🦰 Home User", "🧠 Researcher/Doctor"])

if mode == "👩‍🦰 Home User":
    st.header("🏠 Self-Check Mode (For Home Users)")
    st.markdown("💗 *Take a few minutes to answer honestly — this quick check helps you understand your breast health.*")

    tabs_home = st.tabs(["💖 Self-Check", "🌸 Awareness & Education"])

    # --- Tab 1: Self-Check ---
    with tabs_home[0]:
        age = st.number_input("🎂 How old are you?", min_value=10, max_value=100, step=1)
        height = st.number_input("📏 Your height (in cm)", min_value=100, max_value=200, step=1)
        weight = st.number_input("⚖️ Your weight (in kg)", min_value=30, max_value=200, step=1)
        bmi = weight / ((height / 100) ** 2)
        st.info(f"💡 **Your BMI:** {bmi:.2f}")

        st.subheader("🌸 Your Breast Health Observations")
        lump = st.radio("🤲 Have you recently felt a small lump or thick area?", ["No", "Yes"])
        pain = st.radio("💢 Do you feel pain or tenderness?", ["No", "Yes"])
        discharge = st.radio("💧 Any discharge or nipple changes?", ["No", "Yes"])

        st.subheader("✨ More Signs to Watch For")
        q4 = st.radio("⚪ Nipple inverted or changed direction?", ["No", "Yes"])
        q5 = st.radio("🌺 Redness or puckering on skin?", ["No", "Yes"])
        q6 = st.radio("🔥 Burning or tenderness unrelated to cycle?", ["No", "Yes"])
        q7 = st.radio("🧍‍♀️ Swelling or lump near collarbone?", ["No", "Yes"])

        st.subheader("👩‍👧 Family & Personal History")
        q8 = st.radio("🧬 Any family history of cancer?", ["No", "Yes"])
        q9 = st.radio("🩹 Previous breast lump or surgery?", ["No", "Yes"])
        q10 = st.radio("👶 First child after 30 or no childbirth?", ["No", "Yes"])
        q11 = st.radio("🕐 Are you above 40 years old?", ["No", "Yes"])
        q12 = st.radio("💊 Using hormonal therapy or birth control pills?", ["No", "Yes"])

        if st.button("💖 Check My Risk"):
            score = sum(ans == "Yes" for ans in [lump, pain, discharge, q4, q5, q6, q7, q8, q9, q10, q11, q12])

            st.markdown("---")
            if score <= 2:
                st.success("🟢 Low Risk: Everything seems fine.")
            elif 3 <= score <= 5:
                st.warning("🟠 Moderate Risk: Some symptoms need attention.")
            else:
                st.error("🔴 High Risk: Please consult a specialist.")
            st.markdown("_This is an awareness tool, not a medical diagnosis._")
    
    # --- Tab 2: Awareness ---
    with tabs_home[1]:
        st.subheader("🌸 Learn & Stay Aware")
        st.markdown("💗 *Taking care of your health means taking care of your power.*")

        st.markdown("#### 🪞 Step-by-Step Self-Check")
        st.markdown("""
        1️⃣ Stand before a mirror — look for swelling or dimples.  
        2️⃣ Raise your arms — check both sides for symmetry.  
        3️⃣ Press each breast gently in small circles — feel for lumps.  
        4️⃣ Squeeze your nipple — check for discharge or tenderness.  
        5️⃣ Lie down and repeat — tissue spreads evenly.  
        💡 *Do this monthly, 3–5 days after your period ends.*""")

        st.markdown("#### 💬 Common Myths and Facts")
        st.markdown("""
        | ❌ Myth | ✅ Fact |
        |---------|---------|
        | Only older women get breast cancer. | It can happen at any age. |
        | Pain means cancer. | Most pain isn’t cancer-related. |
        | Mammograms are painful. | They’re quick and mildly uncomfortable. |
        | Men can’t get breast cancer. | They can, though it’s rare. |""")

        st.info("💕 *If you notice anything unusual, don’t panic — 8 out of 10 lumps are non-cancerous.*")

elif mode == "🧠 Researcher/Doctor":
    st.header("🧠 Researcher/Doctor Mode")
    st.write("Upload dataset and analyze..")


st.sidebar.header(T("📂 Upload Dataset","📂 ڈیٹا سیٹ اپ لوڈ کریں ", lang))
f=st.sidebar.file_uploader(T("Upload CSV file","سی ایس وی فائل اپ لوڈ کریں"),type=["csv"])
if not f:
    st.sidebar.warning(T("Please upload dataset.","ڈیٹا اپ لوڈ کریں۔"))
    st.stop()
models,scaler,accs,features=train_models(f)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tabs=st.tabs([
    T("📊 Clinical Measurements","📊 کلینیکل پیمائشیں", lang),
    T("🩺 Symptom Checker","🩺 علامات چیکر", lang),
    T("📈 Model Performance","📈 ماڈل کارکردگی", lang),
    T("📁 Records","📁 ریکارڈز", lang),
    T("⚙️ Privacy","⚙️ رازداری", lang)
])

# --------------------------------------------------
# CLINICAL TAB — with session-state saving and PDF
# --------------------------------------------------
with tabs[0]:
    st.subheader(T("📊 Enter Clinical Data & Predict","📊 کلینیکل ڈیٹا درج کریں اور پیشن گوئی کریں"))
    c1,c2=st.columns(2)
    with c1:
        name=st.text_input(T("Patient Name","مریض کا نام"))
        age=st.number_input(T("Age","عمر"),1,120,45)
        gender=st.selectbox(T("Gender","صنف"),["Female","Male","Other"])
        bmi=st.number_input("BMI",10.0,50.0,25.0)
        contact=st.text_input(T("Contact Number","رابطہ نمبر"))
    with c2:
        doctor=st.text_input(T("Doctor Name","ڈاکٹر کا نام"))
        date=st.date_input(T("Examination Date","معائنہ کی تاریخ"),value=datetime.now())
        medical_history=st.text_area(T("Medical History","طبی تاریخ"),height=100)

    st.markdown("### "+T("Top 5 Clinical Measurements","اہم 5 کلینیکل پیمائشیں"))
    cols=st.columns(3)
    vals=[]
    for i,fname in enumerate(features):
        with cols[i%3]:
            vals.append(st.number_input(fname,value=0.0,step=0.01))

    model_choice=st.selectbox(T("Select Model","ماڈل منتخب کریں"),list(models.keys()))

    # ⬇️ THIS MUST STAY INSIDE THE TAB!
    if st.button(T("🔍 Predict", "🔍 پیشن گوئی کریں")):
        feats = vals
        Xin = np.array([feats])
        Xs = scaler.transform(Xin)
        m = models[model_choice]
        p = m.predict(Xs)[0]
        prob = m.predict_proba(Xs)[0][1]
        conf = prob * 100 if p == 1 else (1 - prob) * 100

        result = {'prediction': p, 'confidence': conf, 'model_used': model_choice}
        info = {
            'name': name, 'age': age, 'gender': gender, 'bmi': bmi,
            'contact': contact, 'medical_history': medical_history,
            'doctor': doctor, 'date': str(date)
        }

        st.session_state['last_prediction'] = {'info': info, 'feats': feats, 'result': result}

        if p == 0:
            st.success(f"✅ {T('BENIGN — Non-cancerous', 'غیر سرطانی')} ({conf:.2f}%)")
            st.info("🌸 Great news! No cancer detected. Still, perform self-checks monthly and stay healthy!")
        else:
            st.error(f"⚠️ {T('MALIGNANT — Possible Cancerous', 'ممکنہ طور پر سرطانی')} ({conf:.2f}%)")
            st.warning("⚠️ Possible cancer signs detected. Please visit a doctor for further screening immediately.")

        st.markdown("### 📊 Prediction Confidence")
        fig = px.bar(
            x=["Benign", "Malignant"],
            y=[100 - conf, conf] if p == 1 else [conf, 100 - conf],
            labels={'x': 'Diagnosis', 'y': 'Confidence (%)'},
            color=["Benign", "Malignant"],
            color_discrete_sequence=["green", "red"]
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- Feature Importance (Explainability) ----
        if hasattr(models[model_choice], "feature_importances_"):
            importances = models[model_choice].feature_importances_
            imp_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            st.markdown("### 🧠 Feature Importance (Explainable AI)")
            st.bar_chart(imp_df.set_index("Feature"))
        else:
            st.info("Explainable AI not available for this model (e.g., SVM or Logistic Regression).")

        # ---- SAVE & PDF ----
        if 'last_prediction' in st.session_state:
            if st.button("💾 Save Record"):
                try:
                    lp = st.session_state['last_prediction']
                    save_to_database(lp['info'], lp['feats'], lp['result'])
                    st.success("✅ Record saved successfully!")
                except Exception as e:
                    st.error(f"❌ Error saving record: {e}")

            lp = st.session_state['last_prediction']
            pdf = generate_pdf_report(lp['info'], lp['feats'], lp['result'])
            st.download_button(
                T("📄 Download PDF Report", "📄 رپورٹ ڈاؤن لوڈ کریں"),
                pdf,
                file_name=f"{lp['info']['name']}_Report.pdf",
                mime="application/pdf"
            )

# SYMPTOM CHECKER
# --------------------------------------------------
with tabs[1]:
    st.subheader(T("🩺 Symptom Self-Assessment","🩺 علامات کا خود جائزہ"))

    # Symptoms
    l = st.radio(T("Any lumps or pain?","کیا کسی گانٹھ یا درد کا احساس ہے؟"), ["No","Yes"])
    if l == "Yes":
        duration = st.selectbox(
            T("How long have you felt this symptom?","یہ علامات کب سے ہیں؟"),
            ["Less than a week", "1–2 weeks", "More than 2 weeks"]
        )
    else:
        duration = "None"

    d = st.radio(T("Any unusual discharge?","کوئی غیر معمولی اخراج؟"),["No","Yes"])
    h = st.radio(T("Family history of cancer?","کیا خاندان میں تاریخ ہے؟"),["No","Yes"])

    # Check result button
    if st.button(T("Check Result","نتیجہ دیکھیں")):
        risk = sum([l=="Yes", d=="Yes", h=="Yes"])
        if duration == "1–2 weeks": 
            risk += 1
        elif duration == "More than 2 weeks": 
            risk += 2

        # Show risk message
        if risk <= 1:
            st.success(T("Low risk — stay aware and check monthly.","کم خطرہ — ہر ماہ خود معائنہ کریں۔"))
        elif risk == 2:
            st.warning(T("Mild concern — monitor and consult if needed.","معمولی تشویش — تبدیلی کی صورت میں ڈاکٹر سے رجوع کریں۔"))
        else:
            st.error(T("High concern — visit a doctor soon.","زیادہ خطرہ — جلد ڈاکٹر سے رجوع کریں۔"))


# --------------------------------------------------
# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
with tabs[2]:
    st.subheader(T("📈 Model Accuracy","📈 ماڈل درستگی"))
    df=pd.DataFrame({'Model':list(accs.keys()),'Accuracy (%)':[v*100 for v in accs.values()]})
    st.bar_chart(df.set_index("Model"))

# --------------------------------------------------
# RECORDS TAB
# --------------------------------------------------
with tabs[3]:
    st.subheader(T("📁 Patient Records","📁 مریضوں کے ریکارڈ"))
    conn=sqlite3.connect(DB_PATH)
    dfrec=pd.read_sql_query("SELECT * FROM patient_records ORDER BY timestamp DESC",conn)
    conn.close()
    if not dfrec.empty:
        st.dataframe(dfrec)
        d=dfrec['diagnosis'].value_counts()
        st.plotly_chart(px.pie(values=d.values,names=d.index,
            title=T("Diagnosis Distribution","تشخیص کی تقسیم")),use_container_width=True)
    else:
        st.info(T("No records found.","کوئی ریکارڈ نہیں ملا۔"))

# --------------------------------------------------
# PRIVACY TAB
# --------------------------------------------------
with tabs[4]:
    st.subheader(T("⚙️ Privacy Settings","⚙️ رازداری"))
    st.write(T("Delete all saved data for privacy.","رازداری کے لیے تمام ڈیٹا حذف کریں۔"))
    if st.button(T("🧹 Clear All Data","🧹 تمام ڈیٹا حذف کریں")):
        clear_all_data()
        st.success(T("All data deleted.","تمام ڈیٹا حذف کر دیا گیا۔"))

