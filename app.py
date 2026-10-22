"""BurnoutIQ 2.0 — exploratory student wellbeing intelligence (synthetic data)."""
from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="BurnoutIQ | Intelligence Workspace", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "student_burnout_dropout_dataset_2.csv"
MODEL = ROOT / "burnout_prediction_model.pkl"
SEGMENTS = ROOT / "student_segments.csv"
PALETTE = {"Low": "#2dd4bf", "Medium": "#fbbf24", "High": "#fb7185"}

st.markdown("""
<style>
:root {color-scheme:dark;}
.stApp {background:radial-gradient(ellipse at 90% 0%,#1b2350 0%,#0c1125 40%,#080d1c 100%); color:#eaf0ff;}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none !important;}
.block-container {max-width:1450px; padding-top:1.1rem; padding-bottom:3rem;}
h1,h2,h3,h4,p,span,label {color:inherit;}
h1 {font-weight:800;letter-spacing:-.04em;}
h2 {font-weight:750;letter-spacing:-.03em;}
[data-testid="stMetric"], div[data-testid="stForm"], div[data-testid="stExpander"] {
 background:rgba(20,29,58,.85);border:1px solid #293557;border-radius:18px;padding:14px;
}
[data-testid="stMetricLabel"] {color:#9fb1d5;}
[data-testid="stMetricValue"] {color:#f4f7ff;}
.stButton button[kind="primary"], div[data-testid="stFormSubmitButton"] button {
 background:linear-gradient(110deg,#7357f5,#3a8af7);border:0;color:white;border-radius:12px;
}
.stButton button, [data-testid="stDownloadButton"] button {border-radius:12px;}
div[role="radiogroup"] {background:#141d3a;padding:8px;border:1px solid #293557;border-radius:16px;gap:5px;}
div[role="radiogroup"] label {padding:6px 12px;border-radius:10px;}
[data-testid="stAlert"] {border-radius:14px;}
[data-testid="stDataFrame"] {border:1px solid #293557;border-radius:14px;overflow:hidden;}
.hero {padding:28px 30px;border-radius:22px;background:linear-gradient(115deg,#252060,#172d55 60%,#123f55);border:1px solid #3a4279;margin:10px 0 23px;}
.hero h1 {font-size:2.45rem;margin:5px 0 8px;color:#fff;}
.hero p {color:#c7d4ef;margin:0;max-width:750px;}
.eyebrow {font-size:.78rem;letter-spacing:.16em;color:#9eb8ff;font-weight:800;}
.tile {padding:18px 20px;border:1px solid #303d64;border-radius:17px;background:#151f3c;min-height:118px;}
.tile b {font-size:1.7rem;color:#f8faff;display:block;margin:7px 0;}
.tile small {color:#a8b9d7;}
.note {background:#111d35;border-left:3px solid #8c83ff;border-radius:9px;padding:13px 17px;color:#b9c9e6;margin:16px 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

@st.cache_resource
def load_model():
    return joblib.load(MODEL)

try:
    df = load_data()
    model = load_model()
except Exception as exc:
    st.error(f"Could not load project data/model: {exc}")
    st.info("Keep app.py, student_burnout_dropout_dataset_2.csv, and burnout_prediction_model.pkl together.")
    st.stop()

EXCLUDE = {"Student_ID", "Burnout_Level", "Dropout_Risk", "Student_Segment"}
features = list(getattr(model, "feature_names_in_", [])) or [c for c in df.columns if c not in EXCLUDE]
missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"Model input columns missing from dataset: {missing}")
    st.stop()

PLOT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#d9e4ff", margin=dict(l=15,r=15,t=55,b=15))
def chart(fig, key=None):
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True, key=key)

def hero(tag, title, desc):
    st.markdown(f'<div class="hero"><div class="eyebrow">{html.escape(tag)}</div><h1>{html.escape(title)}</h1><p>{html.escape(desc)}</p></div>', unsafe_allow_html=True)

def tile(label, value, note):
    st.markdown(f'<div class="tile"><small>{html.escape(str(label))}</small><b>{html.escape(str(value))}</b><small>{html.escape(str(note))}</small></div>', unsafe_allow_html=True)

def baseline():
    row = {}
    for col in features:
        s = df[col].dropna()
        if pd.api.types.is_numeric_dtype(df[col]):
            row[col] = float(s.median()) if not s.empty else 0.0
        else:
            row[col] = str(s.mode().iloc[0]) if not s.empty else "Unknown"
    return row

def field_input(col, default, key):
    s = df[col].dropna()
    label = col.replace("_", " ")
    if pd.api.types.is_numeric_dtype(df[col]):
        lo, hi = float(s.min()), float(s.max())
        if pd.api.types.is_integer_dtype(df[col]) and np.isclose(s.dropna() % 1, 0).all():
            return st.number_input(label, min_value=int(lo), max_value=int(hi), value=int(np.clip(round(float(default)),lo,hi)), step=1, key=key)
        return st.number_input(label, min_value=lo, max_value=hi, value=float(np.clip(float(default),lo,hi)), step=0.1, key=key)
    opts = sorted(s.astype(str).unique().tolist())
    return st.selectbox(label, opts, index=opts.index(str(default)) if str(default) in opts else 0, key=key)

def predict(row):
    data = pd.DataFrame([row])[features]
    p = model.predict_proba(data)[0]
    return str(model.predict(data)[0]), pd.Series(p, index=model.classes_)

def probability_chart(probs, title):
    p = pd.DataFrame({"Category":probs.index, "Probability":probs.values * 100})
    fig = px.bar(p, x="Probability", y="Category", orientation="h", color="Category", color_discrete_map=PALETTE, text="Probability", title=title)
    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    fig.update_xaxes(range=[0,108], title="Model-estimated probability (%)")
    fig.update_yaxes(title=None)
    chart(fig)

st.markdown('<div class="eyebrow">◈ BURNOUTIQ / INTELLIGENCE WORKSPACE</div>', unsafe_allow_html=True)
nav = ["◈ Command Center", "◎ Student Explorer", "◇ Risk Intelligence", "⚗ What-If Lab", "▦ Research Studio"]
page = st.radio("Workspace", nav, horizontal=True, label_visibility="collapsed")

if page == nav[0]:
    hero("01 / OVERVIEW", "The wellbeing command center", "Explore academic and lifestyle patterns across a synthetic student population. This is exploratory analytics, not a screening tool.")
    cols = st.columns(4)
    vals = [("Students",f"{len(df):,}","Synthetic observations"), ("High burnout",f"{(df.Burnout_Level=='High').sum():,}","Dataset labels"), ("Dropout risk",f"{(df.Dropout_Risk=='Yes').sum():,}","Dataset labels"), ("Average stress",f"{df.Stress_Level.mean():.2f}","Dataset score")]
    for c,(a,b,d) in zip(cols,vals):
        with c: tile(a,b,d)
    st.write("")
    c1,c2 = st.columns([1.35,1])
    with c1:
        fig = px.scatter(df, x="Sleep_Hours", y="Motivation_Score", color="Burnout_Level", color_discrete_map=PALETTE, opacity=.65, hover_data=["Stress_Level","Anxiety_Score"], title="Sleep and motivation • burnout labels")
        chart(fig)
    with c2:
        group = df.groupby(["Department","Burnout_Level"]).size().reset_index(name="Students")
        fig = px.bar(group,x="Department",y="Students",color="Burnout_Level",color_discrete_map=PALETTE,barmode="stack",title="Burnout composition by department")
        chart(fig)
    st.markdown('<div class="note">Dataset labels and relationships are synthetic. Associations do not demonstrate causation or clinical risk.</div>', unsafe_allow_html=True)

elif page == nav[1]:
    hero("02 / PROFILE EXPLORATION", "Student explorer", "Select a synthetic student record and inspect their academic, lifestyle, and model-estimated burnout profile.")
    left,right = st.columns([1,2])
    with left:
        index = st.selectbox("Choose dataset record", range(len(df)), format_func=lambda i: f"Record {i+1} · {df.iloc[i]['Student_ID']}")
        row = df.iloc[index]
        st.write("**Recorded burnout label:**", row["Burnout_Level"])
        st.write("**Recorded dropout risk:**", row["Dropout_Risk"])
        st.write("**Department:**", row.get("Department", "Not available"))
        st.write("**Year of study:**", row.get("Year_of_Study", "Not available"))
    with right:
        display_cols = [c for c in ["Sleep_Hours","Study_Hours_Per_Day","Stress_Level","Anxiety_Score","Motivation_Score","Family_Support_Score"] if c in df]
        values = pd.DataFrame({"Characteristic":display_cols,"Student value":[row[c] for c in display_cols],"Dataset median":[df[c].median() for c in display_cols]})
        st.dataframe(values, hide_index=True, use_container_width=True)
        if not pd.isna(row[features]).any():
            label, probs = predict(row[features].to_dict())
            st.markdown(f"**Model prediction:** {label} (not a clinical assessment)")
            probability_chart(probs,"Predicted category distribution")
        else:
            st.info("This record has missing input values. Use Risk Intelligence to create a complete profile.")

elif page == nav[2]:
    hero("03 / CLASSIFICATION", "Risk intelligence", "Build a hypothetical student profile and examine a classification model's estimated burnout category.")
    st.warning("Educational model trained on synthetic records. Do not use for diagnosis, student decisions, or real-world screening.")
    with st.form("risk_profile"):
        st.markdown("### Build a student profile")
        c1,c2,c3 = st.columns(3)
        defaults = baseline()
        user = {}
        for i,col in enumerate(features):
            with [c1,c2,c3][i%3]:
                user[col] = field_input(col,defaults[col],f"risk_{col}")
        submitted = st.form_submit_button("Analyze profile", use_container_width=True)
    if submitted:
        try:
            label, probs = predict(user)
            st.subheader(f"Estimated category: {label}")
            probability_chart(probs,"Model-estimated category probabilities")
            st.caption("Model probabilities are uncalibrated unless independently validated; they are not medical risk probabilities.")
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")

elif page == nav[3]:
    hero("04 / EXPERIMENTATION", "What-If Lab", "Compare model outputs before and after hypothetical changes to a student profile. This does not estimate causal effects.")
    defaults = baseline()
    editable = [c for c in ["Sleep_Hours","Study_Hours_Per_Day","Screen_Time_Hours","Stress_Level","Anxiety_Score","Motivation_Score","Exercise_Freq_Per_Week"] if c in features]
    if not editable:
        st.info("No scenario features found in the current model.")
    else:
        c1,c2 = st.columns(2)
        with c1:
            st.subheader("Baseline profile")
            base = defaults.copy()
            for col in editable:
                base[col] = field_input(col, defaults[col], f"before_{col}")
        with c2:
            st.subheader("Alternative scenario")
            alt = defaults.copy()
            for col in editable:
                alt[col] = field_input(col, base[col], f"after_{col}")
        if st.button("Compare scenarios",type="primary",use_container_width=True):
            try:
                b_label,b_probs = predict(base)
                a_label,a_probs = predict(alt)
                st.write(f"**Baseline:** {b_label}  →  **Alternative:** {a_label}")
                comparison = pd.DataFrame({"Category":list(model.classes_)*2,"Estimated probability":list(b_probs.values*100)+list(a_probs.values*100),"Scenario":["Baseline"]*len(b_probs)+["Alternative"]*len(a_probs)})
                fig = px.bar(comparison,x="Category",y="Estimated probability",color="Scenario",barmode="group",title="Model response to hypothetical inputs")
                fig.update_yaxes(range=[0,100],title="Model-estimated probability (%)")
                chart(fig)
                st.info("These differences are model sensitivity, not evidence that changing habits will cause burnout to change.")
            except Exception as exc:
                st.error(f"Scenario comparison failed: {exc}")

else:
    hero("05 / VALIDATION", "Research studio", "Investigate dataset relationships, exploratory clustering, and held-out model evaluation.")
    tabs = st.tabs(["Model evaluation","Feature relationships","Exploratory segments","Data export"])
    with tabs[0]:
        X = df[features]
        y = df["Burnout_Level"]
        _,X_test,_,y_test = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
        pred = model.predict(X_test)
        a,b = st.columns(2)
        a.metric("Test accuracy",f"{accuracy_score(y_test,pred)*100:.2f}%")
        b.metric("Macro F1",f"{f1_score(y_test,pred,average='macro')*100:.2f}%")
        st.caption("Evaluation reproduces the original held-out split; not external validation.")
        report = pd.DataFrame(classification_report(y_test,pred,output_dict=True,zero_division=0)).T
        st.dataframe(report,use_container_width=True)
        labels = list(model.classes_)
        cm = confusion_matrix(y_test,pred,labels=labels)
        fig = px.imshow(cm,x=labels,y=labels,text_auto=True,color_continuous_scale="Purples",title="Confusion matrix")
        fig.update_layout(xaxis_title="Predicted",yaxis_title="Actual")
        chart(fig)
    with tabs[1]:
        numeric = [c for c in df.select_dtypes(include="number").columns if c not in ["Student_ID","Student_Segment"]]
        chosen = st.multiselect("Choose numerical characteristics",numeric,default=[c for c in ["Sleep_Hours","Stress_Level","Anxiety_Score","Motivation_Score","Study_Hours_Per_Day"] if c in numeric])
        if len(chosen)>=2:
            corr = df[chosen].corr()
            fig = px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu",zmin=-1,zmax=1,title="Correlation explorer")
            chart(fig)
        else:
            st.info("Choose at least two characteristics.")
    with tabs[2]:
        if SEGMENTS.exists():
            segments = pd.read_csv(SEGMENTS)
            if "Student_Segment" in segments and len(segments)==len(df):
                st.warning("Exploratory K-Means clustering: silhouette score was 0.056. Segments are weakly separated and must not be interpreted as validated student types.")
                xcol = st.selectbox("Horizontal feature",[c for c in ["Sleep_Hours","Study_Hours_Per_Day","Stress_Level","Anxiety_Score"] if c in segments])
                ycol = st.selectbox("Vertical feature",[c for c in ["Motivation_Score","Anxiety_Score","Family_Support_Score","Stress_Level"] if c in segments])
                segments["Student_Segment"] = segments["Student_Segment"].astype(str)
                fig = px.scatter(segments,x=xcol,y=ycol,color="Student_Segment",opacity=.65,title="Exploratory cluster assignment")
                chart(fig)
                st.dataframe(segments.groupby("Student_Segment")[[c for c in ["Sleep_Hours","Stress_Level","Anxiety_Score","Motivation_Score"] if c in segments]].mean().round(2),use_container_width=True)
            else:
                st.info("Segment file doesn't match the dataset; rerun student_segmentation.py.")
        else:
            st.info("Run student_segmentation.py to generate student_segments.csv.")
    with tabs[3]:
        selected = st.multiselect("Burnout labels",["Low","Medium","High"],default=["Low","Medium","High"])
        filtered = df[df["Burnout_Level"].isin(selected)]
        st.dataframe(filtered.head(100),hide_index=True,use_container_width=True)
        st.download_button("Export filtered synthetic records",filtered.to_csv(index=False).encode("utf-8"),"burnoutiq_filtered.csv","text/csv")

st.markdown('<div class="note">BURNOUTIQ 2.0 · Research demonstration only · Synthetic dataset · Not a clinical or institutional decision tool</div>', unsafe_allow_html=True)
