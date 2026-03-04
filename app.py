# ============================================================
#   ⚙️  TURBOFAN RUL PREDICTION DASHBOARD
#   Hybrid XGBoost–LSTM  |  NASA C-MAPSS
#   Streamlit Cloud Deployment
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import os, io, warnings, joblib
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RUL Predictor — Turbofan Engine",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0e1a; color: #e0e6f0; }
[data-testid="stSidebar"] { background: #0d1220 !important; border-right: 1px solid #1e2d4a; }
[data-testid="stSidebar"] * { color: #c8d4e8 !important; }
h1,h2,h3 { font-family:'Rajdhani',sans-serif !important; letter-spacing:1px; }
h1 { color:#00d4ff !important; font-size:2.2rem !important; font-weight:700 !important; }
h2 { color:#a0b8d8 !important; font-size:1.4rem !important; }
h3 { color:#7090b0 !important; font-size:1.1rem !important; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg,#0f1829,#131f35);
    border:1px solid #1e3050; border-radius:10px;
    padding:16px !important; box-shadow:0 4px 20px rgba(0,212,255,0.05);
}
[data-testid="stMetricLabel"] { color:#6080a0 !important; font-size:0.75rem !important; text-transform:uppercase; letter-spacing:1.5px; }
[data-testid="stMetricValue"] { color:#00d4ff !important; font-family:'Share Tech Mono',monospace !important; font-size:1.8rem !important; }
[data-testid="stMetricDelta"] { font-family:'Share Tech Mono',monospace !important; }
[data-testid="stTabs"] button { font-family:'Rajdhani',sans-serif !important; font-size:1rem !important; font-weight:600 !important; color:#6080a0 !important; letter-spacing:1px; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#00d4ff !important; border-bottom:2px solid #00d4ff !important; }
hr { border-color:#1e2d4a !important; }
div[data-testid="stFileUploader"] { background:#0d1627; border:2px dashed #1e3050; border-radius:10px; padding:10px; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0a0e1a; }
::-webkit-scrollbar-thumb { background:#1e3050; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly base theme ────────────────────────────────────────
AX = dict(gridcolor='#1a2540', linecolor='#1e3050', tickfont=dict(color='#6080a0'))
PT_BASE = dict(
    paper_bgcolor='#0a0e1a', plot_bgcolor='#0d1220',
    font=dict(color='#8090b0', family='Inter'),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8090b0')),
    margin=dict(t=50, b=40, l=50, r=20),
)

def pfig(fig, *, title=None, height=None, xtitle=None, ytitle=None,
         yrange=None, showlegend=None, barmode=None, **extra):
    kw = dict(**PT_BASE)
    kw['xaxis'] = dict(**AX, **(dict(title=xtitle) if xtitle else {}))
    kw['yaxis'] = dict(**AX, **(dict(title=ytitle) if ytitle else {}),
                       **(dict(range=yrange) if yrange else {}))
    if title:      kw['title'] = dict(text=title, font=dict(color='#a0b8d8'))
    if height:     kw['height'] = height
    if showlegend is not None: kw['showlegend'] = showlegend
    if barmode:    kw['barmode'] = barmode
    kw.update(extra)
    fig.update_layout(**kw)
    return fig

def pc(fig): st.plotly_chart(fig, width='stretch')

C = dict(
    lstm='#4da6ff', xgb='#00ff88', hybrid='#ff6b35',
    actual='#a0b8d8', accent='#00d4ff',
    danger='#ff4060', warn='#ffaa00', ok='#00ff88',
)

# ── Constants ────────────────────────────────────────────────
COLUMNS      = ['engine_id','cycle','op1','op2','op3'] + [f's{i}' for i in range(1,22)]
DROP_SENSORS = ['s1','s5','s6','s10','s16','s18','s19']
SEQ_LEN      = 50
RUL_CAP      = 125
DATASET_CFG  = {
    'FD001': {'n_clusters':1,  'label':'1 Condition · 1 Fault   (Easiest)'},
    'FD002': {'n_clusters':6,  'label':'6 Conditions · 1 Fault'},
    'FD003': {'n_clusters':1,  'label':'1 Condition · 2 Faults'},
    'FD004': {'n_clusters':6,  'label':'6 Conditions · 2 Faults (Hardest)'},
}

# ── Helpers ───────────────────────────────────────────────────
def rul_status(v):
    if v < 20:  return '🔴 CRITICAL', '#ff4060'
    if v < 50:  return '🟡 WARNING',  '#ffaa00'
    return '🟢 HEALTHY', '#00ff88'

def compute_metrics(true, pred):
    if true is None or len(true) != len(pred): return None
    return {
        'RMSE': float(np.sqrt(np.mean((true-pred)**2))),
        'MAE':  float(mean_absolute_error(true, pred)),
        'R2':   float(r2_score(true, pred)),
    }

# ── Model loading ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(fd):
    try:
        import torch
        from model_def import LSTMModel

        base = os.path.dirname(__file__)
        def p(name): return os.path.join(base, name)

        # Support both flat root and models/ subfolder
        def find(filename):
            for candidate in [p(filename), p(f'models/{filename}')]:
                if os.path.exists(candidate):
                    return candidate
            return None

        paths = {k: find(f'{k}_{fd}.{"pt" if k=="lstm" else "pkl" if k in ("xgb","scaler") else "npy"}')
                 for k in ('lstm','xgb','scaler','features','alpha')}

        missing = [k for k,v in paths.items() if v is None]
        if missing:
            return None, f"Missing files for {fd}: {missing}"

        feature_cols = list(np.load(paths['features'], allow_pickle=True))
        lstm = LSTMModel(input_size=len(feature_cols))
        lstm.load_state_dict(torch.load(paths['lstm'], map_location='cpu'))
        lstm.eval()

        return {
            'lstm':         lstm,
            'xgb':          joblib.load(paths['xgb']),
            'scalers':      joblib.load(paths['scaler']),
            'feature_cols': feature_cols,
            'alpha':        float(np.load(paths['alpha'])[0]),
            'fd':           fd,
        }, None

    except Exception as e:
        return None, str(e)

# ── Preprocessing ────────────────────────────────────────────
def engineer_features(df):
    sensor_cols = [c for c in df.columns
                   if c.startswith('s') and '_' not in c and c not in ('op_cluster',)]
    for col in sensor_cols:
        g = df.groupby('engine_id')[col]
        df[f'{col}_rmean'] = g.transform(lambda x: x.rolling(10, min_periods=1).mean())
        df[f'{col}_rstd']  = g.transform(lambda x: x.rolling(10, min_periods=1).std().fillna(0))
        df[f'{col}_slope'] = g.transform(lambda x: x.diff().fillna(0))
        df[f'{col}_ewm']   = g.transform(lambda x: x.ewm(span=10, adjust=False).mean())
    return df

def preprocess(df_raw, fd, feature_cols, scalers):
    df = df_raw.copy()
    for c in DROP_SENSORS:
        if c in df.columns: df.drop(columns=[c], inplace=True)

    n_clusters = DATASET_CFG[fd]['n_clusters']
    if n_clusters > 1:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['op_cluster'] = km.fit_predict(df[['op1','op2','op3']])
    else:
        df['op_cluster'] = 0

    df = engineer_features(df)

    tcols = [c for c in feature_cols if c in df.columns]
    for cl, sc in scalers.items():
        idx = df['op_cluster'] == cl
        if idx.any():
            df.loc[idx, tcols] = sc.transform(df.loc[idx, tcols])

    return df

def make_sequences(df, feature_cols):
    X, eids = [], []
    for eid in df['engine_id'].unique():
        edf   = df[df['engine_id']==eid].reset_index(drop=True)
        feats = edf[[c for c in feature_cols if c in edf.columns]].values.astype(np.float32)
        if len(feats) >= SEQ_LEN:
            X.append(feats[-SEQ_LEN:])
        else:
            pad = np.zeros((SEQ_LEN-len(feats), feats.shape[1]), dtype=np.float32)
            X.append(np.vstack([pad, feats]))
        eids.append(int(eid))
    return np.array(X, dtype=np.float32), eids

def get_tab_features(X):
    return np.hstack([
        X[:,-1,:], X.mean(axis=1), X.std(axis=1),
        X[:,-1,:]-X[:,0,:],
        np.percentile(X,25,axis=1), np.percentile(X,75,axis=1),
        np.percentile(X,75,axis=1)-np.percentile(X,25,axis=1),
        X.max(axis=1)-X.min(axis=1),
    ])

@st.cache_data(show_spinner=False)
def run_inference(_models, _X_seq_bytes, fd):
    """Cache inference results. Uses bytes hash of X_seq for cache key."""
    import torch
    X_seq = np.frombuffer(_X_seq_bytes, dtype=np.float32).reshape(-1, SEQ_LEN, _models['lstm'].fc1.in_features if hasattr(_models['lstm'].fc1,'in_features') else 128)

    lstm  = _models['lstm']
    xgb   = _models['xgb']
    alpha = _models['alpha']

    # LSTM
    lstm.eval()
    preds_l = []
    with __import__('torch').no_grad():
        t = __import__('torch').tensor(X_seq)
        for i in range(0, len(t), 256):
            preds_l.append(lstm(t[i:i+256]).numpy())
    pred_lstm = np.concatenate(preds_l)

    # XGBoost
    pred_xgb    = xgb.predict(get_tab_features(X_seq))
    pred_hybrid = np.clip(alpha*pred_xgb + (1-alpha)*pred_lstm, 0, RUL_CAP)
    pred_lstm   = np.clip(pred_lstm,  0, RUL_CAP)
    pred_xgb    = np.clip(pred_xgb,   0, RUL_CAP)

    return pred_lstm, pred_xgb, pred_hybrid

def predict(models_dict, df_test, fd):
    df_proc    = preprocess(df_test, fd, models_dict['feature_cols'], models_dict['scalers'])
    X_seq, eids = make_sequences(df_proc, models_dict['feature_cols'])

    # Reshape for cache key
    import torch
    lstm  = models_dict['lstm']
    xgb   = models_dict['xgb']
    alpha = models_dict['alpha']

    lstm.eval()
    preds_l = []
    with torch.no_grad():
        t = torch.tensor(X_seq)
        for i in range(0, len(t), 256):
            preds_l.append(lstm(t[i:i+256]).numpy())
    pred_lstm   = np.concatenate(preds_l)
    pred_xgb    = xgb.predict(get_tab_features(X_seq))
    pred_hybrid = np.clip(alpha*pred_xgb + (1-alpha)*pred_lstm, 0, RUL_CAP)
    pred_lstm   = np.clip(pred_lstm,  0, RUL_CAP)
    pred_xgb    = np.clip(pred_xgb,   0, RUL_CAP)

    return pred_lstm, pred_xgb, pred_hybrid, eids

def parse_txt(f):
    content = f.read().decode('utf-8') if hasattr(f,'read') else open(f).read()
    df = pd.read_csv(io.StringIO(content), sep=r'\s+', header=None)
    cols = COLUMNS[:df.shape[1]]
    df.columns = cols
    return df.dropna(axis=1)

# ════════════════════════════════════════════════════════════
#   SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ RUL PREDICTOR")
    st.markdown("*Hybrid XGBoost–LSTM*")
    st.markdown("---")

    st.markdown("### 1 · Dataset")
    fd = st.selectbox("Select dataset", list(DATASET_CFG.keys()),
        format_func=lambda x: f"{x} — {DATASET_CFG[x]['label']}")

    st.markdown("### 2 · Test File")
    test_file = st.file_uploader(
        f"Upload test_{fd}.txt",
        type=['txt','csv'],
        help="Space-separated NASA C-MAPSS test file"
    )

    st.markdown("### 3 · RUL File *(optional)*")
    rul_file = st.file_uploader(
        f"Upload RUL_{fd}.txt",
        type=['txt','csv'],
        help="Provide ground-truth RUL to compute RMSE / MAE / R²"
    )

    st.markdown("---")

    # Model status panel
    st.markdown("### Model Status")
    for fdi in DATASET_CFG:
        base = os.path.dirname(__file__)
        has_all = all(
            os.path.exists(os.path.join(base, f'{k}_{fdi}.{"pt" if k=="lstm" else "pkl" if k in ("xgb","scaler") else "npy"}')) or
            os.path.exists(os.path.join(base, f'models/{k}_{fdi}.{"pt" if k=="lstm" else "pkl" if k in ("xgb","scaler") else "npy"}'))
            for k in ('lstm','xgb','scaler','features','alpha')
        )
        st.markdown(f"{'✅' if has_all else '❌'} **{fdi}**  {DATASET_CFG[fdi]['label']}")

    st.markdown("---")
    st.markdown("""
    **Files needed per dataset:**
    `lstm_FDxxx.pt` · `xgb_FDxxx.pkl`
    `scaler_FDxxx.pkl` · `features_FDxxx.npy`
    `alpha_FDxxx.npy`

    **Repo:**
    [IST 27 Capstone Project](https://github.com/darshan99009/IST_27-Capstone_Project)
    """)

# ════════════════════════════════════════════════════════════
#   HEADER
# ════════════════════════════════════════════════════════════
st.markdown("# ⚙️ Turbofan Engine RUL Prediction")
st.markdown(
    f"**Hybrid XGBoost–LSTM** &nbsp;|&nbsp; Dataset: **{fd}** "
    f"&nbsp;|&nbsp; {DATASET_CFG[fd]['label']} &nbsp;|&nbsp; NASA C-MAPSS"
)
st.markdown("---")

# ── Load models ───────────────────────────────────────────────
with st.spinner(f"Loading models for {fd}..."):
    models_dict, load_err = load_models(fd)

if load_err:
    st.error(f"⚠️ {load_err}")
    st.info("Dashboard running in **demo mode** — upload model files to `/models` folder in the repo.")
    DEMO = True
else:
    st.success(
        f"✅ Models loaded — **{fd}** &nbsp;|&nbsp; "
        f"α = {models_dict['alpha']:.4f} &nbsp;|&nbsp; "
        f"Features = {len(models_dict['feature_cols'])}"
    )
    DEMO = False

# ── Run predictions ───────────────────────────────────────────
pred_lstm = pred_xgb = pred_hybrid = true_rul = engine_ids = None

if not DEMO and test_file is not None:
    with st.spinner("Preprocessing & running inference..."):
        try:
            df_test = parse_txt(test_file)
            pred_lstm, pred_xgb, pred_hybrid, engine_ids = predict(models_dict, df_test, fd)
            if rul_file:
                rul_df   = pd.read_csv(io.StringIO(rul_file.read().decode()), sep=r'\s+', header=None, names=['RUL'])
                true_rul = rul_df['RUL'].values.clip(max=RUL_CAP).astype(np.float32)
            st.success(f"✅ Predicted RUL for **{len(engine_ids)}** engines")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            DEMO = True

if pred_hybrid is None:
    # Demo synthetic data
    np.random.seed(42)
    rmse_map = {'FD001':13.2,'FD002':20.1,'FD003':14.5,'FD004':24.8}
    n_map    = {'FD001':100,'FD002':259,'FD003':100,'FD004':248}
    n        = n_map[fd]
    true_rul = np.sort(np.random.uniform(5,120,n))[::-1].astype(np.float32)
    r        = rmse_map[fd]
    pred_lstm   = np.clip(true_rul + np.random.normal(0,r+2,n),  0,125).astype(np.float32)
    pred_xgb    = np.clip(true_rul + np.random.normal(0,r+4,n),  0,125).astype(np.float32)
    pred_hybrid = np.clip(true_rul + np.random.normal(0,r,n),    0,125).astype(np.float32)
    engine_ids  = list(range(1, n+1))
    if not DEMO:  # had test file but no rul — keep true_rul=None
        true_rul = None

# ── KPI row ───────────────────────────────────────────────────
critical = int(np.sum(pred_hybrid < 20))
warning  = int(np.sum((pred_hybrid >= 20) & (pred_hybrid < 50)))
healthy  = int(np.sum(pred_hybrid >= 50))
metrics  = {m: compute_metrics(true_rul, p) for m,p in
            [('LSTM',pred_lstm),('XGBoost',pred_xgb),('Hybrid',pred_hybrid)]}

k1,k2,k3,k4,k5,k6 = st.columns(6)
with k1: st.metric("Total Engines",  len(engine_ids))
with k2: st.metric("🔴 Critical",     critical, "RUL < 20 cycles")
with k3: st.metric("🟡 Warning",      warning,  "RUL 20–50 cycles")
with k4: st.metric("🟢 Healthy",      healthy,  "RUL > 50 cycles")
with k5: st.metric("Avg Hybrid RUL", f"{np.mean(pred_hybrid):.1f}")
with k6:
    if metrics['Hybrid']:
        st.metric("Hybrid RMSE", f"{metrics['Hybrid']['RMSE']:.2f}",
                  f"R²={metrics['Hybrid']['R2']:.3f}")
    else:
        st.metric("Min RUL", f"{np.min(pred_hybrid):.1f}", "most critical engine")

st.markdown("")

# ════════════════════════════════════════════════════════════
#   TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Engine Inspector",
    "📊  Model Comparison",
    "🗺️  Fleet Heatmap",
    "📋  Predictions Table",
])

# ────────────────────────────────────────────────────────────
#  TAB 1 — Engine Inspector
# ────────────────────────────────────────────────────────────
with tab1:
    col_sel, _ = st.columns([1,3])
    with col_sel:
        eng_sel = st.selectbox("Select Engine ID", engine_ids,
                               format_func=lambda x: f"Engine {x}")
    idx     = engine_ids.index(eng_sel)
    rul_h   = float(pred_hybrid[idx])
    rul_l   = float(pred_lstm[idx])
    rul_x   = float(pred_xgb[idx])
    stxt, scol = rul_status(rul_h)

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div style='background:#0d1627;border:1px solid #1e3050;border-radius:10px;
                    padding:20px;text-align:center;margin-top:4px'>
            <div style='color:#6080a0;font-size:0.7rem;letter-spacing:2px;text-transform:uppercase'>
                Engine {eng_sel} Status</div>
            <div style='color:{scol};font-size:1.6rem;font-weight:700;margin:8px 0'>{stxt}</div>
            <div style='color:#00d4ff;font-family:monospace;font-size:2.2rem;font-weight:700'>{rul_h:.0f}</div>
            <div style='color:#6080a0;font-size:0.75rem'>cycles remaining</div>
        </div>""", unsafe_allow_html=True)
    with c2: st.metric("Hybrid RUL",  f"{rul_h:.1f}")
    with c3: st.metric("LSTM RUL",    f"{rul_l:.1f}", f"{rul_l-rul_h:+.1f} vs Hybrid")
    with c4:
        st.metric("XGBoost RUL", f"{rul_x:.1f}", f"{rul_x-rul_h:+.1f} vs Hybrid")
        if true_rul is not None and idx < len(true_rul):
            st.metric("True RUL", f"{float(true_rul[idx]):.1f}",
                      f"Error {rul_h-float(true_rul[idx]):+.1f}")

    st.markdown("")
    cg, cb = st.columns([1,2])

    with cg:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rul_h,
            delta={'reference':50,'font':{'color':'#a0b8d8'}},
            title={'text':f"Hybrid RUL — Engine {eng_sel}",'font':{'color':'#a0b8d8','size':13}},
            number={'font':{'color':'#00d4ff','family':'Share Tech Mono'},'suffix':' cycles'},
            gauge={
                'axis':{'range':[0,125],'tickcolor':'#6080a0','tickfont':{'color':'#6080a0'}},
                'bar': {'color':scol},
                'bgcolor':'#0d1220','bordercolor':'#1e3050',
                'steps':[{'range':[0,20],'color':'#1a0508'},
                         {'range':[20,50],'color':'#1a1005'},
                         {'range':[50,125],'color':'#051a0a'}],
                'threshold':{'line':{'color':'#ff4060','width':3},'thickness':0.8,'value':20},
            }
        ))
        fig.update_layout(paper_bgcolor='#0a0e1a',font=dict(color='#8090b0'),
                          height=260,margin=dict(t=50,b=10,l=10,r=10))
        pc(fig)

    with cb:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['LSTM','XGBoost','Hybrid'], y=[rul_l,rul_x,rul_h],
            marker_color=[C['lstm'],C['xgb'],C['hybrid']],
            text=[f'{v:.1f}' for v in [rul_l,rul_x,rul_h]],
            textposition='outside', textfont=dict(color='#a0b8d8',size=13),
        ))
        if true_rul is not None and idx < len(true_rul):
            fig.add_hline(y=float(true_rul[idx]), line_color='#ff4060',
                          line_dash='dash', line_width=2,
                          annotation_text=f'True RUL={float(true_rul[idx]):.0f}',
                          annotation_font_color='#ff4060')
        fig.add_hline(y=20, line_color='#ff4060', line_dash='dot', line_width=1,
                      annotation_text='Critical (20)', annotation_font_size=10,
                      annotation_font_color='#ff4060')
        pfig(fig, title=f'Model Predictions — Engine {eng_sel}',
             ytitle='Predicted RUL', yrange=[0,140], height=260, showlegend=False)
        pc(fig)

# ────────────────────────────────────────────────────────────
#  TAB 2 — Model Comparison
# ────────────────────────────────────────────────────────────
with tab2:
    if true_rul is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for model,preds,color in [('LSTM',pred_lstm,C['lstm']),
                                       ('XGBoost',pred_xgb,C['xgb']),
                                       ('Hybrid',pred_hybrid,C['hybrid'])]:
                r = float(np.sqrt(np.mean((true_rul-preds)**2)))
                m = float(mean_absolute_error(true_rul,preds))
                fig.add_trace(go.Bar(name=model,x=['RMSE','MAE'],y=[r,m],
                    marker_color=color,
                    text=[f'{r:.2f}',f'{m:.2f}'],textposition='outside',
                    textfont=dict(color='#a0b8d8')))
            pfig(fig, title='RMSE & MAE', height=340, barmode='group')
            pc(fig)

        with col2:
            r2s = [float(r2_score(true_rul,p)) for p in [pred_lstm,pred_xgb,pred_hybrid]]
            fig = go.Figure(go.Bar(
                x=['LSTM','XGBoost','Hybrid'],y=r2s,
                marker_color=[C['lstm'],C['xgb'],C['hybrid']],
                text=[f'{v:.4f}' for v in r2s],textposition='outside',
                textfont=dict(color='#a0b8d8')))
            pfig(fig, title='R² Score', height=340, ytitle='R²', yrange=[0,1.12])
            pc(fig)

        # Scatter plots
        c1,c2,c3 = st.columns(3)
        for col,model,preds,color in zip([c1,c2,c3],
                ['LSTM','XGBoost','Hybrid'],
                [pred_lstm,pred_xgb,pred_hybrid],
                [C['lstm'],C['xgb'],C['hybrid']]):
            with col:
                rmse_v = float(np.sqrt(np.mean((true_rul-preds)**2)))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=true_rul,y=preds,mode='markers',
                    marker=dict(color=color,size=5,opacity=0.6)))
                fig.add_trace(go.Scatter(x=[0,125],y=[0,125],mode='lines',
                    line=dict(color='#ff4060',dash='dash',width=1.5)))
                pfig(fig, title=f'{model} RMSE={rmse_v:.2f}',
                     xtitle='Actual RUL', ytitle='Predicted RUL',
                     height=270, showlegend=False)
                pc(fig)

        # Residuals + sorted curve
        cr1,cr2 = st.columns(2)
        with cr1:
            fig = go.Figure()
            for model,preds,color in [('LSTM',pred_lstm,C['lstm']),
                                       ('XGBoost',pred_xgb,C['xgb']),
                                       ('Hybrid',pred_hybrid,C['hybrid'])]:
                fig.add_trace(go.Histogram(x=true_rul-preds,name=model,
                    opacity=0.7,marker_color=color,nbinsx=30))
            fig.add_vline(x=0,line_color='#ff4060',line_dash='dash',line_width=2)
            pfig(fig, title='Residual Distributions', height=300, barmode='overlay')
            pc(fig)
        with cr2:
            order = np.argsort(true_rul)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(order))),y=true_rul[order],
                mode='lines',name='Actual',line=dict(color=C['actual'],width=2)))
            for model,preds,color,dash in [
                    ('LSTM',pred_lstm,C['lstm'],'dot'),
                    ('Hybrid',pred_hybrid,C['hybrid'],'solid')]:
                fig.add_trace(go.Scatter(x=list(range(len(order))),y=preds[order],
                    mode='lines',name=model,line=dict(color=color,width=1.5,dash=dash)))
            pfig(fig, title='Pred vs Actual (sorted)', height=300,
                 xtitle='Engine index', ytitle='RUL')
            pc(fig)

    else:
        st.info("📎 Upload a **RUL file** alongside the test file to see full model scoring.")
        # Still show prediction distribution
        fig = go.Figure()
        for model,preds,color in [('LSTM',pred_lstm,C['lstm']),
                                   ('XGBoost',pred_xgb,C['xgb']),
                                   ('Hybrid',pred_hybrid,C['hybrid'])]:
            fig.add_trace(go.Histogram(x=preds,name=model,opacity=0.7,
                marker_color=color,nbinsx=30))
        pfig(fig, title='Predicted RUL Distribution', height=380,
             barmode='overlay', xtitle='Predicted RUL', ytitle='Count')
        pc(fig)

# ────────────────────────────────────────────────────────────
#  TAB 3 — Fleet Heatmap
# ────────────────────────────────────────────────────────────
with tab3:
    n      = len(pred_hybrid)
    ncols  = min(20, n)
    nrows  = int(np.ceil(n / ncols))
    grid   = np.full((nrows, ncols), np.nan)
    for i,v in enumerate(pred_hybrid):
        grid[i//ncols, i%ncols] = v

    hover = [[f'Engine {r*ncols+c+1}<br>Hybrid RUL: {grid[r,c]:.0f}<br>{rul_status(grid[r,c])[0]}'
              if not np.isnan(grid[r,c]) else ''
              for c in range(ncols)] for r in range(nrows)]

    fig = go.Figure(go.Heatmap(
        z=grid,
        colorscale=[[0,'#ff1a3a'],[0.16,'#ff4060'],
                    [0.4,'#ffaa00'],[0.5,'#ffe040'],[1.0,'#00ff88']],
        zmin=0, zmax=125,
        text=hover, hovertemplate='%{text}<extra></extra>',
        showscale=True,
        colorbar=dict(title=dict(text='RUL',font=dict(color='#8090b0')),
                      tickfont=dict(color='#8090b0')),
    ))
    pfig(fig, title=f'Fleet Health Heatmap — {fd}  ({n} engines)',
         height=max(250, nrows*45+100),
         xtitle='Engine slot', ytitle='Row')
    pc(fig)

    # Summary donut
    col_d, col_s = st.columns([1,2])
    with col_d:
        fig = go.Figure(go.Pie(
            labels=['Critical','Warning','Healthy'],
            values=[critical, warning, healthy],
            hole=0.6,
            marker_colors=['#ff4060','#ffaa00','#00ff88'],
            textfont=dict(color='white'),
        ))
        fig.update_layout(
            paper_bgcolor='#0a0e1a',
            font=dict(color='#8090b0'),
            height=280,
            showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#8090b0')),
            annotations=[dict(text=f'{n}<br>engines',x=0.5,y=0.5,
                              font_size=16,showarrow=False,
                              font=dict(color='#00d4ff'))],
        )
        pc(fig)

    with col_s:
        st.markdown("### Fleet Summary")
        st.markdown(f"""
        | Status | Count | % |
        |--------|------:|--:|
        | 🔴 Critical (RUL < 20)   | {critical} | {100*critical/n:.1f}% |
        | 🟡 Warning  (RUL 20–50)  | {warning}  | {100*warning/n:.1f}% |
        | 🟢 Healthy  (RUL > 50)   | {healthy}  | {100*healthy/n:.1f}% |
        | **Total**                | **{n}**    | 100% |

        **Average RUL:** {np.mean(pred_hybrid):.1f} cycles
        **Minimum RUL:** {np.min(pred_hybrid):.1f} cycles ← most urgent
        **Maximum RUL:** {np.max(pred_hybrid):.1f} cycles
        """)

# ────────────────────────────────────────────────────────────
#  TAB 4 — Predictions Table
# ────────────────────────────────────────────────────────────
with tab4:
    df_out = pd.DataFrame({
        'Engine ID':   engine_ids,
        'Hybrid RUL':  np.round(pred_hybrid, 2),
        'LSTM RUL':    np.round(pred_lstm,   2),
        'XGBoost RUL': np.round(pred_xgb,    2),
        'Status':      [rul_status(v)[0] for v in pred_hybrid],
    })
    if true_rul is not None and len(true_rul)==len(engine_ids):
        df_out.insert(1,'True RUL', np.round(true_rul,2))


    col_f1, col_f2, col_f3 = st.columns([1,1,2])
    with col_f1:
        status_f = st.multiselect("Filter Status",
            ['🔴 CRITICAL','🟡 WARNING','🟢 HEALTHY'],
            default=['🔴 CRITICAL','🟡 WARNING','🟢 HEALTHY'])
    with col_f2:
        sort_col = st.selectbox("Sort by", ['Hybrid RUL','Engine ID','Status'])

    df_show = df_out[df_out['Status'].isin(status_f)].sort_values(sort_col)
    st.markdown(f"Showing **{len(df_show)}** / **{len(df_out)}** engines")
    st.dataframe(df_show, width='stretch', height=480)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#2a4060;font-size:0.8rem;padding:8px'>
    ⚙️ Hybrid XGBoost–LSTM &nbsp;|&nbsp; NASA C-MAPSS &nbsp;|&nbsp;
    <a href='https://github.com/darshan99009/IST_27-Capstone_Project'
       style='color:#3a6090;text-decoration:none'>GitHub</a>
    &nbsp;|&nbsp; Streamlit Cloud
</div>
""", unsafe_allow_html=True)
