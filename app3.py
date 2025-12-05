import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

df = pd.read_csv("data/Loan_approval_data_2025.csv")

# Configuration de la page
st.set_page_config(
    page_title="Credit Analytics Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thème de couleurs personnalisé
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
ACCENT_COLOR = "#2ca02c"
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-title">Credit Analytics Dashboard</h1>', unsafe_allow_html=True)

# Vérification que le DataFrame df existe
if 'df' not in locals() and 'df' not in globals():
    st.error("❌ Erreur : Le DataFrame 'df' n'est pas chargé. Veuillez importer vos données d'abord.")
    st.stop()

# Fonction pour détecter les colonnes ID
def is_id_column(col_name):
    """Détecte si une colonne est un identifiant"""
    col_lower = col_name.lower()
    id_keywords = ['id', '_id', 'customer_id', 'user_id', 'client_id', 'identifier']
    return any(keyword in col_lower for keyword in id_keywords)

# Identification des colonnes numériques et catégorielles
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Exclure les colonnes ID des colonnes catégorielles
categorical_cols_filtered = [col for col in categorical_cols if not is_id_column(col)]

# ========== SIDEBAR : NAVIGATION ==========
st.sidebar.markdown("## 🧭 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Sélectionnez une section :",
    ["📈 Statistiques", "📊 Visualisations"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Informations Dataset")
st.sidebar.metric("📏 Lignes", f"{df.shape[0]:,}")
st.sidebar.metric("📊 Colonnes", df.shape[1])
st.sidebar.metric("🔢 Numériques", len(numeric_cols))
st.sidebar.metric("🏷️ Catégorielles", len(categorical_cols))

st.sidebar.markdown("---")
st.sidebar.markdown("**Créé avec Streamlit & Plotly**")
st.sidebar.markdown("*Dashboard v2.0*")

# ========== PAGE : STATISTIQUES ==========
if page == "📈 Statistiques":
    st.markdown('<div class="section-header">📈 Analyse Statistique</div>', unsafe_allow_html=True)
    
    # Statistiques des variables numériques
    st.markdown("### 🔢 Variables Numériques")
    if len(numeric_cols) > 0:
        desc_stats = df[numeric_cols].describe().T
        desc_stats = desc_stats.round(2)
        
        # Affichage avec style
        st.dataframe(
            desc_stats.style.background_gradient(cmap='Blues', subset=['mean', '50%'])
                           .background_gradient(cmap='Oranges', subset=['std'])
                           .format("{:.2f}"),
            use_container_width=True,
            height=400
        )
        
        # Informations complémentaires
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**📊 Total variables numériques :** {len(numeric_cols)}")
        with col2:
            total_missing = df[numeric_cols].isnull().sum().sum()
            st.warning(f"**⚠️ Valeurs manquantes :** {total_missing}")
        with col3:
            st.success(f"**✅ Valeurs totales :** {df[numeric_cols].notna().sum().sum():,}")
    else:
        st.warning("⚠️ Aucune colonne numérique trouvée dans le dataset.")
    
    st.markdown("---")
    
    # Statistiques des variables catégorielles
    st.markdown("### 🏷️ Variables Catégorielles")
    if len(categorical_cols_filtered) > 0:
        selected_cat_col = st.selectbox(
            "Sélectionnez une variable catégorielle :",
            categorical_cols_filtered,
            key="cat_stats"
        )
        
        if selected_cat_col:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**📊 Distribution : {selected_cat_col}**")
                value_counts = df[selected_cat_col].value_counts()
                value_counts_df = pd.DataFrame({
                    'Catégorie': value_counts.index,
                    'Nombre': value_counts.values,
                    'Pourcentage (%)': (value_counts.values / len(df) * 100).round(2)
                })
                
                st.dataframe(
                    value_counts_df.style.background_gradient(cmap='Greens', subset=['Nombre'])
                                         .format({'Pourcentage (%)': '{:.2f}%'}),
                    use_container_width=True,
                    height=350
                )
            
            with col2:
                st.markdown(f"**📈 Métriques : {selected_cat_col}**")
                # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Catégories uniques", df[selected_cat_col].nunique())
                st.markdown('</div>', unsafe_allow_html=True)
                
                # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                mode_value = df[selected_cat_col].mode()[0] if len(df[selected_cat_col].mode()) > 0 else "N/A"
                st.metric("⭐ Plus fréquente", mode_value)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📊 Fréquence max", f"{df[selected_cat_col].value_counts().max():,}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                # missing = df[selected_cat_col].isnull().sum()
                # st.metric("❓ Valeurs manquantes", missing)
                # st.markdown('</div>', unsafe_allow_html=True)
            
            # Graphique de distribution
            st.markdown("---")
            st.markdown(f"**📊 Visualisation : {selected_cat_col}**")
            
            value_counts_plot = df[selected_cat_col].value_counts().head(15).reset_index()
            value_counts_plot.columns = [selected_cat_col, 'count']
            
            fig = px.bar(
                value_counts_plot,
                x=selected_cat_col,
                y='count',
                title=f'Distribution de {selected_cat_col} (Top 15)',
                labels={'count': 'Nombre', selected_cat_col: selected_cat_col},
                color='count',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis_title=selected_cat_col,
                yaxis_title="Nombre d'occurrences",
                showlegend=False,
                height=500,
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Aucune colonne catégorielle valide trouvée (colonnes ID exclues).")

# ========== PAGE : VISUALISATIONS ==========
elif page == "📊 Visualisations":
    st.markdown('<div class="section-header">📊 Visualisations Interactives</div>', unsafe_allow_html=True)
    
    # Histogrammes
    st.markdown("### 📊 Histogrammes")
    st.markdown("*Visualisez la distribution des variables numériques*")
    
    if len(numeric_cols) > 0:
        hist_col = st.selectbox(
            "Sélectionnez une variable numérique :",
            numeric_cols,
            key="hist"
        )
        
        if hist_col:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig = px.histogram(
                    df, 
                    x=hist_col,
                    nbins=30,
                    title=f"📊 Distribution de {hist_col}",
                    labels={hist_col: hist_col},
                    color_discrete_sequence=[PRIMARY_COLOR]
                )
                fig.update_layout(
                    xaxis_title=hist_col,
                    yaxis_title="Fréquence",
                    showlegend=False,
                    height=500,
                    font=dict(size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**📈 Statistiques**")
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📊 Moyenne", f"{df[hist_col].mean():.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📍 Médiane", f"{df[hist_col].median():.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📏 Écart-type", f"{df[hist_col].std():.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Aucune colonne numérique disponible.")
    
    st.markdown("---")
    
    # Boxplots
    st.markdown("### 📦 Boxplots")
    st.markdown("*Identifiez les valeurs aberrantes et la dispersion*")
    
    if len(numeric_cols) > 0:
        box_col = st.selectbox(
            "Sélectionnez une variable numérique :",
            numeric_cols,
            key="box"
        )
        
        if box_col:
            fig = px.box(
                df,
                y=box_col,
                title=f"📦 Boxplot de {box_col}",
                color_discrete_sequence=[SECONDARY_COLOR]
            )
            fig.update_layout(
                yaxis_title=box_col,
                showlegend=False,
                height=500,
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Aucune colonne numérique disponible.")
    
    st.markdown("---")
    
    # Matrice de corrélation
    st.markdown("### 🔥 Matrice de Corrélation")
    st.markdown("*Explorez les relations entre variables numériques*")
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title='🔥 Matrice de Corrélation',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )
        fig.update_layout(
            height=700,
            font=dict(size=11)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top corrélations
        st.markdown("**🔝 Top 5 Corrélations Positives**")
        corr_pairs = corr_matrix.unstack()
        corr_pairs = corr_pairs[corr_pairs < 1].sort_values(ascending=False).head(5)
        
        for idx, (pair, value) in enumerate(corr_pairs.items(), 1):
            st.markdown(f"**{idx}.** `{pair[0]}` ↔️ `{pair[1]}` : **{value:.3f}**")
    else:
        st.warning("⚠️ Au moins 2 colonnes numériques sont nécessaires.")
    
    st.markdown("---")
    
    # Scatter plot
    st.markdown("### 🔵 Nuage de Points")
    st.markdown("*Analysez la relation entre deux variables*")
    
    if len(numeric_cols) >= 2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox(
                "Variable X :",
                numeric_cols,
                key="scatter_x"
            )
        
        with col2:
            y_col = st.selectbox(
                "Variable Y :",
                [col for col in numeric_cols if col != x_col],
                key="scatter_y"
            )
        
        with col3:
            color_option = st.selectbox(
                "Colorier par :",
                ['Aucune'] + categorical_cols_filtered,
                key="scatter_color"
            )
        
        if x_col and y_col:
            if color_option != 'Aucune':
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_option,
                    title=f"🔵 Relation : {x_col} vs {y_col}",
                    labels={x_col: x_col, y_col: y_col},
                    hover_data=df.columns,
                    color_discrete_sequence=COLOR_PALETTE
                )
            else:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"🔵 Relation : {x_col} vs {y_col}",
                    labels={x_col: x_col, y_col: y_col},
                    hover_data=df.columns,
                    color_discrete_sequence=[ACCENT_COLOR]
                )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=600,
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Au moins 2 colonnes numériques sont nécessaires.")