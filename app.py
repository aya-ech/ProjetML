import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Credit Analytics Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS style
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
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #ff7f0e;
    }
    .nav-link {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("# 🧭 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a page:",
    ["📊 Dashboard", "🔮 Credit Prediction"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**Credit Analytics Platform**\n\n"
    "- 📊 Explore data insights\n"
    "- 🔮 Predict loan approvals\n"
    "- 📈 Analyze trends"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**v2.0** | Built with Streamlit")

# Load the selected page
if page == "📊 Dashboard":
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    df = pd.read_csv("data/Loan_approval_data_2025.csv")

    # Thème de couleurs personnalisé
    PRIMARY_COLOR = "#1f77b4"
    SECONDARY_COLOR = "#ff7f0e"
    ACCENT_COLOR = "#2ca02c"
    COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Titre principal
    st.markdown('<h1 class="main-title">Credit Analytics Dashboard</h1>', unsafe_allow_html=True)

    # Vérification que le DataFrame df existe
    if 'df' not in locals() and 'df' not in globals():
        st.error("❌ Error: DataFrame 'df' is not loaded. Please import your data first.")
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

    # ========== SUB-NAVIGATION ==========
    st.markdown("---")
    sub_page = st.radio(
        "Select a section:",
        ["📈 Statistics", "📊 Visualizations"],
        horizontal=True
    )
    st.markdown("---")

    # ========== PAGE : STATISTIQUES ==========
    if sub_page == "📈 Statistics":
        st.markdown('<div class="section-header">📈 Statistical Analysis</div>', unsafe_allow_html=True)
        
        # Statistiques des variables numériques
        st.markdown("### 🔢 Numeric Variables")
        if len(numeric_cols) > 0:
            desc_stats = df[numeric_cols].describe().T
            desc_stats = desc_stats.round(2)
            
            # Affichage avec style
            st.dataframe(
                desc_stats,
                use_container_width=True,
                height=400
            )
            
            # Informations complémentaires
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**📊 Total numeric variables:** {len(numeric_cols)}")
            with col2:
                total_missing = df[numeric_cols].isnull().sum().sum()
                st.warning(f"**⚠️ Missing values:** {total_missing}")
            with col3:
                st.success(f"**✅ Total values:** {df[numeric_cols].notna().sum().sum():,}")
        else:
            st.warning("⚠️ No numeric columns found in dataset.")
        
        st.markdown("---")
        
        # Statistiques des variables catégorielles
        st.markdown("### 🏷️ Categorical Variables")
        if len(categorical_cols_filtered) > 0:
            selected_cat_col = st.selectbox(
                "Select a categorical variable:",
                categorical_cols_filtered,
                key="cat_stats"
            )
            
            if selected_cat_col:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📊 Distribution: {selected_cat_col}**")
                    value_counts = df[selected_cat_col].value_counts()
                    value_counts_df = pd.DataFrame({
                        'Category': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage (%)': (value_counts.values / len(df) * 100).round(2)
                    })
                    
                    st.dataframe(
                        value_counts_df,
                        use_container_width=True,
                        height=350
                    )
                
                with col2:
                    st.markdown(f"**📈 Metrics: {selected_cat_col}**")
                    st.metric("🎯 Unique categories", df[selected_cat_col].nunique())
                    
                    mode_value = df[selected_cat_col].mode()[0] if len(df[selected_cat_col].mode()) > 0 else "N/A"
                    st.metric("⭐ Most frequent", mode_value)
                    
                    st.metric("📊 Max frequency", f"{df[selected_cat_col].value_counts().max():,}")
                
                # Graphique de distribution
                st.markdown("---")
                st.markdown(f"**📊 Visualization: {selected_cat_col}**")
                
                value_counts_plot = df[selected_cat_col].value_counts().head(15).reset_index()
                value_counts_plot.columns = [selected_cat_col, 'count']
                
                fig = px.bar(
                    value_counts_plot,
                    x=selected_cat_col,
                    y='count',
                    title=f'Distribution of {selected_cat_col} (Top 15)',
                    labels={'count': 'Count', selected_cat_col: selected_cat_col},
                    color='count',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    xaxis_title=selected_cat_col,
                    yaxis_title="Number of occurrences",
                    showlegend=False,
                    height=500,
                    font=dict(size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No valid categorical columns found (ID columns excluded).")

    # ========== PAGE : VISUALISATIONS ==========
    elif sub_page == "📊 Visualizations":
        st.markdown('<div class="section-header">📊 Interactive Visualizations</div>', unsafe_allow_html=True)
        
        # Histogrammes
        st.markdown("### 📊 Histograms")
        st.markdown("*Visualize the distribution of numeric variables*")
        
        if len(numeric_cols) > 0:
            hist_col = st.selectbox(
                "Select a numeric variable:",
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
                        title=f"📊 Distribution of {hist_col}",
                        labels={hist_col: hist_col},
                        color_discrete_sequence=[PRIMARY_COLOR]
                    )
                    fig.update_layout(
                        xaxis_title=hist_col,
                        yaxis_title="Frequency",
                        showlegend=False,
                        height=500,
                        font=dict(size=12)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**📈 Statistics**")
                    st.metric("📊 Mean", f"{df[hist_col].mean():.2f}")
                    st.metric("📍 Median", f"{df[hist_col].median():.2f}")
                    st.metric("📏 Std Dev", f"{df[hist_col].std():.2f}")
        else:
            st.warning("⚠️ No numeric columns available.")
        
        st.markdown("---")
        
        # Boxplots
        st.markdown("### 📦 Boxplots")
        st.markdown("*Identify outliers and dispersion*")
        
        if len(numeric_cols) > 0:
            box_col = st.selectbox(
                "Select a numeric variable:",
                numeric_cols,
                key="box"
            )
            
            if box_col:
                fig = px.box(
                    df,
                    y=box_col,
                    title=f"📦 Boxplot of {box_col}",
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
            st.warning("⚠️ No numeric columns available.")
        
        st.markdown("---")
        
        # Matrice de corrélation
        st.markdown("### 🔥 Correlation Matrix")
        st.markdown("*Explore relationships between numeric variables*")
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                title='🔥 Correlation Matrix',
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
            st.markdown("**🔝 Top 5 Positive Correlations**")
            corr_pairs = corr_matrix.unstack()
            corr_pairs = corr_pairs[corr_pairs < 1].sort_values(ascending=False).head(5)
            
            for idx, (pair, value) in enumerate(corr_pairs.items(), 1):
                st.markdown(f"**{idx}.** `{pair[0]}` ↔️ `{pair[1]}` : **{value:.3f}**")
        else:
            st.warning("⚠️ At least 2 numeric columns are required.")
        
        st.markdown("---")
        
        # Scatter plot
        st.markdown("### 🔵 Scatter Plot")
        st.markdown("*Analyze the relationship between two variables*")
        
        if len(numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_col = st.selectbox(
                    "Variable X:",
                    numeric_cols,
                    key="scatter_x"
                )
            
            with col2:
                y_col = st.selectbox(
                    "Variable Y:",
                    [col for col in numeric_cols if col != x_col],
                    key="scatter_y"
                )
            
            with col3:
                color_option = st.selectbox(
                    "Color by:",
                    ['None'] + categorical_cols_filtered,
                    key="scatter_color"
                )
            
            if x_col and y_col:
                if color_option != 'None':
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color=color_option,
                        title=f"🔵 Relationship: {x_col} vs {y_col}",
                        labels={x_col: x_col, y_col: y_col},
                        hover_data=df.columns,
                        color_discrete_sequence=COLOR_PALETTE
                    )
                else:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"🔵 Relationship: {x_col} vs {y_col}",
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
            st.warning("⚠️ At least 2 numeric columns are required.")

elif page == "🔮 Credit Prediction":
    import pandas as pd
    import numpy as np
    import joblib
    import os
    from models.models import LinearSVCScratch, LogisticRegressionScratch, KNN_Scratch

    # Fonction de prétraitement
    def preprocess_user_input(user_input):
        df = pd.DataFrame([user_input])

        # Derived numeric features
        df['debt_to_income_ratio'] = df['current_debt'] / df['annual_income']
        df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']

        # One-hot encoding for occupation_status (only Self-Employed and Student)
        df['occupation_status_Self-Employed'] = int(user_input.get('occupation_status') == 'Self-Employed')
        df['occupation_status_Student'] = int(user_input.get('occupation_status') == 'Student')

        # One-hot encoding for product_type
        df['product_type_Line of Credit'] = int(user_input.get('product_type') == 'Line of Credit')
        df['product_type_Personal Loan'] = int(user_input.get('product_type') == 'Personal Loan')

        # One-hot encoding for loan_intent
        df['loan_intent_Debt Consolidation'] = int(user_input.get('loan_intent') == 'Debt Consolidation')
        df['loan_intent_Education'] = int(user_input.get('loan_intent') == 'Education')
        df['loan_intent_Home Improvement'] = int(user_input.get('loan_intent') == 'Home Improvement')
        df['loan_intent_Medical'] = int(user_input.get('loan_intent') == 'Medical')
        df['loan_intent_Personal'] = int(user_input.get('loan_intent') == 'Personal')

        # Age bins
        df['age_bin_26-35'] = int(26 <= df['age'][0] <= 35)
        df['age_bin_36-45'] = int(36 <= df['age'][0] <= 45)
        df['age_bin_46-55'] = int(46 <= df['age'][0] <= 55)
        df['age_bin_56-65'] = int(56 <= df['age'][0] <= 65)
        df['age_bin_65+'] = int(df['age'][0] > 65)

        # Years employed bins
        df['years_employed_bin_2-3'] = int(2 <= df['years_employed'][0] <= 3)
        df['years_employed_bin_4-5'] = int(4 <= df['years_employed'][0] <= 5)
        df['years_employed_bin_6-10'] = int(6 <= df['years_employed'][0] <= 10)
        df['years_employed_bin_11-20'] = int(11 <= df['years_employed'][0] <= 20)
        df['years_employed_bin_20+'] = int(df['years_employed'][0] > 20)

        # Credit history bins
        df['credit_history_bin_2-3'] = int(2 <= df['credit_history_years'][0] <= 3)
        df['credit_history_bin_4-5'] = int(4 <= df['credit_history_years'][0] <= 5)
        df['credit_history_bin_6-10'] = int(6 <= df['credit_history_years'][0] <= 10)
        df['credit_history_bin_11-20'] = int(11 <= df['credit_history_years'][0] <= 20)
        df['credit_history_bin_20+'] = int(df['credit_history_years'][0] > 20)

        # Drop unnecessary columns
        df = df.drop(columns=['age', 'years_employed', 'credit_history_years', 
                            'occupation_status', 'product_type', 'loan_intent'])

        # Réorganiser les colonnes dans l'ordre exact requis
        columns_order = [
            'annual_income', 'credit_score', 'savings_assets', 'current_debt',
            'defaults_on_file', 'delinquencies_last_2yrs', 'derogatory_marks',
            'loan_amount', 'interest_rate', 'debt_to_income_ratio',
            'loan_to_income_ratio', 'occupation_status_Self-Employed',
            'occupation_status_Student', 'product_type_Line of Credit',
            'product_type_Personal Loan', 'loan_intent_Debt Consolidation',
            'loan_intent_Education', 'loan_intent_Home Improvement',
            'loan_intent_Medical', 'loan_intent_Personal', 'age_bin_26-35',
            'age_bin_36-45', 'age_bin_46-55', 'age_bin_56-65', 'age_bin_65+',
            'years_employed_bin_2-3', 'years_employed_bin_4-5',
            'years_employed_bin_6-10', 'years_employed_bin_11-20',
            'years_employed_bin_20+', 'credit_history_bin_2-3',
            'credit_history_bin_4-5', 'credit_history_bin_6-10',
            'credit_history_bin_11-20', 'credit_history_bin_20+'
        ]
        
        df = df[columns_order]
        return df

    # Main title
    st.markdown('<h1 class="main-title">Credit Approval Prediction</h1>', unsafe_allow_html=True)

    # Form section
    st.markdown('<div class="section-header">Information Entry</div>', unsafe_allow_html=True)
    st.markdown("*Please fill in all fields below to get a prediction*")

    # Create the form
    with st.form("credit_form"):
        st.markdown("### Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
                help="Applicant's age in years"
            )
            
            annual_income = st.number_input(
                "Annual Income ($)",
                min_value=0,
                max_value=1000000,
                value=50000,
                step=1000,
                help="Annual income in dollars"
            )
            
            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=650,
                step=10,
                help="Credit score (300-850)"
            )
            
            savings_assets = st.number_input(
                "Savings and Assets ($)",
                min_value=0,
                max_value=1000000,
                value=10000,
                step=1000,
                help="Total amount of savings and assets"
            )
        
        with col2:
            years_employed = st.number_input(
                "Years Employed",
                min_value=0,
                max_value=50,
                value=5,
                step=1,
                help="Number of years employed"
            )
            
            credit_history_years = st.number_input(
                "Credit History (years)",
                min_value=0,
                max_value=50,
                value=8,
                step=1,
                help="Number of years of credit history"
            )
            
            current_debt = st.number_input(
                "Current Debt ($)",
                min_value=0,
                max_value=500000,
                value=5000,
                step=1000,
                help="Total amount of current debt"
            )
            
            defaults_on_file = st.number_input(
                "Defaults on File",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help="Number of payment defaults on record"
            )
        
        st.markdown("---")
        
        st.markdown("### Financial History")
        col3, col4 = st.columns(2)
        
        with col3:
            delinquencies_last_2yrs = st.number_input(
                "Delinquencies (last 2 years)",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help="Number of delinquencies in the last 2 years"
            )
        
        with col4:
            derogatory_marks = st.number_input(
                "Derogatory Marks",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help="Number of derogatory marks on file"
            )
        
        st.markdown("---")
        
        st.markdown("### Loan Details")
        col5, col6 = st.columns(2)
        
        with col5:
            loan_amount = st.number_input(
                "Loan Amount ($)",
                min_value=1000,
                max_value=500000,
                value=20000,
                step=1000,
                help="Requested loan amount"
            )
            
            occupation_status = st.selectbox(
                "Occupation Status",
                options=['Employed', 'Self-Employed', 'Student'],
                help="Current employment situation"
            )
            
            loan_intent = st.selectbox(
                "Loan Intent",
                options=['Debt Consolidation', 'Education', 'Home Improvement', 'Medical', 'Personal'],
                help="Reason for loan request"
            )
        
        with col6:
            interest_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                max_value=30.0,
                value=10.0,
                step=0.5,
                format="%.2f",
                help="Proposed interest rate"
            )
            
            product_type = st.selectbox(
                "Product Type",
                options=['Personal Loan', 'Line of Credit'],
                help="Type of credit product"
            )
        
        st.markdown("---")
        
        # Submit button
        submitted = st.form_submit_button("Get Prediction")

    # Traitement après soumission
    if submitted:
        # Créer un dictionnaire avec toutes les valeurs saisies
        user_input = {
            'age': age,
            'years_employed': years_employed,
            'annual_income': annual_income,
            'credit_score': credit_score,
            'credit_history_years': credit_history_years,
            'savings_assets': savings_assets,
            'current_debt': current_debt,
            'defaults_on_file': defaults_on_file,
            'delinquencies_last_2yrs': delinquencies_last_2yrs,
            'derogatory_marks': derogatory_marks,
            'loan_amount': loan_amount,
            'interest_rate': interest_rate,
            'occupation_status': occupation_status,
            'product_type': product_type,
            'loan_intent': loan_intent
        }
        
        # Display entered data
        st.markdown("---")
        st.markdown('<div class="section-header">Entered Data Summary</div>', unsafe_allow_html=True)
        
        st.success("Form validated successfully!")
        
        # Display information in organized columns
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("#### Personal Profile")
            st.write(f"**Age:** {age} years")
            st.write(f"**Years employed:** {years_employed} years")
            st.write(f"**Annual income:** ${annual_income:,}")
            st.write(f"**Occupation status:** {occupation_status}")
        
        with col_b:
            st.markdown("#### Credit History")
            st.write(f"**Credit score:** {credit_score}")
            st.write(f"**History:** {credit_history_years} years")
            st.write(f"**Savings:** ${savings_assets:,}")
            st.write(f"**Current debt:** ${current_debt:,}")
            st.write(f"**Defaults on file:** {defaults_on_file}")
            st.write(f"**Delinquencies (2 years):** {delinquencies_last_2yrs}")
            st.write(f"**Derogatory marks:** {derogatory_marks}")
        
        with col_c:
            st.markdown("#### Loan Details")
            st.write(f"**Requested amount:** ${loan_amount:,}")
            st.write(f"**Interest rate:** {interest_rate}%")
            st.write(f"**Product type:** {product_type}")
            st.write(f"**Intent:** {loan_intent}")
        
        st.markdown("---")
        
        # Apply preprocessing (hidden from user)
        df_preprocessed = preprocess_user_input(user_input)
        
        st.markdown("---")
        
        # Model paths
        model_svc_path = "./models/pipeline_svc.pkl"
        model_logistic_path = "./models/pipeline_logistic.pkl"
        model_knn_path = "./models/pipeline_knn.pkl"
        
        # Load models silently
        models_loaded = {}
        
        if os.path.exists(model_svc_path):
            try:
                models_loaded['SVC'] = joblib.load(model_svc_path)
            except Exception as e:
                st.error(f"Error loading SVC model: {str(e)}")
        
        if os.path.exists(model_logistic_path):
            try:
                models_loaded['Logistic Regression'] = joblib.load(model_logistic_path)
            except Exception as e:
                st.error(f"Error loading Logistic model: {str(e)}")
        
        if os.path.exists(model_knn_path):
            try:
                models_loaded['KNN'] = joblib.load(model_knn_path)
            except Exception as e:
                st.error(f"Error loading KNN model: {str(e)}")
        
        # Run predictions
        if len(models_loaded) > 0:
            st.markdown('<div class="section-header">Model Predictions</div>', unsafe_allow_html=True)
            
            # Store prediction results
            predictions_results = []
            
            for model_name, model in models_loaded.items():
                try:
                    # Make prediction
                    prediction = model.predict(df_preprocessed)[0]
                    
                    # Convert prediction to text
                    prediction_text = "✅ Approved" if prediction == 1 else "❌ Rejected"
                    
                    # Get accuracy based on model name
                    accuracy_map = {
                        'KNN': '82%',
                        'Logistic Regression': '86%',
                        'SVC': '85%'
                    }
                    accuracy = accuracy_map.get(model_name, 'N/A')
                    
                    # Add to results
                    predictions_results.append({
                        'Model': model_name,
                        'Prediction': prediction_text,
                        'Value': int(prediction),
                        'Accuracy': accuracy
                    })
                    
                except Exception as e:
                    st.error(f"Error during prediction with {model_name}: {str(e)}")
            
            # Display results
            if len(predictions_results) > 0:
                st.success(f"Predictions completed successfully for {len(predictions_results)} model(s)!")
                
                st.markdown("---")
                st.markdown("### Prediction Results")
                
                # Create DataFrame to display results
                df_results = pd.DataFrame(predictions_results)
                
                # Visual display of predictions
                num_models = len(predictions_results)
                if num_models == 1:
                    cols = st.columns(1)
                elif num_models == 2:
                    cols = st.columns(2)
                else:
                    cols = st.columns(3)
                
                for idx, result in enumerate(predictions_results):
                    with cols[idx]:
                        st.markdown(f"#### {result['Model']}")
                        
                        if result['Value'] == 1:
                            st.success(f"**Prediction:** {result['Prediction']}")
                        else:
                            st.error(f"**Prediction:** {result['Prediction']}")
                        
                        st.info(f"**Accuracy:** {result['Accuracy']}")
                
                st.markdown("---")
                
                # Summary table
                st.markdown("### Summary Table")
                
                # Prepare table for display (without 'Value' column)
                df_display_results = df_results[['Model', 'Prediction', 'Accuracy']].copy()
                
                st.dataframe(
                    df_display_results,
                    width='stretch',
                    hide_index=True
                )
                
                # Download results
                st.markdown("---")
                csv_results = df_display_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results (CSV)",
                    data=csv_results,
                    file_name='prediction_results.csv',
                    mime='text/csv',
                )
