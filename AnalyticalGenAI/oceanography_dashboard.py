import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import folium
from streamlit_folium import folium_static
import requests
import warnings
import pandas as pd
from sqlalchemy import create_engine
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Oceanography Data Analysis Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .ocean-insight {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the oceanography dataset from PostgreSQL"""
    
    # PostgreSQL connection settings
    db_user = 'postgres'
    db_password = 'sama1234'
    db_host = 'localhost'
    db_port = '5432'
    db_name = 'SIH'
    table_name = 'sample_gold_layer'

    # Create database engine
    connection_string = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(connection_string)
    
    try:
        # Read the PostgreSQL table into a DataFrame
        df = pd.read_sql_table(table_name, engine)
        print(f"Successfully loaded data from table '{table_name}'.")
    except Exception as e:
        print(f"Error reading table '{table_name}': {e}")
        return None

    # Convert year and month to datetime for better time series analysis
    if 'year' in df.columns and 'month' in df.columns:
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    else:
        print("Missing 'year' or 'month' columns.")
        return df

    # Add season information
    df['season'] = pd.cut(df['month'], bins=[0, 3, 6, 9, 12], 
                          labels=['Winter', 'Spring', 'Summer', 'Fall'], right=True)

    return df

def main():
    # Load data
    df = load_data()
    
    # Header
    st.markdown('<h1 class="main-header">🌊 Oceanography Data Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Country filter
    countries = ['All'] + sorted(df['Country'].unique().tolist())
    selected_country = st.sidebar.selectbox("Select Country", countries)
    
    # Year range filter
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(int(df['year'].min()), int(df['year'].max()))
    )
    
    # Parameter selection
    parameters = ['sst', 'chlor_a', 'Kd_490', 'poc', 'pic', 'aot_862']
    selected_params = st.sidebar.multiselect(
        "Select Parameters to Analyze",
        parameters,
        default=['sst', 'chlor_a', 'Kd_490']
    )
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_country != 'All':
        filtered_df = filtered_df[filtered_df['Country'] == selected_country]
    
    filtered_df = filtered_df[
        (filtered_df['year'] >= year_range[0]) & 
        (filtered_df['year'] <= year_range[1])
    ]
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", "📈 Trends", "🌱 Carbon Dynamics", "🗺️ Maps", 
        "🔗 Correlations", "📋 Insights", "🤖 Natural Language"
    ])
    
    with tab1:
        show_overview_tab(filtered_df, df)
    
    with tab2:
        show_trends_tab(filtered_df, selected_params)
    
    with tab3:
        show_carbon_dynamics_tab(filtered_df)
    
    with tab4:
        show_maps_tab(filtered_df)
    
    with tab5:
        show_correlations_tab(filtered_df)
    
    with tab6:
        show_insights_tab(filtered_df)
    
    with tab7:
        show_chatbot_tab()

def show_overview_tab(filtered_df, full_df):
    """Overview tab with key metrics and summary statistics"""
    st.header("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Observations", f"{len(filtered_df):,}")
    
    with col2:
        st.metric("Countries", len(filtered_df['Country'].unique()))
    
    with col3:
        st.metric("Date Range", f"{filtered_df['year'].min()} - {filtered_df['year'].max()}")
    
    with col4:
        st.metric("Avg SST (°C)", f"{filtered_df['sst'].mean():.2f}")
    
    # Oceanographic insights
    st.markdown('<div class="ocean-insight">', unsafe_allow_html=True)
    st.markdown("**🌊 Oceanographic Context:**")
    st.markdown("""
    - **SST (Sea Surface Temperature):** Critical for understanding ocean heat content and climate patterns
    - **Chlorophyll-a (chlor_a):** Primary indicator of phytoplankton biomass and ocean productivity
    - **Kd_490 (Diffuse Attenuation Coefficient):** Measures water clarity and light penetration
    - **POC (Particulate Organic Carbon):** Important for carbon cycling and marine food webs
    - **PIC (Particulate Inorganic Carbon):** Related to calcifying organisms and ocean acidification
    - **AOT_862 (Aerosol Optical Thickness):** Atmospheric conditions affecting ocean observations
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary statistics by country
    st.subheader("📈 Country-Level Summary Statistics")
    
    country_stats = filtered_df.groupby('Country').agg({
        'sst': ['mean', 'std', 'min', 'max'],
        'chlor_a': ['mean', 'std'],
        'Kd_490': ['mean', 'std']
    }).round(3)
    
    st.dataframe(country_stats, use_container_width=True)
    
    # Boxplots by country
    st.subheader("📊 Parameter Distribution by Country")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sea Surface Temperature (°C)', 'Chlorophyll-a (mg/m³)', 
                       'Diffuse Attenuation (Kd_490)', 'Particulate Organic Carbon (mg/m³)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # SST boxplot
    for country in filtered_df['Country'].unique():
        country_data = filtered_df[filtered_df['Country'] == country]['sst']
        fig.add_trace(
            go.Box(y=country_data, name=country, boxpoints='outliers'),
            row=1, col=1
        )
    
    # Chlorophyll-a boxplot
    for country in filtered_df['Country'].unique():
        country_data = filtered_df[filtered_df['Country'] == country]['chlor_a']
        fig.add_trace(
            go.Box(y=country_data, name=country, boxpoints='outliers'),
            row=1, col=2
        )
    
    # Kd_490 boxplot
    for country in filtered_df['Country'].unique():
        country_data = filtered_df[filtered_df['Country'] == country]['Kd_490']
        fig.add_trace(
            go.Box(y=country_data, name=country, boxpoints='outliers'),
            row=2, col=1
        )
    
    # POC boxplot
    for country in filtered_df['Country'].unique():
        country_data = filtered_df[filtered_df['Country'] == country]['poc']
        fig.add_trace(
            go.Box(y=country_data, name=country, boxpoints='outliers'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_trends_tab(filtered_df, selected_params):
    """Trends tab with time series analysis"""
    st.header("📈 Temporal Trends Analysis")
    
    # Check if parameters are selected
    if not selected_params:
        st.warning("⚠️ Please select at least one parameter from the sidebar to view trends.")
        return
    
    # Time series plots
    st.subheader("🕒 Time Series Analysis")
    
    # Monthly trends
    monthly_avg = filtered_df.groupby(['year', 'month']).agg({
        'sst': 'mean',
        'chlor_a': 'mean',
        'Kd_490': 'mean',
        'poc': 'mean',
        'pic': 'mean'
    }).reset_index()
    
    monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))
    
    fig = make_subplots(
        rows=len(selected_params), cols=1,
        subplot_titles=[f"{param.upper()} Over Time" for param in selected_params],
        vertical_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, param in enumerate(selected_params, 1):
        fig.add_trace(
            go.Scatter(
                x=monthly_avg['date'],
                y=monthly_avg[param],
                mode='lines+markers',
                name=param.upper(),
                line=dict(color=colors[i-1])
            ),
            row=i, col=1
        )
        
        # Add trend line
        z = np.polyfit(range(len(monthly_avg)), monthly_avg[param], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=monthly_avg['date'],
                y=p(range(len(monthly_avg))),
                mode='lines',
                name=f'{param.upper()} Trend',
                line=dict(color=colors[i-1], dash='dash')
            ),
            row=i, col=1
        )
    
    fig.update_layout(height=300*len(selected_params), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal analysis
    st.subheader("🌍 Seasonal Patterns")
    
    seasonal_avg = filtered_df.groupby('season').agg({
        'sst': 'mean',
        'chlor_a': 'mean',
        'Kd_490': 'mean'
    }).reset_index()
    
    fig = px.bar(
        seasonal_avg,
        x='season',
        y=['sst', 'chlor_a', 'Kd_490'],
        title="Seasonal Averages",
        barmode='group'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Yearly trends
    st.subheader("📅 Yearly Trends")
    
    yearly_avg = filtered_df.groupby('year').agg({
        'sst': 'mean',
        'chlor_a': 'mean',
        'Kd_490': 'mean'
    }).reset_index()
    
    fig = px.line(
        yearly_avg,
        x='year',
        y=['sst', 'chlor_a', 'Kd_490'],
        title="Yearly Averages",
        markers=True
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def show_carbon_dynamics_tab(filtered_df):
    """Carbon dynamics tab focusing on POC and PIC relationships"""
    st.header("🌱 Carbon Dynamics Analysis")
    
    st.markdown('<div class="ocean-insight">', unsafe_allow_html=True)
    st.markdown("**🌱 Carbon Cycle Context:**")
    st.markdown("""
    - **POC (Particulate Organic Carbon):** Represents organic matter in the ocean, crucial for marine food webs
    - **PIC (Particulate Inorganic Carbon):** Associated with calcifying organisms like coccolithophores and foraminifera
    - The POC:PIC ratio indicates the balance between organic and inorganic carbon production
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # POC vs PIC scatter plot
    st.subheader("📊 POC vs PIC Relationship")
    
    fig = px.scatter(
        filtered_df,
        x='poc',
        y='pic',
        color='Country',
        size='sst',
        hover_data=['year', 'month'],
        title="POC vs PIC with SST as size and Country as color"
    )
    
    # Add trend line
    z = np.polyfit(filtered_df['poc'], filtered_df['pic'], 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=filtered_df['poc'],
            y=p(filtered_df['poc']),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # POC:PIC ratio analysis
    st.subheader("⚖️ POC:PIC Ratio Analysis")
    
    filtered_df['poc_pic_ratio'] = filtered_df['poc'] / filtered_df['pic']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            filtered_df,
            x='poc_pic_ratio',
            nbins=30,
            title="Distribution of POC:PIC Ratio"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        ratio_by_country = filtered_df.groupby('Country')['poc_pic_ratio'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=ratio_by_country.index,
            y=ratio_by_country.values,
            title="Average POC:PIC Ratio by Country"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_maps_tab(filtered_df):
    """Maps tab with geospatial visualizations"""
    st.header("🗺️ Geospatial Analysis")
    
    # SST heatmap
    st.subheader("🌡️ Sea Surface Temperature Heatmap")
    
    # Create a sample for better visualization (too many points can be slow)
    sample_df = filtered_df.sample(min(1000, len(filtered_df)))
    
    fig = px.density_mapbox(
        sample_df,
        lat='lat',
        lon='lon',
        z='sst',
        radius=20,
        center=dict(lat=0, lon=0),
        zoom=1,
        mapbox_style="carto-positron",
        title="Sea Surface Temperature Distribution"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Chlorophyll-a heatmap
    st.subheader("🌿 Chlorophyll-a Distribution")
    
    fig = px.density_mapbox(
        sample_df,
        lat='lat',
        lon='lon',
        z='chlor_a',
        radius=20,
        center=dict(lat=0, lon=0),
        zoom=1,
        mapbox_style="carto-positron",
        title="Chlorophyll-a Distribution"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive map with multiple parameters
    st.subheader("🎯 Interactive Parameter Map")
    
    param_choice = st.selectbox(
        "Select parameter to visualize:",
        ['sst', 'chlor_a', 'Kd_490', 'poc', 'pic', 'aot_862']
    )
    
    fig = px.scatter_mapbox(
        sample_df,
        lat='lat',
        lon='lon',
        color=param_choice,
        size=param_choice,
        hover_data=['Country', 'year', 'month'],
        zoom=1,
        mapbox_style="carto-positron",
        title=f"{param_choice.upper()} Distribution"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_correlations_tab(filtered_df):
    """Correlations tab with correlation analysis"""
    st.header("🔗 Correlation Analysis")
    
    # Select numeric columns for correlation
    numeric_cols = ['sst', 'poc', 'pic', 'aot_862', 'chlor_a', 'Kd_490']
    
    # Calculate correlation matrix
    corr_matrix = filtered_df[numeric_cols].corr()
    
    # Create correlation heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Oceanographic Parameters",
        color_continuous_scale='RdBu_r'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed correlation analysis
    st.subheader("📊 Detailed Correlation Analysis")
    
    # Create correlation pairs
    param_pairs = [
        ('sst', 'chlor_a'),
        ('sst', 'Kd_490'),
        ('chlor_a', 'Kd_490'),
        ('poc', 'pic'),
        ('poc', 'chlor_a'),
        ('pic', 'chlor_a')
    ]
    
    cols = st.columns(2)
    
    for i, (param1, param2) in enumerate(param_pairs):
        with cols[i % 2]:
            fig = px.scatter(
                filtered_df,
                x=param1,
                y=param2,
                color='Country',
                title=f"{param1.upper()} vs {param2.upper()}",
                trendline="ols"
            )
            
            # Calculate correlation coefficient
            corr = filtered_df[param1].corr(filtered_df[param2])
            fig.add_annotation(
                x=0.05, y=0.95,
                xref='paper', yref='paper',
                text=f'Correlation: {corr:.3f}',
                showarrow=False,
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_insights_tab(filtered_df):
    """Insights tab with oceanographic interpretations"""
    st.header("📋 Oceanographic Insights")
    
    # Key findings
    st.subheader("🔍 Key Findings")
    
    # Calculate key statistics
    sst_mean = filtered_df['sst'].mean()
    sst_std = filtered_df['sst'].std()
    chlor_mean = filtered_df['chlor_a'].mean()
    chlor_std = filtered_df['chlor_a'].std()
    
    # SST insights
    st.markdown('<div class="ocean-insight">', unsafe_allow_html=True)
    st.markdown("**🌡️ Sea Surface Temperature Insights:**")
    st.markdown(f"""
    - **Average SST:** {sst_mean:.2f}°C ± {sst_std:.2f}°C
    - **Range:** {filtered_df['sst'].min():.1f}°C to {filtered_df['sst'].max():.1f}°C
    - **Seasonal Variation:** {filtered_df.groupby('season')['sst'].std().mean():.2f}°C
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chlorophyll insights
    st.markdown('<div class="ocean-insight">', unsafe_allow_html=True)
    st.markdown("**🌿 Chlorophyll-a Insights:**")
    st.markdown(f"""
    - **Average Chlorophyll-a:** {chlor_mean:.2f} mg/m³ ± {chlor_std:.2f} mg/m³
    - **Range:** {filtered_df['chlor_a'].min():.2f} to {filtered_df['chlor_a'].max():.2f} mg/m³
    - **Productivity Indicator:** Values suggest {'high' if chlor_mean > 1.0 else 'moderate' if chlor_mean > 0.5 else 'low'} primary productivity
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Water clarity insights
    st.markdown('<div class="ocean-insight">', unsafe_allow_html=True)
    st.markdown("**💧 Water Clarity Insights (Kd_490):**")
    kd_mean = filtered_df['Kd_490'].mean()
    st.markdown(f"""
    - **Average Kd_490:** {kd_mean:.3f}
    - **Water Clarity:** {'Clear' if kd_mean < 0.1 else 'Moderate' if kd_mean < 0.3 else 'Turbid'} water conditions
    - **Light Penetration:** {'High' if kd_mean < 0.1 else 'Moderate' if kd_mean < 0.3 else 'Low'} light penetration
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Carbon cycle insights
    st.markdown('<div class="ocean-insight">', unsafe_allow_html=True)
    st.markdown("**🌱 Carbon Cycle Insights:**")
    poc_mean = filtered_df['poc'].mean()
    pic_mean = filtered_df['pic'].mean()
    poc_pic_ratio = poc_mean / pic_mean if pic_mean > 0 else 0
    
    st.markdown(f"""
    - **Average POC:** {poc_mean:.2f} mg/m³
    - **Average PIC:** {pic_mean:.2f} mg/m³
    - **POC:PIC Ratio:** {poc_pic_ratio:.2f}
    - **Carbon Balance:** {'Organic carbon dominant' if poc_pic_ratio > 1.5 else 'Balanced' if poc_pic_ratio > 0.5 else 'Inorganic carbon dominant'}
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Regional patterns
    st.subheader("🌍 Regional Patterns")
    
    regional_analysis = filtered_df.groupby('Country').agg({
        'sst': 'mean',
        'chlor_a': 'mean',
        'Kd_490': 'mean',
        'poc': 'mean',
        'pic': 'mean'
    }).round(3)
    
    st.dataframe(regional_analysis, use_container_width=True)

def show_chatbot_tab():
    """Natural Language chatbot tab integrated with the backend"""
    st.header("🤖 Natural Language Query Interface")
    
    st.markdown('<div class="ocean-insight">', unsafe_allow_html=True)
    st.markdown("**🌊 Ask questions about oceanographic data in natural language:**")
    st.markdown("""
    - Ask about specific locations: "What's the SST data for Miami?"
    - Use coordinates: "Show me data for 25.7617, -80.1918"
    - General queries: "Tell me about ocean temperature near Australia"
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Backend URL configuration
    FLASK_BACKEND_URL = "http://localhost:5000/api/chat"
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about oceanographic data...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message immediately
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        
        # Get response from backend
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing oceanographic data..."):
                try:
                    response = requests.post(
                        FLASK_BACKEND_URL, 
                        json={"prompt": user_input},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        bot_response = response.json().get("response", "No response received")
                    else:
                        bot_response = f"❌ Backend error: {response.status_code}"
                        
                except requests.exceptions.ConnectionError:
                    bot_response = """
                    ❌ **Backend Connection Error**
                    
                    The chatbot backend is not running. To use the natural language interface:
                    
                    1. **Start the backend server** by running:
                       ```bash
                       python app.py
                       ```
                    
                    2. **Or use the auto-launcher**:
                       ```bash
                       python run.py
                       ```
                    
                    3. Make sure the backend is running on `http://localhost:5000`
                    
                    Once the backend is running, you can ask questions like:
                    - "What's the SST data for Miami?"
                    - "Show me data for coordinates 25.7617, -80.1918"
                    - "Tell me about ocean temperature near Australia"
                    """
                except requests.exceptions.Timeout:
                    bot_response = "⏱️ Request timed out. The backend might be processing a large query."
                except Exception as e:
                    bot_response = f"❌ Error: {str(e)}"
            
            st.markdown(bot_response)
            
            # Add bot response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Backend status check
    st.subheader("🔧 Backend Status")
    
    if st.button("🔍 Check Backend Status"):
        try:
            response = requests.get("http://localhost:5000", timeout=5)
            if response.status_code < 500:
                st.success("✅ Backend is running and accessible!")
            else:
                st.error(f"❌ Backend returned status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Backend is not running. Please start the Flask backend first.")
        except Exception as e:
            st.error(f"❌ Error checking backend: {str(e)}")
    
    # Instructions
    with st.expander("📖 How to use the Natural Language Interface"):
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Start the Backend**: Make sure the Flask backend (`app.py`) is running
        2. **Ask Questions**: Type natural language questions about oceanographic data
        3. **Get Insights**: The AI will analyze your query and provide relevant data
        
        ### 📝 Example Queries
        
        - **Location-based**: "What's the sea surface temperature in Miami?"
        - **Coordinate-based**: "Show me data for 25.7617, -80.1918"
        - **Regional**: "Tell me about chlorophyll levels near Australia"
        - **Comparative**: "Compare SST between different locations"
        
        ### 🔧 Backend Setup
        
        If the backend isn't running, use one of these methods:
        
        **Method 1 - Direct:**
        ```bash
        python app.py
        ```
        
        **Method 2 - Auto-launcher:**
        ```bash
        python run.py
        ```
        
        The backend will be available at `http://localhost:5000`
        """)

if __name__ == "__main__":
    main()