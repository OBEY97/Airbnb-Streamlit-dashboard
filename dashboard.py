import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import List
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title='London Airbnb', page_icon='🏙️ 📊', layout='wide')

st.title('🏙️ 📊 London Airbnb Analysis')
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)


fl = st.file_uploader(':file_folder: Upload a file', type=(['csv', 'txt', 'xlsx', 'xls']))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename)
else:
    #os.chdir(r'/Users/capt.lilo/Desktop/London-Airbnb-project/STREAMLIT')
    df = pd.read_csv(r'data/Cleaned-London-Airbnb-Dataset.csv')

st.header('Univariate Analysis')

col1, col2, col3 = st.columns(3)
#Bar Chart
with col1:
    usable_cols = ['room_type', 'superhost', 'guests', 'bedrooms', 'registration', 
                     'instant_book', 'min_nights', 'baths', 'beds']

    selected_col = st.selectbox('👇 Select your column', options=[c for c in usable_cols if c in df.columns], key='col1')
    
    if selected_col:
        df_filtered= df[selected_col].value_counts().reset_index()
        df_filtered.columns = [f'{selected_col}', 'Count']
    fig1 = px.bar(
        df_filtered,
        x=f'{selected_col}',
        y='Count',
        title=f'{selected_col} Distribution',
        color=f'{selected_col}',
        color_discrete_sequence=px.colors.qualitative.Prism,
        height=400, 
        #template='seaborn'
    )
    fig1.update_traces(textposition='outside')
    fig1.update_layout(xaxis_title=selected_col, yaxis_title='Count')

    st.plotly_chart(fig1, use_container_width=True)

    with st.expander(f'{selected_col} Data'):
        var = df[selected_col].value_counts().reset_index()
        st.write(var.style.background_gradient(cmap='Blues'))
        csv = var.to_csv(index=False).encode('utf-8')
        st.download_button('Download Data', data=csv, file_name=f'{selected_col}.csv', mime='text/csv',
                           help= 'Click here to download the data as a CSV file')

# Pie chart
with col2:
    selected_col = st.selectbox('👇 Select your column', options=[c for c in usable_cols if c in df.columns], key="col2")
        
    if selected_col:
            df_filtered= df[selected_col].value_counts().reset_index()
            df_filtered.columns = [f'{selected_col}', 'Count']
    fig1 = px.pie(
            df_filtered,
            names=f'{selected_col}',
            values='Count',
            title=f'{selected_col} Distribution',
            color=f'{selected_col}',
            #color_discrete_sequence=px.colors.qualitative.Prism,
            hole=0.3,
            height=400, 
            template='seaborn'
        )
    fig1.update_traces(textposition='outside')
        #fig1.update_layout(xaxis_title=selected_col, yaxis_title='Count')

    st.plotly_chart(fig1, use_container_width=True)

    with st.expander(f'{selected_col} Data'):
        var = df[selected_col].value_counts().reset_index()
        st.write(var.style.background_gradient(cmap='Reds'))
        csv = var.to_csv(index=False).encode('utf-8')
        st.download_button('Download Data', data=csv, file_name=f'{selected_col}.csv', mime='text/csv',
                           key='db2',
                        help= 'Click here to download the data as a CSV file')



#Histogram    
with col3:
    hist_cols = ['ttm_revenue', 'ttm_reserved_days', 'ttm_avg_rate_native', 
                 'cleaning_fee', 'extra_guest_fee']
    
    selected_col= st.selectbox('👇 Select your column', options=[c for c in hist_cols if c in df.columns], key='col3')
    if selected_col:
        df_filtered= df[selected_col].value_counts().reset_index()
        df_filtered.columns = [f'{selected_col}', 'Frequency']
    fig2 = px.histogram(
        df_filtered, 
        x=f'{selected_col}', 
        y='Frequency',
        nbins=30, 
        #color_discrete_sequence=['#007BFF'], 
        color_discrete_sequence=px.colors.qualitative.Vivid,
        height=400, 
        title=f'{selected_col} Distribution'
    )
    fig2.update_layout(xaxis_title=f'{selected_col}', yaxis_title='Frequency')

    st.plotly_chart(fig2, use_container_width=True)

    with st.expander(f'{selected_col} Data'):
        var = df[selected_col].value_counts().reset_index()
        st.write(var.style.background_gradient(cmap='Oranges'))
        csv = var.to_csv(index=False).encode('utf-8')
        st.download_button('Download Data', data=csv, file_name=f'{selected_col}.csv', mime='text/csv',
                           help='Click here to download the data as a CSV file')


# ------ HELPERS - Filter Functions -------

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get categorical columns from DataFrame."""
    l1 = [col for col in df.columns if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col])
            and col not in ['listing_id', 'listingname', 'host_id', 'host_name']]
    l1 += ['bedrooms', 'superhost', 'baths', 'beds', 'ttm_avg_rate', 'ttm_revenue', 'guests']
    return l1

def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Get numerical columns from DataFrame."""
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

def limit_categories(series: pd.Series, max_categories: int) -> pd.Series:
    """Return series but collapse least frequent categories into 'Other' if > max_categories."""
    if series.nunique() > max_categories:
        top_categories = series.value_counts().nlargest(max_categories).index
        return series.where(series.isin(top_categories), other='Other')
    return series

# ------- Sidebar Filters -------
st.sidebar.header('Filters & Plot controls')

# Filter by one Column
filter_cols = st.sidebar.multiselect(
    'Select categorical Column (optional)', 
    options= get_categorical_columns(df),
    default=None
    )

selected_values = None
if filter_cols:
    selected_values = st.sidebar.multiselect(
        f"Select your Category from {filter_cols}", 
        options=df[filter_cols].dropna().unique(),
        default=None
    )

# Choose X (categorical or numerical) and Y (numerical) columns
st.markdown('### Plot axes')
x_col = st.sidebar.selectbox('X-axis (feature):', options=df.columns.tolist(), index=3)
y_col = st.sidebar.selectbox('Y-axis (numerical):', options=get_numerical_columns(df), index=7)

# Optional second column (Hue / Group)
second_col = st.sidebar.selectbox(
    'Optional: Select second Column for grouping (Hue)', 
    options=[None] + df.columns.tolist()
    )

# Plot Type options - adapt to column types
plot_type = st.sidebar.selectbox(
    'Plot Type:', 
    options=[
        'Auto (Choose best)', 
        'Bar Plot', 
        'Box Plot', 
        'Violin Plot', 
        'Scatter Plot', 
        'Line Plot', 
        'Heatmap'
    ]
)

agg_func = st.sidebar.selectbox(
    'Aggregation Function for aggregated plots:', 
    options=['Count', 'Mean', 'Sum', 'Median']
)
max_categories = st.sidebar.slider(
    'Max Categories for to show:', 
    min_value=3, 
    max_value=50, 
    value=12
)

# ------- Apply Filters -------
if x_col == y_col:
    st.warning('X and Y are the same plot may not be meaningful.')

if filter_cols and selected_values:
    df_filtered = df[df[filter_cols].isin(selected_values)].copy()
else:
    df_filtered = df.copy()

# Limit categories if necessary
if x_col in get_categorical_columns(df_filtered):
    df_filtered[x_col] = limit_categories(df_filtered[x_col].fillna('Missing'), max_categories)

# If second column is categorical and user select max_categories, collapse
if second_col and second_col in get_categorical_columns(df_filtered):
    df_filtered[second_col] = limit_categories(df_filtered[second_col].fillna('Missing'), max_categories)


#------- Plotting Logic---------
def aggregated_bar_plot(df, x, y, hue=None, agg_func='mean'):
    agg_lower = agg_func.lower()

    if agg_lower == 'count':
        y_label = 'Count'
        title_base = f' Row Count by {x}'
    else:
        y_label = f'{agg_lower} of {y}'
        title_base = f'{agg_lower} of {y} by {x}'
    if hue and (hue in get_categorical_columns(df)):
        grouped = df.groupby([x, hue])[y].agg(agg_lower).reset_index()
        fig = px.bar(grouped, x=x, y=y, color=hue, barmode='group',
                     title= title_base + f' and {hue}', 
                     labels= {y: y_label, x: x, 'color': hue})
    else:
        grouped = df.groupby(x)[y].agg(agg_lower).reset_index().sort_values(by=y, ascending=False)
        fig = px.bar(grouped, x=x, y=y, 
                     title= title_base,
                     labels= {y: y_label, x: x})
        
    return fig

def box_plot(df, x, y, hue=None):
    df_plot = df.copy()
    x_original = x
#Bin numeric x or if too many unique values
    if pd.api.types.is_numeric_dtype(df_plot[x]) and df_plot[x].nunique() > 20:
        df_plot['_x_binned'] = pd.cut(df_plot[x], bins=10, include_lowest=True)
        x = '_x_binned'

    title = f'Box Plot of {y} by {x_original}'
    if hue:
        title += f' and {hue}'
        fig = px.box(df_plot, x=x, y=y, color=hue, 
                     title=title,
                     labels={y: y, x: x_original, 'color': hue if hue else ''})
    else:
        fig = px.box(df_plot, x=x, y=y, 
                     title= title,
                     labels={y: y, x: x_original})
    return fig

def violin_plot(df, x, y, color=None):
    df_plot = df.copy()
    x_original = x
#Bin numeric x or if too many unique values
    if pd.api.types.is_numeric_dtype(df_plot[x]) and df_plot[x].nunique() > 20:
        df_plot['_x_binned'] = pd.cut(df_plot[x], bins=10, include_lowest=True)
        x = '_x_binned'

    title = f'Violin Plot of {y} by {x_original}'
    if color:
        title += f' and {color}'
    fig = px.violin(df_plot, x=x, y=y, color=color, box=True, 
                    title= title,
                    labels={y: y, x: x_original, color: color if color else ''})
    return fig

def scatter_plot(df, x, y, color=None):
    title = f'Scatter Plot of {y} vs {x}'
    if color:
        title += f' colored by {color}'
    fig = px.scatter(df, x=x, y=y, color=color, hover_data=df.columns.tolist(),
                     title= title,
                     labels={y: y, x: x, color: color if color else ''})
    return fig

def line_plot(df, x, y, color=None):
    df_plot = df.sort_values(by=x)
    title = f'Line Plot of {y} vs {x}'
    if color:
        title += f' colored by {color}'
    fig = px.line(df_plot, x=x, y=y, color=color, 
                  title= title,
                  labels={y: y, x: x, color: color if color else ''})
    return fig

def heatmap(df, x, y, hue=None, agg_func='Mean'):
    agg_lower = agg_func.lower()  # Assuming agg_func is passed or global; better to pass as param if refactoring
    
    # Customize title based on agg
    if agg_lower == 'count':
        title = f'Heatmap of Row Count by {x}' + (f' and {hue}' if hue else '')
        color_label = 'Row Count'
    else:
        title = f'Heatmap of {agg_func} of {y} by {x}' + (f' and {hue}' if hue else '')
        color_label = f'{agg_func} ({y})'

    if hue:
        df.pivot_table(values=y, index=x, columns=hue, aggfunc=agg_lower)
    else:
        pivot_table = df.pivot_table(values=y, index=x, aggfunc=agg_lower)
    pivot_table = pivot_table.fillna(0)

    if pd.api.types.is_numeric_dtype(df[x]) and df[x].nunique() > 20:
        df['_x_binned'] = pd.cut(df[x], bins=10)
        pivot_table = df.pivot_table(values=y, index='_x_binned', columns=hue if hue else None, aggfunc=agg_lower).fillna(0)
        
    fig = px.imshow(pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    title=title,
                    labels={ 'x': hue if hue else y, 'y': x, 'color': color_label})
    return fig

#Auto plot type selection
def auto_plot(df, x, y, second, plot_type, agg_func):

    x_is_num = pd.api.types.is_numeric_dtype(df[x])
    second_cat = second and (pd.api.types.is_categorical_dtype(df[second]) or pd.api.types.is_object_dtype(df[second]))

    ptype = plot_type
    if plot_type == 'Auto (Choose best)':
        if x_is_num and pd.api.types.is_numeric_dtype(df[y]):
            ptype = 'Scatter Plot'
        elif not x_is_num and pd.api.types.is_numeric_dtype(df[y]):
            ptype = 'Box Plot'
        else:
            ptype = 'Bar Plot'  # Fallback
        
    if ptype == 'Bar Plot':
        return aggregated_bar_plot(df, x, y, hue=second if second_cat else None, agg_func=agg_func.lower())
    elif ptype == 'Box Plot':
        return box_plot(df, x, y, hue=second if second_cat else None)
    elif ptype == 'Scatter Plot':
        if not x_is_num:
            # create a small jitter for better visibility
            df_plot = df.copy()
            df_plot[x] = df_plot[x].astype('category')
            cat_to_num = {cat: i for i, cat in enumerate(df_plot[x].cat.categories)}
            df_plot['_x_num'] = df_plot[x].map(cat_to_num) + np.random.uniform(-0.2, 0.2,len(df_plot))
            fig = scatter_plot(df_plot, '_x_num', y, color=second)
            fig.update_layout(xaxis= {'tickvals': list(cat_to_num.values()), 'ticktext': list(cat_to_num.keys())})
            return fig
        else:
            return scatter_plot(df, x, y, color=second)
        
    elif ptype == 'Line Plot':
        return line_plot(df, x, y, color=second)
    elif ptype == 'Heatmap':
        return heatmap(df, x, y, hue=second if second_cat else None)
    elif ptype == 'Violin Plot':
        return violin_plot(df, x, y, color=second if second_cat else None)
    else:
        return aggregated_bar_plot(df, x, y, hue=second if second_cat else None, agg_func=agg_func.lower())
    

#------- Generate and Show Plot -------
if df[x_col].nunique() > 50:
    st.warning(f'High unique values in {x_col}. The plot may be cluttered or slow to render. Consider filtering.')

if df_filtered.empty:
    st.warning('No data available for the selected filters. Please adjust your filter selections.')
else:
    fig = auto_plot(df_filtered, x_col, y_col, second_col, plot_type, agg_func)
    st.plotly_chart(fig, use_container_width=True)

    # Summary Statistics for selection
    st.markdown('### Summary Statistics for selection')
    with st.expander('View Data'):
        desc = df_filtered.groupby([x_col])[y_col].agg(['count', 'mean', 'median', 'min', 'max', 'std']).sort_values(by='count', ascending=False)
        st.dataframe(desc.style.background_gradient(cmap='Greens'))
