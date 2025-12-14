"""
Interactive Visualization Dashboard
LO3 Implementation: Tools to view and control simulations and results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

# Set styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class InvestmentVisualizationDashboard:
    """
    Interactive dashboard for visualizing investment analysis results
    """
    
    def __init__(self, data_path='../data/cleaned_investment_data.csv'):
        """
        Initialize the visualization dashboard
        
        Args:
            data_path (str): Path to cleaned data
        """
        self.data_path = data_path
        self.df = None
        self.cluster_labels = None
        self.predictions = None
        
    def load_data(self):
        """Load and prepare data for visualization"""
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def create_correlation_heatmap(self):
        """
        Create interactive correlation heatmap
        
        Returns:
            plotly.graph_objects.Figure: Interactive heatmap
        """
        print("Creating correlation heatmap...")
        
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        # Create interactive heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Correlation Heatmap of Investment Variables',
            labels=dict(color='Correlation')
        )
        
        # Update layout
        fig.update_layout(
            width=800,
            height=700,
            title_x=0.5,
            title_font=dict(size=20, family='Arial, sans-serif'),
            font=dict(size=12),
            coloraxis_colorbar=dict(
                title='Correlation',
                thickness=20,
                len=0.75
            )
        )
        
        # Add annotations
        fig.update_traces(
            hovertemplate='<br>'.join([
                'X: %{x}',
                'Y: %{y}',
                'Correlation: %{z:.3f}'
            ])
        )
        
        fig.write_html('../output/correlation_heatmap.html')
        print("Correlation heatmap saved to HTML file")
        
        return fig
    
    def create_distribution_plots(self):
        """
        Create distribution plots for key variables
        
        Returns:
            plotly.graph_objects.Figure: Distribution plots
        """
        print("Creating distribution plots...")
        
        # Select key investment variables
        investment_vars = ['Mutual_Funds', 'Equity_Market', 'Debentures', 
                          'Government_Bonds', 'Fixed_Deposits', 'PPF', 'Gold']
        
        # Filter to available columns
        available_vars = [var for var in investment_vars if var in self.df.columns]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=available_vars,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Add distribution plots
        for i, var in enumerate(available_vars):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            fig.add_trace(
                go.Histogram(
                    x=self.df[var],
                    name=var,
                    nbinsx=20,
                    marker_color=px.colors.qualitative.Set3[i % 12],
                    opacity=0.7,
                    hovertemplate=f'{var}<br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add mean line
            mean_val = self.df[var].mean()
            fig.add_vline(
                x=mean_val, 
                line_dash='dash', 
                line_color='red',
                row=row, col=col
            )
            
            # Update axes
            fig.update_xaxes(title_text='Score (1-7)', row=row, col=col)
            fig.update_yaxes(title_text='Count', row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title_text='Distribution of Investment Preferences',
            title_x=0.5,
            title_font=dict(size=24, family='Arial, sans-serif'),
            showlegend=False,
            height=900,
            width=1200,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        fig.write_html('../output/distribution_plots.html')
        print("Distribution plots saved to HTML file")
        
        return fig
    
    def create_age_distribution(self):
        """
        Create age distribution visualization
        
        Returns:
            plotly.graph_objects.Figure: Age distribution plot
        """
        if 'Age' not in self.df.columns:
            print("Age column not found in data")
            return None
        
        print("Creating age distribution plot...")
        
        fig = px.histogram(
            self.df,
            x='Age',
            nbins=15,
            title='Age Distribution of Investors',
            labels={'Age': 'Age (years)', 'count': 'Number of Investors'},
            color_discrete_sequence=['#636EFA'],
            opacity=0.7
        )
        
        # Add KDE curve
        from scipy.stats import gaussian_kde
        
        ages = self.df['Age'].values
        kde = gaussian_kde(ages)
        x_range = np.linspace(ages.min(), ages.max(), 100)
        y_kde = kde(x_range)
        
        # Scale KDE to match histogram
        hist_counts, _ = np.histogram(ages, bins=15)
        scale_factor = hist_counts.max() / y_kde.max()
        y_kde_scaled = y_kde * scale_factor
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_kde_scaled,
                mode='lines',
                name
