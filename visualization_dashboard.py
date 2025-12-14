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
                name='Density',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            )
        )
        
        # Add statistics
        mean_age = ages.mean()
        median_age = np.median(ages)
        
        fig.add_vline(
            x=mean_age,
            line_dash='dash',
            line_color='green',
            annotation_text=f'Mean: {mean_age:.1f}',
            annotation_position='top right'
        )
        
        fig.add_vline(
            x=median_age,
            line_dash='dot',
            line_color='orange',
            annotation_text=f'Median: {median_age:.1f}',
            annotation_position='top left'
        )
        
        # Update layout
        fig.update_layout(
            width=800,
            height=500,
            title_x=0.5,
            title_font=dict(size=20, family='Arial, sans-serif'),
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            title_font=dict(size=14)
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            title_font=dict(size=14)
        )
        
        fig.write_html('../output/age_distribution.html')
        print("Age distribution plot saved to HTML file")
        
        return fig
    
    def create_investment_radar_chart(self, investor_id=None):
        """
        Create radar chart for investment preferences
        
        Args:
            investor_id: Specific investor to visualize (if None, shows average)
            
        Returns:
            plotly.graph_objects.Figure: Radar chart
        """
        print("Creating investment radar chart...")
        
        investment_vars = ['Mutual_Funds', 'Equity_Market', 'Debentures', 
                          'Government_Bonds', 'Fixed_Deposits', 'PPF', 'Gold']
        
        available_vars = [var for var in investment_vars if var in self.df.columns]
        
        if investor_id is not None and 'customer_id' in self.df.columns:
            # Get specific investor data
            investor_data = self.df[self.df['customer_id'] == investor_id]
            if len(investor_data) == 0:
                print(f"Investor {investor_id} not found. Showing average instead.")
                investor_data = self.df
                title = f'Average Investment Profile (All Investors)'
            else:
                title = f'Investment Profile - Investor {investor_id}'
            values = investor_data[available_vars].mean().values
        else:
            # Use average of all investors
            values = self.df[available_vars].mean().values
            title = 'Average Investment Profile (All Investors)'
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available_vars,
            fill='toself',
            name='Investment Score',
            line_color='blue',
            fillcolor='rgba(0, 0, 255, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 7],
                    tickfont=dict(size=10),
                    gridcolor='lightgray',
                    linecolor='gray'
                ),
                angularaxis=dict(
                    tickfont=dict(size=11),
                    gridcolor='lightgray',
                    linecolor='gray'
                ),
                bgcolor='white'
            ),
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, family='Arial, sans-serif')
            ),
            showlegend=True,
            width=700,
            height=600,
            paper_bgcolor='white'
        )
        
        fig.write_html('../output/investment_radar.html')
        print("Radar chart saved to HTML file")
        
        return fig
    
    def create_cluster_visualization(self, cluster_data_path='../output/segmentation_results.csv'):
        """
        Create visualization for customer clusters
        
        Args:
            cluster_data_path (str): Path to clustering results
            
        Returns:
            plotly.graph_objects.Figure: Cluster visualization
        """
        print("Creating cluster visualization...")
        
        # Load cluster data
        try:
            cluster_df = pd.read_csv(cluster_data_path)
            if 'Cluster' not in cluster_df.columns:
                print("Cluster column not found in data")
                return None
        except FileNotFoundError:
            print(f"Cluster data not found at {cluster_data_path}")
            return None
        
        # Merge with main data if needed
        if 'Cluster' not in self.df.columns:
            self.df = cluster_df
        
        # Create 3D scatter plot of clusters
        fig = px.scatter_3d(
            self.df,
            x='Mutual_Funds' if 'Mutual_Funds' in self.df.columns else self.df.columns[0],
            y='Equity_Market' if 'Equity_Market' in self.df.columns else self.df.columns[1],
            z='Fixed_Deposits' if 'Fixed_Deposits' in self.df.columns else self.df.columns[2],
            color='Cluster',
            title='3D Customer Segmentation',
            labels={
                'Mutual_Funds': 'Mutual Funds',
                'Equity_Market': 'Equity Market',
                'Fixed_Deposits': 'Fixed Deposits',
                'Cluster': 'Customer Cluster'
            },
            opacity=0.7,
            hover_data=['Age'] if 'Age' in self.df.columns else None,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Update layout
        fig.update_layout(
            width=900,
            height=700,
            title_x=0.5,
            title_font=dict(size=20, family='Arial, sans-serif'),
            scene=dict(
                xaxis_title='Mutual Funds Preference',
                yaxis_title='Equity Market Preference',
                zaxis_title='Fixed Deposits Preference',
                bgcolor='white',
                gridcolor='lightgray'
            ),
            legend=dict(
                title='Cluster',
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01
            ),
            paper_bgcolor='white'
        )
        
        fig.write_html('../output/cluster_3d_visualization.html')
        print("Cluster 3D visualization saved to HTML file")
        
        return fig
    
    def create_time_series_analysis(self, date_column='transaction_date'):
        """
        Create time series analysis visualization (if date data exists)
        
        Args:
            date_column (str): Name of date column
            
        Returns:
            plotly.graph_objects.Figure: Time series plot or None
        """
        if date_column not in self.df.columns:
            print(f"Date column '{date_column}' not found. Skipping time series analysis.")
            return None
        
        print("Creating time series analysis...")
        
        # Convert to datetime if needed
        self.df[date_column] = pd.to_datetime(self.df[date_column], errors='coerce')
        
        # Resample by month
        self.df.set_index(date_column, inplace=True)
        
        # Check for transaction amount column
        amount_col = None
        for col in ['transaction_amount', 'amount', 'value']:
            if col in self.df.columns:
                amount_col = col
                break
        
        if amount_col:
            monthly_data = self.df[amount_col].resample('M').agg(['sum', 'mean', 'count'])
            monthly_data.columns = ['Total Amount', 'Average Amount', 'Transaction Count']
        else:
            # Use count of transactions
            monthly_data = self.df.resample('M').size().to_frame('Transaction Count')
        
        # Create time series plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Transaction Volume Over Time', 'Transaction Metrics'),
            vertical_spacing=0.15
        )
        
        # Plot 1: Transaction count
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data['Transaction Count'] if 'Transaction Count' in monthly_data.columns else monthly_data.iloc[:, 0],
                mode='lines+markers',
                name='Transaction Count',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 0, 255, 0.1)'
            ),
            row=1, col=1
        )
        
        # Plot 2: Amount metrics (if available)
        if 'Total Amount' in monthly_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=monthly_data.index,
                    y=monthly_data['Total Amount'],
                    mode='lines',
                    name='Total Amount',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_data.index,
                    y=monthly_data['Average Amount'],
                    mode='lines',
                    name='Average Amount',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text='Time Series Analysis of Investment Activity',
            title_x=0.5,
            title_font=dict(size=22, family='Arial, sans-serif'),
            height=800,
            width=1000,
            showlegend=True,
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(title_text='Date', row=1, col=1)
        fig.update_yaxes(title_text='Transaction Count', row=1, col=1)
        
        if 'Total Amount' in monthly_data.columns:
            fig.update_xaxes(title_text='Date', row=2, col=1)
            fig.update_yaxes(title_text='Amount ($)', row=2, col=1)
        
        fig.write_html('../output/time_series_analysis.html')
        print("Time series analysis saved to HTML file")
        
        # Reset index
        self.df.reset_index(inplace=True)
        
        return fig
    
    def create_interactive_dashboard(self):
        """
        Create an interactive dashboard with widgets
        
        Returns:
            ipywidgets.VBox: Interactive dashboard
        """
        print("Creating interactive dashboard...")
        
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
        
        # Create widgets
        variable_dropdown = widgets.Dropdown(
            options=list(self.df.select_dtypes(include=[np.number]).columns),
            value='Age' if 'Age' in self.df.columns else self.df.select_dtypes(include=[np.number]).columns[0],
            description='Variable:',
            style={'description_width': 'initial'}
        )
        
        plot_type_dropdown = widgets.Dropdown(
            options=['Histogram', 'Box Plot', 'Violin Plot', 'Scatter Plot'],
            value='Histogram',
            description='Plot Type:',
            style={'description_width': 'initial'}
        )
        
        color_by_dropdown = widgets.Dropdown(
            options=['None'] + list(self.df.select_dtypes(include=['object', 'category']).columns),
            value='None',
            description='Color By:',
            style={'description_width': 'initial'}
        )
        
        bins_slider = widgets.IntSlider(
            value=20,
            min=5,
            max=50,
            step=5,
            description='Bins:',
            disabled=plot_type_dropdown.value != 'Histogram',
            style={'description_width': 'initial'}
        )
        
        # Output widget
        output = widgets.Output()
        
        def update_plot(change):
            """Update plot based on widget values"""
            with output:
                output.clear_output(wait=True)
                
                variable = variable_dropdown.value
                plot_type = plot_type_dropdown.value
                color_by = color_by_dropdown.value if color_by_dropdown.value != 'None' else None
                bins = bins_slider.value
                
                # Create plot based on selections
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if plot_type == 'Histogram':
                    if color_by:
                        # Grouped histogram
                        unique_groups = self.df[color_by].unique()
                        for group in unique_groups:
                            group_data = self.df[self.df[color_by] == group][variable]
                            ax.hist(group_data, bins=bins, alpha=0.5, label=str(group))
                        ax.legend(title=color_by)
                    else:
                        # Simple histogram
                        ax.hist(self.df[variable], bins=bins, alpha=0.7, color='blue', edgecolor='black')
                    
                    ax.set_xlabel(variable)
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Distribution of {variable}')
                    
                elif plot_type == 'Box Plot':
                    if color_by:
                        # Grouped box plot
                        box_data = [self.df[self.df[color_by] == group][variable].dropna() 
                                  for group in self.df[color_by].unique()]
                        ax.boxplot(box_data)
                        ax.set_xticklabels(self.df[color_by].unique())
                        ax.set_xlabel(color_by)
                    else:
                        # Single box plot
                        ax.boxplot(self.df[variable].dropna())
                    
                    ax.set_ylabel(variable)
                    ax.set_title(f'Box Plot of {variable}')
                    
                elif plot_type == 'Violin Plot':
                    if color_by:
                        # Grouped violin plot
                        import seaborn as sns
                        sns.violinplot(data=self.df, x=color_by, y=variable, ax=ax)
                        ax.set_xlabel(color_by)
                    else:
                        # Single violin plot
                        import seaborn as sns
                        sns.violinplot(y=self.df[variable], ax=ax)
                    
                    ax.set_ylabel(variable)
                    ax.set_title(f'Violin Plot of {variable}')
                    
                elif plot_type == 'Scatter Plot' and 'Age' in self.df.columns:
                    # Scatter plot with Age on x-axis
                    if color_by:
                        scatter = ax.scatter(self.df['Age'], self.df[variable], 
                                           c=self.df[color_by].astype('category').cat.codes, 
                                           cmap='viridis', alpha=0.6)
                        plt.colorbar(scatter, ax=ax, label=color_by)
                    else:
                        ax.scatter(self.df['Age'], self.df[variable], alpha=0.6)
                    
                    ax.set_xlabel('Age')
                    ax.set_ylabel(variable)
                    ax.set_title(f'{variable} vs Age')
                
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
        
        def update_bins_disabled(change):
            """Update bins slider disabled state"""
            bins_slider.disabled = (plot_type_dropdown.value != 'Histogram')
        
        # Link widgets to update functions
        variable_dropdown.observe(update_plot, names='value')
        plot_type_dropdown.observe(update_plot, names='value')
        plot_type_dropdown.observe(update_bins_disabled, names='value')
        color_by_dropdown.observe(update_plot, names='value')
        bins_slider.observe(update_plot, names='value')
        
        # Create initial plot
        update_plot(None)
        
        # Create dashboard layout
        controls = widgets.VBox([
            variable_dropdown,
            plot_type_dropdown,
            color_by_dropdown,
            bins_slider
        ])
        
        dashboard = widgets.VBox([
            widgets.HTML("<h2>Interactive Investment Analysis Dashboard</h2>"),
            widgets.HBox([controls, output])
        ])
        
        # Save dashboard HTML
        self._save_dashboard_html(dashboard)
        
        return dashboard
    
    def _save_dashboard_html(self, dashboard):
        """Save dashboard as HTML file"""
        import base64
        from io import BytesIO
        
        # Create a comprehensive static dashboard
        fig = self.create_correlation_heatmap()
        
        # Create dashboard HTML
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Investment Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    text-align: center;
                    padding: 20px;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }
                .plots-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                    gap: 20px;
                }
                .plot-card {
                    background-color: white;
                    padding: 15px;
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                h2 {
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }
                .description {
                    color: #666;
                    font-size: 14px;
                    margin-bottom: 15px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Digital Transformation: Investment Analysis Dashboard</h1>
                <p>Interactive visualization of investment preferences and behaviors</p>
            </div>
            
            <div class="plots-container">
                <div class="plot-card">
                    <h2>Correlation Analysis</h2>
                    <p class="description">Heatmap showing correlations between different investment variables</p>
                    <div id="correlation-plot"></div>
                </div>
                
                <div class="plot-card">
                    <h2>Age Distribution</h2>
                    <p class="description">Distribution of investor ages with density curve</p>
                    <div id="age-plot"></div>
                </div>
                
                <div class="plot-card">
                    <h2>Investment Preferences Radar</h2>
                    <p class="description">Radar chart showing average investment preferences</p>
                    <div id="radar-plot"></div>
                </div>
                
                <div class="plot-card">
                    <h2>Customer Segmentation</h2>
                    <p class="description">3D visualization of customer clusters</p>
                    <div id="cluster-plot"></div>
                </div>
            </div>
            
            <script>
                // Load Plotly figures
                Promise.all([
                    fetch('../output/correlation_heatmap.html').then(r => r.text()),
                    fetch('../output/age_distribution.html').then(r => r.text()),
                    fetch('../output/investment_radar.html').then(r => r.text()),
                    fetch('../output/cluster_3d_visualization.html').then(r => r.text())
                ]).then(([corrHtml, ageHtml, radarHtml, clusterHtml]) => {
                    // Extract Plotly JSON from HTML files
                    function extractPlotlyData(html) {
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');
                        const scriptTag = doc.querySelector('script[type="application/json"]');
                        return JSON.parse(scriptTag.textContent);
                    }
                    
                    // Plot correlation heatmap
                    const corrData = extractPlotlyData(corrHtml);
                    Plotly.newPlot('correlation-plot', corrData.data, corrData.layout);
                    
                    // Plot age distribution
                    const ageData = extractPlotlyData(ageHtml);
                    Plotly.newPlot('age-plot', ageData.data, ageData.layout);
                    
                    // Plot radar chart
                    const radarData = extractPlotlyData(radarHtml);
                    Plotly.newPlot('radar-plot', radarData.data, radarData.layout);
                    
                    // Plot cluster visualization
                    const clusterData = extractPlotlyData(clusterHtml);
                    Plotly.newPlot('cluster-plot', clusterData.data, clusterData.layout);
                });
            </script>
        </body>
        </html>
        """
        
        # Save HTML file
        with open('../output/interactive_dashboard.html', 'w') as f:
            f.write(dashboard_html)
        
        print("Interactive dashboard saved to HTML file")
    
    def generate_report(self):
        """
        Generate comprehensive visualization report
        
        Returns:
            str: HTML report
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE VISUALIZATION REPORT")
        print("="*60)
        
        # Create all visualizations
        visualizations = []
        
        # 1. Correlation heatmap
        print("\n1. Creating correlation heatmap...")
        corr_fig = self.create_correlation_heatmap()
        visualizations.append(('Correlation Analysis', corr_fig))
        
        # 2. Distribution plots
        print("\n2. Creating distribution plots...")
        dist_fig = self.create_distribution_plots()
        visualizations.append(('Distribution Analysis', dist_fig))
        
        # 3. Age distribution
        print("\n3. Creating age distribution...")
        age_fig = self.create_age_distribution()
        if age_fig:
            visualizations.append(('Age Distribution', age_fig))
        
        # 4. Radar chart
        print("\n4. Creating radar chart...")
        radar_fig = self.create_investment_radar_chart()
        visualizations.append(('Investment Profile', radar_fig))
        
        # 5. Cluster visualization
        print("\n5. Creating cluster visualization...")
        cluster_fig = self.create_cluster_visualization()
        if cluster_fig:
            visualizations.append(('Customer Segmentation', cluster_fig))
        
        # 6. Time series analysis
        print("\n6. Creating time series analysis...")
        ts_fig = self.create_time_series_analysis()
        if ts_fig:
            visualizations.append(('Time Series Analysis', ts_fig))
        
        # 7. Create dashboard
        print("\n7. Creating interactive dashboard...")
        self.create_interactive_dashboard()
        
        print("\n" + "="*60)
        print("VISUALIZATION REPORT COMPLETED!")
        print("="*60)
        print("\nAll visualizations saved to ../output/ directory")
        print("Interactive dashboard: ../output/interactive_dashboard.html")
        
        return visualizations


def main():
    """Main function to run visualization dashboard"""
    print("="*60)
    print("INVESTMENT ANALYSIS VISUALIZATION DASHBOARD")
    print("="*60)
    
    # Initialize dashboard
    dashboard = InvestmentVisualizationDashboard()
    
    # Load data
    print("\n1. Loading data...")
    df = dashboard.load_data()
    
    # Generate complete report
    print("\n2. Generating visualizations...")
    visualizations = dashboard.generate_report()
    
    print("\n" + "="*60)
    print("DASHBOARD READY!")
    print("="*60)
    print("\nAccess your visualizations:")
    print("1. Interactive Dashboard: open ../output/interactive_dashboard.html")
    print("2. Individual Visualizations: check ../output/ directory")
    
    return dashboard, visualizations


if __name__ == "__main__":
    dashboard, visuals = main()
