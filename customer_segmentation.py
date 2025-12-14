"""
Customer Segmentation Module
LO2 Implementation: Machine learning library implementation
LO3 Implementation: Visualization of segmentation results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CustomerSegmenter:
    """
    Class for performing customer segmentation using clustering algorithms
    """
    
    def __init__(self, data_path='../data/cleaned_investment_data.csv'):
        """
        Initialize the customer segmenter
        
        Args:
            data_path (str): Path to cleaned data
        """
        self.data_path = data_path
        self.df = None
        self.features = None
        self.scaled_features = None
        self.scaler = None
        self.kmeans_model = None
        self.cluster_labels = None
        self.optimal_k = None
        
    def load_and_prepare_data(self):
        """
        Load and prepare data for clustering
        
        Returns:
            pd.DataFrame: Prepared data
        """
        print("Loading and preparing data for clustering...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Select features for clustering
        investment_features = [
            'Mutual_Funds', 'Equity_Market', 'Debentures',
            'Government_Bonds', 'Fixed_Deposits', 'PPF', 'Gold'
        ]
        
        # Check which features are available
        available_features = [f for f in investment_features if f in self.df.columns]
        
        # Add demographic features if available
        if 'Age' in self.df.columns:
            available_features.append('Age')
        
        if 'Total_Investment_Score' in self.df.columns:
            available_features.append('Total_Investment_Score')
        
        if 'Risk_Score' in self.df.columns:
            available_features.append('Risk_Score')
        
        self.features = available_features
        
        print(f"Selected {len(self.features)} features for clustering:")
        print(self.features)
        
        # Extract feature matrix
        X = self.df[self.features].values
        
        # Scale features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(X)
        
        return self.df
    
    def determine_optimal_clusters(self, max_k=10):
        """
        Determine optimal number of clusters using elbow method and silhouette score
        
        Args:
            max_k (int): Maximum number of clusters to test
            
        Returns:
            int: Optimal number of clusters
        """
        print("\nDetermining optimal number of clusters...")
        
        if self.scaled_features is None:
            self.load_and_prepare_data()
        
        inertia = []
        silhouette_scores = []
        K_range = range(2, max_k + 1)
        
        for k in K_range:
            # Fit K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_features)
            
            # Calculate inertia
            inertia.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if k > 1:  # Silhouette score requires at least 2 clusters
                silhouette_avg = silhouette_score(self.scaled_features, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_avg:.3f}")
        
        # Plot elbow curve and silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        ax1.plot(K_range, inertia, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12)
        ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette score plot
        ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Scores for Different k', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../output/optimal_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Determine optimal k (highest silhouette score)
        self.optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"\nOptimal number of clusters: {self.optimal_k}")
        
        return self.optimal_k
    
    def perform_clustering(self, n_clusters=None):
        """
        Perform K-Means clustering
        
        Args:
            n_clusters (int): Number of clusters, if None uses optimal_k
            
        Returns:
            array: Cluster labels
        """
        if n_clusters is None:
            if self.optimal_k is None:
                self.determine_optimal_clusters()
            n_clusters = self.optimal_k
        
        print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")
        
        # Fit K-Means
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,
            max_iter=300
        )
        
        self.cluster_labels = self.kmeans_model.fit_predict(self.scaled_features)
        self.df['Cluster'] = self.cluster_labels
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.scaled_features, self.cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        # Calculate inertia
        print(f"Inertia: {self.kmeans_model.inertia_:.2f}")
        
        # Display cluster sizes
        cluster_sizes = self.df['Cluster'].value_counts().sort_index()
        print("\nCluster Sizes:")
        for cluster, size in cluster_sizes.items():
            print(f"  Cluster {cluster}: {size} customers ({size/len(self.df)*100:.1f}%)")
        
        return self.cluster_labels
    
    def analyze_cluster_characteristics(self):
        """
        Analyze and visualize cluster characteristics
        
        Returns:
            pd.DataFrame: Cluster statistics
        """
        if self.cluster_labels is None:
            self.perform_clustering()
        
        print("\nAnalyzing cluster characteristics...")
        
        # Calculate cluster statistics
        cluster_stats = self.df.groupby('Cluster')[self.features].agg(['mean', 'std', 'count'])
        
        print("\nCluster Statistics (Means):")
        print(cluster_stats.xs('mean', axis=1, level=1).round(2))
        
        # Create comprehensive visualization
        self._visualize_clusters()
        
        return cluster_stats
    
    def _visualize_clusters(self):
        """Create comprehensive cluster visualizations"""
        
        # 1. Cluster distribution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot 1: Cluster distribution pie chart
        cluster_counts = self.df['Cluster'].value_counts().sort_index()
        colors = plt.cm.Set3(np.arange(len(cluster_counts)))
        
        axes[0].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index],
                   autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('Cluster Distribution', fontsize=14, fontweight='bold')
        
        # Plot 2: Average investment scores by cluster
        investment_cols = [col for col in self.features if col in ['Mutual_Funds', 'Equity_Market', 
                                                                  'Debentures', 'Government_Bonds', 
                                                                  'Fixed_Deposits', 'PPF', 'Gold']]
        
        if investment_cols:
            cluster_means = self.df.groupby('Cluster')[investment_cols].mean()
            
            x = np.arange(len(cluster_means.index))
            width = 0.15
            
            for i, col in enumerate(investment_cols[:5]):  # Limit to 5 for clarity
                axes[1].bar(x + (i-2)*width, cluster_means[col], width, label=col)
            
            axes[1].set_xlabel('Cluster')
            axes[1].set_ylabel('Average Score')
            axes[1].set_title('Investment Preferences by Cluster', fontsize=14, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f'Cluster {i}' for i in cluster_means.index])
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Age distribution by cluster
        if 'Age' in self.df.columns:
            sns.boxplot(data=self.df, x='Cluster', y='Age', ax=axes[2])
            axes[2].set_title('Age Distribution by Cluster', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Cluster')
            axes[2].set_ylabel('Age')
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Risk score by cluster (if available)
        if 'Risk_Score' in self.df.columns:
            risk_by_cluster = self.df.groupby('Cluster')['Risk_Score'].agg(['mean', 'std'])
            
            axes[3].bar(risk_by_cluster.index, risk_by_cluster['mean'], 
                       yerr=risk_by_cluster['std'], capsize=5, color='salmon', alpha=0.7)
            axes[3].set_xlabel('Cluster')
            axes[3].set_ylabel('Average Risk Score')
            axes[3].set_title('Risk Profile by Cluster', fontsize=14, fontweight='bold')
            axes[3].grid(True, alpha=0.3)
        
        # Plot 5: PCA visualization (2D projection)
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.scaled_features)
        
        scatter = axes[4].scatter(pca_result[:, 0], pca_result[:, 1], 
                                 c=self.cluster_labels, cmap='viridis', 
                                 alpha=0.7, s=50)
        axes[4].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[4].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[4].set_title('2D PCA Projection of Clusters', fontsize=14, fontweight='bold')
        axes[4].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[4], label='Cluster')
        
        # Plot 6: Cluster centroids heatmap
        if self.kmeans_model is not None:
            centroids = self.kmeans_model.cluster_centers_
            
            im = axes[5].imshow(centroids, aspect='auto', cmap='YlOrRd')
            axes[5].set_xlabel('Feature Index')
            axes[5].set_ylabel('Cluster')
            axes[5].set_title('Cluster Centroids Heatmap', fontsize=14, fontweight='bold')
            axes[5].set_xticks(np.arange(len(self.features)))
            axes[5].set_xticklabels([f'F{i+1}' for i in range(len(self.features))], rotation=45)
            axes[5].set_yticks(np.arange(centroids.shape[0]))
            axes[5].set_yticklabels([f'Cluster {i}' for i in range(centroids.shape[0])])
            plt.colorbar(im, ax=axes[5], label='Standardized Value')
        
        plt.tight_layout()
        plt.savefig('../output/cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create detailed cluster profiles
        self._create_cluster_profiles()
    
    def _create_cluster_profiles(self):
        """Create detailed profiles for each cluster"""
        print("\n" + "="*60)
        print("CLUSTER PROFILES")
        print("="*60)
        
        for cluster_num in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster_num]
            
            print(f"\n{'='*40}")
            print(f"CLUSTER {cluster_num} PROFILE")
            print(f"{'='*40}")
            print(f"Size: {len(cluster_data)} customers ({len(cluster_data)/len(self.df)*100:.1f}%)")
            
            # Demographic information
            if 'Age' in cluster_data.columns:
                print(f"\nDemographics:")
                print(f"  Average Age: {cluster_data['Age'].mean():.1f} years")
                print(f"  Age Range: {cluster_data['Age'].min():.0f}-{cluster_data['Age'].max():.0f}")
            
            if 'Gender' in cluster_data.columns:
                gender_dist = cluster_data['Gender'].value_counts()
                if not gender_dist.empty:
                    print(f"  Gender: {gender_dist.to_dict()}")
            
            # Investment preferences
            investment_cols = [col for col in self.features if col in ['Mutual_Funds', 'Equity_Market', 
                                                                      'Debentures', 'Government_Bonds', 
                                                                      'Fixed_Deposits', 'PPF', 'Gold']]
            
            if investment_cols:
                print(f"\nInvestment Preferences (Average Scores 1-7):")
                for col in investment_cols:
                    avg_score = cluster_data[col].mean()
                    # Create simple bar visualization
                    bar_length = int(avg_score)
                    bar = '█' * bar_length + '░' * (7 - bar_length)
                    print(f"  {col:<20}: {avg_score:.2f} {bar}")
            
            # Risk profile
            if 'Risk_Score' in cluster_data.columns:
                risk_avg = cluster_data['Risk_Score'].mean()
                print(f"\nRisk Profile:")
                print(f"  Average Risk Score: {risk_avg:.2f}")
                
                if risk_avg < 3:
                    print(f"  Profile: Conservative Investor")
                    print(f"  Characteristics: Prefers safe investments, capital preservation")
                elif risk_avg < 5:
                    print(f"  Profile: Moderate Investor")
                    print(f"  Characteristics: Balanced approach, steady growth")
                else:
                    print(f"  Profile: Aggressive Investor")
                    print(f"  Characteristics: High-risk tolerance, seeks maximum returns")
            
            # Digital transformation recommendations
            print(f"\nDigital Transformation Recommendations:")
            self._generate_recommendations(cluster_num, cluster_data)
    
    def _generate_recommendations(self, cluster_num, cluster_data):
        """Generate digital transformation recommendations for cluster"""
        recommendations = {
            0: [
                "Focus on mobile-first design with simple navigation",
                "Implement educational content about basic investing",
                "Create automated savings plans with low minimums",
                "Develop gamified learning modules",
                "Offer robo-advisor services with conservative portfolios"
            ],
            1: [
                "Provide advanced portfolio analytics tools",
                "Implement tax optimization features",
                "Create goal-based investment planning",
                "Offer automated rebalancing services",
                "Develop retirement planning calculators"
            ],
            2: [
                "Build advanced trading platforms with real-time data",
                "Implement AI-driven investment recommendations",
                "Create social trading features",
                "Offer margin trading and advanced order types",
                "Develop cryptocurrency integration"
            ]
        }
        
        # Default recommendations if cluster number not in dictionary
        default_recs = recommendations.get(cluster_num, [
            "Personalize dashboard based on investment preferences",
            "Implement automated portfolio monitoring",
            "Create personalized investment alerts",
            "Develop interactive financial planning tools",
            "Offer seamless integration with other financial apps"
        ])
        
        for i, rec in enumerate(default_recs, 1):
            print(f"  {i}. {rec}")
    
    def save_segmentation_results(self, output_path='../output/segmentation_results.csv'):
        """Save segmentation results to CSV"""
        if self.cluster_labels is not None:
            results_df = self.df.copy()
            
            # Add cluster labels and probabilities if available
            if self.kmeans_model is not None:
                distances = self.kmeans_model.transform(self.scaled_features)
                # Convert distances to probabilities (soft clustering)
                probabilities = 1 / (1 + distances)
                probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
                
                for i in range(self.kmeans_model.n_clusters):
                    results_df[f'Cluster_{i}_Probability'] = probabilities[:, i]
            
            results_df.to_csv(output_path, index=False)
            print(f"\nSegmentation results saved to: {output_path}")
            
            # Save cluster statistics
            cluster_stats = self.df.groupby('Cluster').agg({
                'Age': ['mean', 'std', 'count'] if 'Age' in self.df.columns else [],
                'Total_Investment_Score': ['mean', 'std'] if 'Total_Investment_Score' in self.df.columns else [],
                'Risk_Score': ['mean', 'std'] if 'Risk_Score' in self.df.columns else []
            })
            
            # Flatten multi-index columns
            cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
            cluster_stats.to_csv('../output/cluster_statistics.csv')
            
            return results_df
        else:
            print("No segmentation results available. Run perform_clustering() first.")
            return None


def main():
    """Main function to demonstrate customer segmentation"""
    print("="*60)
    print("CUSTOMER SEGMENTATION ANALYSIS")
    print("="*60)
    
    # Initialize segmenter
    segmenter = CustomerSegmenter()
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    df = segmenter.load_and_prepare_data()
    
    # Determine optimal clusters
    print("\n2. Determining optimal number of clusters...")
    optimal_k = segmenter.determine_optimal_clusters(max_k=8)
    
    # Perform clustering
    print("\n3. Performing clustering...")
    cluster_labels = segmenter.perform_clustering(n_clusters=optimal_k)
    
    # Analyze clusters
    print("\n4. Analyzing cluster characteristics...")
    cluster_stats = segmenter.analyze_cluster_characteristics()
    
    # Save results
    print("\n5. Saving results...")
    results_df = segmenter.save_segmentation_results()
    
    print("\n" + "="*60)
    print("SEGMENTATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return results_df, segmenter


if __name__ == "__main__":
    results, segmenter = main()
