"""
Regression Analysis Module
LO2 Implementation: Predictive modeling with Scikit-Learn
LO3 Implementation: Model evaluation and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class InvestmentPredictor:
    """
    Class for predictive modeling of investment behaviors
    """
    
    def __init__(self, data_path='../data/cleaned_investment_data.csv'):
        """
        Initialize the investment predictor
        
        Args:
            data_path (str): Path to cleaned data
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self, target_variable='Age'):
        """
        Load data and prepare features for regression
        
        Args:
            target_variable (str): Variable to predict
            
        Returns:
            tuple: Features (X) and target (y)
        """
        print("Loading and preparing data for regression analysis...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Define feature columns
        feature_columns = [
            'Mutual_Funds', 'Equity_Market', 'Debentures',
            'Government_Bonds', 'Fixed_Deposits', 'PPF', 'Gold'
        ]
        
        # Add binary features if available
        binary_features = [col for col in self.df.columns 
                          if col.startswith(('Reason_', 'Purpose_', 'Savings_Goal_'))]
        feature_columns.extend(binary_features)
        
        # Add derived features if available
        derived_features = ['Total_Investment_Score', 'Risk_Score', 
                          'Average_Investment_Score']
        for feature in derived_features:
            if feature in self.df.columns:
                feature_columns.append(feature)
        
        # Remove target variable from features if it's included
        if target_variable in feature_columns:
            feature_columns.remove(target_variable)
        
        # Check if target variable exists
        if target_variable not in self.df.columns:
            print(f"Warning: Target variable '{target_variable}' not found in data")
            print(f"Available columns: {list(self.df.columns)}")
            # Use first available column as fallback
            target_variable = self.df.columns[0]
            print(f"Using '{target_variable}' as target variable instead")
        
        # Prepare features and target
        self.X = self.df[feature_columns].copy()
        self.y = self.df[target_variable].copy()
        
        print(f"\nData Preparation Summary:")
        print(f"  Features: {len(feature_columns)} variables")
        print(f"  Target: {target_variable}")
        print(f"  Samples: {len(self.X)}")
        
        # Display feature information
        self._display_feature_info()
        
        return self.X, self.y
    
    def _display_feature_info(self):
        """Display information about features"""
        print("\nFeature Information:")
        print("-" * 40)
        
        # Categorical features
        categorical_features = self.X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_features) > 0:
            print(f"Categorical Features ({len(categorical_features)}):")
            for feature in categorical_features:
                unique_values = self.X[feature].nunique()
                print(f"  {feature}: {unique_values} unique values")
        
        # Numerical features
        numerical_features = self.X.select_dtypes(include=[np.number]).columns
        if len(numerical_features) > 0:
            print(f"\nNumerical Features ({len(numerical_features)}):")
            stats = self.X[numerical_features].describe().T[['mean', 'std', 'min', 'max']]
            print(stats.round(2))
        
        # Target variable information
        print(f"\nTarget Variable Information:")
        print(f"  Name: {self.y.name}")
        print(f"  Type: {self.y.dtype}")
        print(f"  Range: {self.y.min():.2f} to {self.y.max():.2f}")
        print(f"  Mean: {self.y.mean():.2f}, Std: {self.y.std():.2f}")
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            test_size (float): Proportion of test data
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        if self.X is None or self.y is None:
            self.load_and_prepare_data()
        
        print(f"\nSplitting data (test_size={test_size})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Testing set: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """
        Scale numerical features
        
        Returns:
            tuple: Scaled X_train and X_test
        """
        print("\nScaling numerical features...")
        
        # Identify numerical features
        numerical_features = self.X.select_dtypes(include=[np.number]).columns
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        
        # Scale training data
        X_train_scaled = self.X_train.copy()
        X_train_scaled[numerical_features] = self.scaler.fit_transform(
            self.X_train[numerical_features]
        )
        
        # Scale testing data
        X_test_scaled = self.X_test.copy()
        X_test_scaled[numerical_features] = self.scaler.transform(
            self.X_test[numerical_features]
        )
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        
        print(f"  Scaled {len(numerical_features)} numerical features")
        
        return self.X_train, self.X_test
    
    def perform_feature_selection(self, k=10):
        """
        Perform feature selection using statistical tests
        
        Args:
            k (int): Number of top features to select
            
        Returns:
            array: Selected feature indices
        """
        print(f"\nPerforming feature selection (top {k} features)...")
        
        # Use SelectKBest with f_regression
        selector = SelectKBest(score_func=f_regression, k=min(k, self.X.shape[1]))
        X_selected = selector.fit_transform(self.X_train, self.y_train)
        
        # Get selected feature indices and scores
        selected_indices = selector.get_support(indices=True)
        feature_scores = selector.scores_
        
        print("\nTop Features by Importance:")
        print("-" * 40)
        
        feature_importance = pd.DataFrame({
            'Feature': self.X.columns,
            'Score': feature_scores,
            'Selected': [i in selected_indices for i in range(len(self.X.columns))]
        })
        
        # Sort by score
        feature_importance = feature_importance.sort_values('Score', ascending=False)
        
        # Display top features
        print(feature_importance.head(k).round(3))
        
        # Visualize feature importance
        self._plot_feature_importance(feature_importance.head(k))
        
        # Update X_train and X_test with selected features
        selected_features = self.X.columns[selected_indices]
        self.X_train = self.X_train[selected_features]
        self.X_test = self.X_test[selected_features]
        
        print(f"\nSelected {len(selected_features)} features:")
        print(list(selected_features))
        
        return selected_indices, feature_scores
    
    def _plot_feature_importance(self, feature_importance):
        """Plot feature importance scores"""
        plt.figure(figsize=(10, 6))
        
        bars = plt.barh(range(len(feature_importance)), 
                       feature_importance['Score'].values,
                       color=plt.cm.viridis(np.linspace(0, 1, len(feature_importance))))
        
        plt.yticks(range(len(feature_importance)), feature_importance['Feature'].values)
        plt.xlabel('Feature Importance Score', fontsize=12)
        plt.title('Top Features by Importance Score', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center')
        
        plt.gca().invert_yaxis()  # Highest score at top
        plt.tight_layout()
        plt.savefig('../output/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_models(self):
        """
        Train multiple regression models
        
        Returns:
            dict: Trained models
        """
        print("\nTraining regression models...")
        
        # Define models to train
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"  Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
        
        print("\nModel training completed!")
        
        return self.models
    
    def evaluate_models(self):
        """
        Evaluate all trained models
        
        Returns:
            pd.DataFrame: Evaluation results
        """
        print("\nEvaluating models...")
        
        evaluation_results = []
        
        for name, model in self.models.items():
            # Make predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            results = {
                'Model': name,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'Train_MSE': train_mse,
                'Test_MSE': test_mse,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'CV_Mean_R2': cv_mean,
                'CV_Std_R2': cv_std
            }
            
            evaluation_results.append(results)
            
            print(f"\n{name} Performance:")
            print(f"  Train R²: {train_r2:.3f}")
            print(f"  Test R²:  {test_r2:.3f}")
            print(f"  CV R²:    {cv_mean:.3f} (±{cv_std:.3f})")
            print(f"  Train MSE: {train_mse:.3f}")
            print(f"  Test MSE:  {test_mse:.3f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(evaluation_results)
        self.results = results_df
        
        # Visualize model comparison
        self._plot_model_comparison(results_df)
        
        return results_df
    
    def _plot_model_comparison(self, results_df):
        """Create visualization comparing model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: R² scores
        x = np.arange(len(results_df))
        width = 0.35
        
        axes[0,0].bar(x - width/2, results_df['Train_R2'], width, label='Train', alpha=0.7)
        axes[0,0].bar(x + width/2, results_df['Test_R2'], width, label='Test', alpha=0.7)
        axes[0,0].set_xlabel('Model')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_title('Model R² Scores', fontsize=14, fontweight='bold')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: MSE scores
        axes[0,1].bar(x - width/2, results_df['Train_MSE'], width, label='Train', alpha=0.7)
        axes[0,1].bar(x + width/2, results_df['Test_MSE'], width, label='Test', alpha=0.7)
        axes[0,1].set_xlabel('Model')
        axes[0,1].set_ylabel('MSE')
        axes[0,1].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Cross-validation scores
        axes[1,0].bar(x, results_df['CV_Mean_R2'], yerr=results_df['CV_Std_R2'], 
                     capsize=5, alpha=0.7, color='green')
        axes[1,0].set_xlabel('Model')
        axes[1,0].set_ylabel('Cross-Validated R²')
        axes[1,0].set_title('Cross-Validation Performance', fontsize=14, fontweight='bold')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Model ranking
        results_df['Overall_Score'] = (
            results_df['Test_R2'] * 0.4 +
            results_df['CV_Mean_R2'] * 0.4 +
            (1 - results_df['Test_MSE']/results_df['Test_MSE'].max()) * 0.2
        )
        
        results_sorted = results_df.sort_values('Overall_Score', ascending=True)
        axes[1,1].barh(range(len(results_sorted)), results_sorted['Overall_Score'], 
                      color=plt.cm.coolwarm(np.linspace(0, 1, len(results_sorted))))
        axes[1,1].set_yticks(range(len(results_sorted)))
        axes[1,1].set_yticklabels(results_sorted['Model'])
        axes[1,1].set_xlabel('Overall Score')
        axes[1,1].set_title('Model Ranking', fontsize=14, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('../output/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nModel Ranking by Overall Score:")
        print(results_sorted[['Model', 'Overall_Score']].round(3))
    
    def plot_residuals(self, model_name='Random Forest'):
        """
        Plot residuals for a specific model
        
        Args:
            model_name (str): Name of model to analyze
            
        Returns:
            tuple: Predicted values and residuals
        """
        if model_name not in self.models:
            print(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            return None, None
        
        print(f"\nAnalyzing residuals for {model_name}...")
        
        model = self.models[model_name]
        
        # Make predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Calculate residuals
        train_residuals = self.y_train - y_train_pred
        test_residuals = self.y_test - y_test_pred
        
        # Create residual plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Residuals vs Predicted (Training)
        axes[0,0].scatter(y_train_pred, train_residuals, alpha=0.5, color='blue')
        axes[0,0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0,0].set_xlabel('Predicted Values')
        axes[0,0].set_ylabel('Residuals')
        axes[0,0].set_title('Residuals vs Predicted (Training)', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals vs Predicted (Testing)
        axes[0,1].scatter(y_test_pred, test_residuals, alpha=0.5, color='green')
        axes[0,1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0,1].set_xlabel('Predicted Values')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residuals vs Predicted (Testing)', fontsize=14, fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Distribution of residuals (Training)
        axes[1,0].hist(train_residuals, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1,0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Residual Distribution (Training)', fontsize=14, fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Distribution of residuals (Testing)
        axes[1,1].hist(test_residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1,1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1,1].set_xlabel('Residuals')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Residual Distribution (Testing)', fontsize=14, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'../output/residuals_{model_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate residual statistics
        residual_stats = {
            'Training': {
                'Mean': np.mean(train_residuals),
                'Std': np.std(train_residuals),
                'Min': np.min(train_residuals),
                'Max': np.max(train_residuals)
            },
            'Testing': {
                'Mean': np.mean(test_residuals),
                'Std': np.std(test_residuals),
                'Min': np.min(test_residuals),
                'Max': np.max(test_residuals)
            }
        }
        
        print("\nResidual Statistics:")
        print(pd.DataFrame(residual_stats).round(3))
        
        return y_test_pred, test_residuals
    
    def plot_learning_curve(self, model_name='Random Forest'):
        """
        Plot learning curve for a model
        
        Args:
            model_name (str): Name of model to analyze
            
        Returns:
            tuple: Learning curve data
        """
        if model_name not in self.models:
            print(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            return None, None, None
        
        print(f"\nGenerating learning curve for {model_name}...")
        
        model = self.models[model_name]
        
        # Generate learning curve data
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X_train, self.y_train,
            cv=5, scoring='r2',
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
        )
        
        # Calculate mean and standard deviation
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        
        plt.fill_between(train_sizes, 
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, 
                        alpha=0.1, color='blue')
        plt.fill_between(train_sizes, 
                        test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, 
                        alpha=0.1, color='orange')
        
        plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', 
                label='Training score', linewidth=2)
        plt.plot(train_sizes, test_scores_mean, 'o-', color='orange', 
                label='Cross-validation score', linewidth=2)
        
        plt.xlabel('Training Examples', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.title(f'Learning Curve: {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'../output/learning_curve_{model_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze learning curve
        gap = train_scores_mean[-1] - test_scores_mean[-1]
        print(f"\nLearning Curve Analysis:")
        print(f"  Final Training Score: {train_scores_mean[-1]:.3f}")
        print(f"  Final CV Score: {test_scores_mean[-1]:.3f}")
        print(f"  Gap (Overfitting): {gap:.3f}")
        
        if gap > 0.1:
            print("  → Model shows signs of overfitting")
        elif gap < 0.05:
            print("  → Model shows good generalization")
        else:
            print("  → Model shows moderate overfitting")
        
        return train_sizes, train_scores_mean, test_scores_mean
    
    def save_model_results(self):
        """Save model results and predictions"""
        print("\nSaving model results...")
        
        # Save evaluation results
        if len(self.results) > 0:
            self.results.to_csv('../output/model_evaluation_results.csv', index=False)
            print("  Model evaluation results saved")
        
        # Save predictions for best model
        best_model_name = self.results.loc[self.results['Overall_Score'].idxmax(), 'Model']
        best_model = self.models[best_model_name]
        
        # Make predictions with best model
        y_pred = best_model.predict(self.X_test)
        
        # Create prediction DataFrame
        predictions_df = pd.DataFrame({
            'Actual': self.y_test.values,
            'Predicted': y_pred,
            'Residual': self.y_test.values - y_pred
        })
        
        predictions_df.to_csv('../output/best_model_predictions.csv', index=False)
        print(f"  Best model ({best_model_name}) predictions saved")
        
        # Save model coefficients if applicable
        if hasattr(best_model, 'coef_'):
            coefficients = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Coefficient': best_model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            coefficients.to_csv('../output/model_coefficients.csv', index=False)
            print("  Model coefficients saved")
        
        # Save feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            importances = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            importances.to_csv('../output/feature_importances.csv', index=False)
            print("  Feature importances saved")
        
        return predictions_df


def main():
    """Main function to demonstrate regression analysis"""
    print("="*60)
    print("REGRESSION ANALYSIS FOR INVESTMENT BEHAVIOR")
    print("="*60)
    
    # Initialize predictor
    predictor = InvestmentPredictor()
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    X, y = predictor.load_and_prepare_data(target_variable='Age')
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = predictor.split_data(test_size=0.2)
    
    # Scale features
    print("\n3. Scaling features...")
    X_train_scaled, X_test_scaled = predictor.scale_features()
    
    # Feature selection
    print("\n4. Performing feature selection...")
    selected_indices, feature_scores = predictor.perform_feature_selection(k=8)
    
    # Train models
    print("\n5. Training models...")
    models = predictor.train_models()
    
    # Evaluate models
    print("\n6. Evaluating models...")
    results_df = predictor.evaluate_models()
    
    # Analyze residuals for best model
    print("\n7. Analyzing residuals...")
    best_model = results_df.loc[results_df['Overall_Score'].idxmax(), 'Model']
    y_pred, residuals = predictor.plot_residuals(model_name=best_model)
    
    # Plot learning curve
    print("\n8. Generating learning curve...")
    train_sizes, train_scores, test_scores = predictor.plot_learning_curve(model_name=best_model)
    
    # Save results
    print("\n9. Saving results...")
    predictions_df = predictor.save_model_results()
    
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return predictor, results_df


if __name__ == "__main__":
    predictor, results = main()
