"""
Data Loading and Cleaning Module
LO2 Implementation: Data analysis library implementation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InvestmentDataProcessor:
    """
    Class for loading and cleaning investment preferences data
    """
    
    def __init__(self, data_path='../data/investment_data.csv'):
        """
        Initialize the data processor
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = Path(data_path)
        self.df = None
        self.cleaned_df = None
        self.logger = logger
        
    def load_data(self):
        """
        Load the investment preferences dataset
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.df = pd.read_csv(self.data_path)
            self.logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            self.logger.info(f"Columns: {list(self.df.columns)}")
            
            # Display basic information
            self._display_data_info()
            
            return self.df
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.data_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _display_data_info(self):
        """Display basic information about the dataset"""
        print("="*50)
        print("DATASET INFORMATION")
        print("="*50)
        print(f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print("\nData Types:")
        print(self.df.dtypes)
        
        print("\nMissing Values:")
        missing_values = self.df.isnull().sum()
        missing_percent = (missing_values / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Values'] > 0])
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nDescriptive Statistics:")
        print(self.df.describe())
    
    def clean_data(self):
        """
        Clean and preprocess the dataset
        
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        if self.df is None:
            self.load_data()
        
        self.logger.info("Starting data cleaning process...")
        
        # Create a copy for cleaning
        self.cleaned_df = self.df.copy()
        
        # 1. Handle missing values
        self._handle_missing_values()
        
        # 2. Validate data ranges
        self._validate_data_ranges()
        
        # 3. Create derived features
        self._create_derived_features()
        
        # 4. Encode categorical variables
        self._encode_categorical_variables()
        
        # 5. Standardize numerical features
        self._standardize_features()
        
        self.logger.info("Data cleaning completed successfully")
        self._display_cleaning_summary()
        
        return self.cleaned_df
    
    def _handle_missing_values(self):
        """Handle any missing values in the dataset"""
        initial_missing = self.cleaned_df.isnull().sum().sum()
        
        if initial_missing > 0:
            self.logger.info(f"Found {initial_missing} missing values")
            
            # For numerical columns, fill with median
            numerical_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.cleaned_df[col].isnull().any():
                    median_val = self.cleaned_df[col].median()
                    self.cleaned_df[col].fillna(median_val, inplace=True)
                    self.logger.info(f"Filled missing values in {col} with median: {median_val}")
            
            # For categorical columns, fill with mode
            categorical_cols = self.cleaned_df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.cleaned_df[col].isnull().any():
                    mode_val = self.cleaned_df[col].mode()[0]
                    self.cleaned_df[col].fillna(mode_val, inplace=True)
                    self.logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        final_missing = self.cleaned_df.isnull().sum().sum()
        if final_missing == 0:
            self.logger.info("All missing values handled successfully")
    
    def _validate_data_ranges(self):
        """Validate that all values are within expected ranges"""
        # Define expected ranges for numerical columns
        expected_ranges = {
            'Age': (21, 35),
            'Mutual_Funds': (1, 7),
            'Equity_Market': (1, 7),
            'Debentures': (1, 7),
            'Government_Bonds': (1, 7),
            'Fixed_Deposits': (1, 7),
            'PPF': (1, 7),
            'Gold': (1, 7)
        }
        
        for col, (min_val, max_val) in expected_ranges.items():
            if col in self.cleaned_df.columns:
                # Check for values outside range
                out_of_range = ((self.cleaned_df[col] < min_val) | 
                               (self.cleaned_df[col] > max_val)).sum()
                if out_of_range > 0:
                    self.logger.warning(f"{col}: {out_of_range} values outside range [{min_val}, {max_val}]")
                    # Clip values to range
                    self.cleaned_df[col] = self.cleaned_df[col].clip(min_val, max_val)
    
    def _create_derived_features(self):
        """Create derived features for analysis"""
        self.logger.info("Creating derived features...")
        
        # Calculate total investment score
        investment_cols = ['Mutual_Funds', 'Equity_Market', 'Debentures', 
                          'Government_Bonds', 'Fixed_Deposits', 'PPF', 'Gold']
        
        if all(col in self.cleaned_df.columns for col in investment_cols):
            self.cleaned_df['Total_Investment_Score'] = self.cleaned_df[investment_cols].sum(axis=1)
            self.cleaned_df['Average_Investment_Score'] = self.cleaned_df[investment_cols].mean(axis=1)
            
            # Calculate risk score (higher weight to equity and mutual funds)
            risk_weights = {
                'Mutual_Funds': 0.8,
                'Equity_Market': 1.0,
                'Debentures': 0.4,
                'Government_Bonds': 0.2,
                'Fixed_Deposits': 0.1,
                'PPF': 0.1,
                'Gold': 0.3
            }
            
            risk_score = 0
            for col, weight in risk_weights.items():
                if col in self.cleaned_df.columns:
                    risk_score += self.cleaned_df[col] * weight
            
            self.cleaned_df['Risk_Score'] = risk_score / sum(risk_weights.values())
        
        # Create age groups
        self.cleaned_df['Age_Group'] = pd.cut(
            self.cleaned_df['Age'],
            bins=[20, 25, 30, 36],
            labels=['21-25', '26-30', '31-35']
        )
        
        # Create investor type based on risk score
        if 'Risk_Score' in self.cleaned_df.columns:
            conditions = [
                self.cleaned_df['Risk_Score'] < 3,
                (self.cleaned_df['Risk_Score'] >= 3) & (self.cleaned_df['Risk_Score'] < 5),
                self.cleaned_df['Risk_Score'] >= 5
            ]
            choices = ['Conservative', 'Moderate', 'Aggressive']
            self.cleaned_df['Investor_Type'] = np.select(conditions, choices, default='Moderate')
        
        self.logger.info("Derived features created successfully")
    
    def _encode_categorical_variables(self):
        """Encode categorical variables for analysis"""
        # Encode gender
        if 'Gender' in self.cleaned_df.columns:
            self.cleaned_df['Gender_Encoded'] = self.cleaned_df['Gender'].map({'Male': 0, 'Female': 1})
        
        # Encode binary columns (Yes/No to 1/0)
        binary_cols = [col for col in self.cleaned_df.columns 
                      if col.startswith(('Reason_', 'Purpose_', 'Savings_Goal_'))]
        
        for col in binary_cols:
            if col in self.cleaned_df.columns:
                if self.cleaned_df[col].dtype == 'object':
                    self.cleaned_df[col] = self.cleaned_df[col].map({'Yes': 1, 'No': 0})
    
    def _standardize_features(self):
        """Standardize numerical features for machine learning"""
        from sklearn.preprocessing import StandardScaler
        
        numerical_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove binary columns from standardization
        binary_cols = [col for col in numerical_cols 
                      if col.startswith(('Reason_', 'Purpose_', 'Savings_Goal_'))]
        numerical_cols = [col for col in numerical_cols if col not in binary_cols]
        
        if numerical_cols:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.cleaned_df[numerical_cols])
            self.cleaned_df[[f'{col}_scaled' for col in numerical_cols]] = scaled_data
            
            self.scaler = scaler
            self.logger.info(f"Standardized {len(numerical_cols)} numerical features")
    
    def _display_cleaning_summary(self):
        """Display summary of cleaning process"""
        print("\n" + "="*50)
        print("DATA CLEANING SUMMARY")
        print("="*50)
        
        print(f"Original shape: {self.df.shape}")
        print(f"Cleaned shape: {self.cleaned_df.shape}")
        
        print("\nNew columns created:")
        original_cols = set(self.df.columns)
        cleaned_cols = set(self.cleaned_df.columns)
        new_cols = cleaned_cols - original_cols
        print(list(new_cols))
        
        print("\nData quality check:")
        print(f"Missing values: {self.cleaned_df.isnull().sum().sum()}")
        print(f"Duplicate rows: {self.cleaned_df.duplicated().sum()}")
    
    def save_cleaned_data(self, output_path='../data/cleaned_investment_data.csv'):
        """Save cleaned data to CSV"""
        if self.cleaned_df is not None:
            output_path = Path(output_path)
            self.cleaned_df.to_csv(output_path, index=False)
            self.logger.info(f"Cleaned data saved to: {output_path}")
        else:
            self.logger.warning("No cleaned data available. Run clean_data() first.")
    
    def get_data_summary(self):
        """Get comprehensive data summary"""
        if self.cleaned_df is None:
            self.clean_data()
        
        summary = {
            'shape': self.cleaned_df.shape,
            'columns': list(self.cleaned_df.columns),
            'data_types': self.cleaned_df.dtypes.to_dict(),
            'missing_values': self.cleaned_df.isnull().sum().to_dict(),
            'numerical_stats': self.cleaned_df.describe().to_dict(),
            'categorical_counts': {}
        }
        
        # Add categorical value counts
        categorical_cols = self.cleaned_df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary['categorical_counts'][col] = self.cleaned_df[col].value_counts().to_dict()
        
        return summary


def main():
    """Main function to demonstrate data loading and cleaning"""
    print("="*60)
    print("DATA LOADING AND CLEANING DEMONSTRATION")
    print("="*60)
    
    # Initialize processor
    processor = InvestmentDataProcessor()
    
    # Load data
    print("\n1. Loading data...")
    df = processor.load_data()
    
    # Clean data
    print("\n2. Cleaning data...")
    cleaned_df = processor.clean_data()
    
    # Save cleaned data
    print("\n3. Saving cleaned data...")
    processor.save_cleaned_data()
    
    # Display summary
    print("\n4. Data Summary:")
    summary = processor.get_data_summary()
    print(f"Final dataset shape: {summary['shape']}")
    print(f"Number of features: {len(summary['columns'])}")
    
    return cleaned_df


if __name__ == "__main__":
    cleaned_data = main()
    print("\nData processing completed successfully!")
