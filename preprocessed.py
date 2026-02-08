"""
Credit Card Dataset Preprocessing Pipeline
==========================================
A professional, modular preprocessing pipeline for credit card fraud detection.

Author: Data Science Pipeline
Date: 2026-02-05
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# For handling class imbalance
try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. Resampling methods unavailable.")
    print("Install with: pip install imbalanced-learn")

warnings.filterwarnings('ignore')

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    data_path: str = "creditcard.csv"
    output_dir: str = "processed_data"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    scaler_type: str = "robust"  # Options: 'standard', 'robust', 'minmax'
    handle_imbalance: bool = True
    imbalance_method: str = "smote"  # Options: 'smote', 'adasyn', 'undersample', 'smote_tomek', 'smote_enn'
    sampling_strategy: float = 0.5  # Ratio of minority to majority after resampling
    remove_outliers: bool = False
    outlier_threshold: float = 3.0  # Z-score threshold for outlier removal


class CreditCardPreprocessor:
    """
    Professional preprocessing pipeline for credit card fraud detection dataset.
    
    Features:
    - Comprehensive EDA and data quality checks
    - Multiple scaling options
    - Class imbalance handling (SMOTE, ADASYN, undersampling)
    - Outlier detection and handling
    - Train/Val/Test split with stratification
    - Feature engineering options
    - Export processed data
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.scaler: Optional[Any] = None
        self.feature_names: Optional[list] = None
        self.preprocessing_report: Dict[str, Any] = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load the credit card dataset."""
        print("=" * 60)
        print("STEP 1: Loading Data")
        print("=" * 60)
        
        self.df = pd.read_csv(self.config.data_path)
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        self.preprocessing_report['original_shape'] = self.df.shape
        return self.df
    
    def explore_data(self) -> Dict[str, Any]:
        """Perform comprehensive exploratory data analysis."""
        print("\n" + "=" * 60)
        print("STEP 2: Exploratory Data Analysis")
        print("=" * 60)
        
        eda_report = {}
        
        # Basic info
        print("\n📊 Dataset Overview:")
        print(f"  • Total samples: {len(self.df):,}")
        print(f"  • Features: {self.df.shape[1]}")
        print(f"  • Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column information
        print(f"\n📋 Columns: {self.df.columns.tolist()}")
        
        # Data types
        print(f"\n🔢 Data Types:")
        for dtype, count in self.df.dtypes.value_counts().items():
            print(f"  • {dtype}: {count} columns")
        
        # Missing values
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        eda_report['missing_values'] = missing[missing > 0].to_dict()
        
        print(f"\n❓ Missing Values:")
        if missing.sum() == 0:
            print("  ✓ No missing values found")
        else:
            for col in missing[missing > 0].index:
                print(f"  • {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")
        
        # Detect target column
        target_col = self._detect_target_column()
        print(f"\n🎯 Target Column Detected: '{target_col}'")
        
        # Class distribution
        if target_col:
            class_dist = self.df[target_col].value_counts()
            class_pct = self.df[target_col].value_counts(normalize=True) * 100
            
            print(f"\n⚖️ Class Distribution:")
            for cls in class_dist.index:
                label = "Legitimate" if cls == 0 else "Fraud"
                print(f"  • Class {cls} ({label}): {class_dist[cls]:,} ({class_pct[cls]:.2f}%)")
            
            imbalance_ratio = class_dist.max() / class_dist.min()
            print(f"  • Imbalance Ratio: {imbalance_ratio:.2f}:1")
            eda_report['class_distribution'] = class_dist.to_dict()
            eda_report['imbalance_ratio'] = imbalance_ratio
        
        # Statistical summary
        print(f"\n📈 Statistical Summary:")
        stats = self.df.describe()
        print(stats.to_string())
        
        # Duplicates check
        duplicates = self.df.duplicated().sum()
        print(f"\n🔄 Duplicate Rows: {duplicates:,} ({(duplicates/len(self.df))*100:.2f}%)")
        eda_report['duplicates'] = duplicates
        
        self.preprocessing_report['eda'] = eda_report
        return eda_report
    
    def _detect_target_column(self) -> str:
        """Auto-detect the target column."""
        potential_targets = ['Class', 'class', 'TARGET', 'target', 'label', 'Label', 'fraud', 'Fraud']
        for col in potential_targets:
            if col in self.df.columns:
                return col
        # If not found, assume last column is target
        return self.df.columns[-1]
    
    def clean_data(self) -> pd.DataFrame:
        """Clean the dataset: handle missing values, duplicates, and outliers."""
        print("\n" + "=" * 60)
        print("STEP 3: Data Cleaning")
        print("=" * 60)
        
        initial_rows = len(self.df)
        
        # Handle missing values
        missing_before = self.df.isnull().sum().sum()
        if missing_before > 0:
            print(f"\n🔧 Handling {missing_before} missing values...")
            # For numerical columns, use median imputation
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy='median')
            self.df[numerical_cols] = imputer.fit_transform(self.df[numerical_cols])
            print(f"  ✓ Missing values imputed using median strategy")
        else:
            print(f"\n✓ No missing values to handle")
        
        # Remove duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"\n🔧 Removing {duplicates:,} duplicate rows...")
            self.df = self.df.drop_duplicates()
            print(f"  ✓ Duplicates removed")
        else:
            print(f"\n✓ No duplicate rows found")
        
        # Handle outliers if configured
        if self.config.remove_outliers:
            print(f"\n🔧 Removing outliers (Z-score > {self.config.outlier_threshold})...")
            target_col = self._detect_target_column()
            feature_cols = [col for col in self.df.columns if col != target_col]
            
            # Calculate z-scores for feature columns
            z_scores = np.abs((self.df[feature_cols] - self.df[feature_cols].mean()) / self.df[feature_cols].std())
            outlier_mask = (z_scores < self.config.outlier_threshold).all(axis=1)
            outliers_removed = (~outlier_mask).sum()
            
            self.df = self.df[outlier_mask]
            print(f"  ✓ Removed {outliers_removed:,} outlier rows")
        
        final_rows = len(self.df)
        print(f"\n📊 Cleaning Summary:")
        print(f"  • Initial rows: {initial_rows:,}")
        print(f"  • Final rows: {final_rows:,}")
        print(f"  • Rows removed: {initial_rows - final_rows:,}")
        
        self.preprocessing_report['cleaning'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'rows_removed': initial_rows - final_rows
        }
        
        return self.df
    
    def feature_engineering(self) -> pd.DataFrame:
        """Apply feature engineering techniques."""
        print("\n" + "=" * 60)
        print("STEP 4: Feature Engineering")
        print("=" * 60)
        
        target_col = self._detect_target_column()
        
        # Check if 'Time' column exists (common in credit card datasets)
        if 'Time' in self.df.columns:
            print("\n🔧 Engineering time-based features...")
            
            # Convert time to hours of day (assuming time is in seconds from start)
            self.df['Hour'] = (self.df['Time'] / 3600) % 24
            
            # Create time periods
            self.df['TimePeriod'] = pd.cut(
                self.df['Hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                include_lowest=True
            )
            
            # One-hot encode time periods
            time_dummies = pd.get_dummies(self.df['TimePeriod'], prefix='Period')
            self.df = pd.concat([self.df, time_dummies], axis=1)
            
            # Drop intermediate columns
            self.df = self.df.drop(['TimePeriod'], axis=1)
            
            print(f"  ✓ Created 'Hour' feature")
            print(f"  ✓ Created time period indicators")
        
        # Check if 'Amount' column exists
        if 'Amount' in self.df.columns:
            print("\n🔧 Engineering amount-based features...")
            
            # Log transform of Amount (handle zeros)
            self.df['Amount_Log'] = np.log1p(self.df['Amount'])
            
            # Amount categories
            amount_quantiles = self.df['Amount'].quantile([0.25, 0.5, 0.75]).values
            self.df['Amount_Category'] = pd.cut(
                self.df['Amount'],
                bins=[-np.inf, amount_quantiles[0], amount_quantiles[1], amount_quantiles[2], np.inf],
                labels=['Low', 'Medium', 'High', 'Very_High']
            )
            
            # One-hot encode amount categories
            amount_dummies = pd.get_dummies(self.df['Amount_Category'], prefix='AmountCat')
            self.df = pd.concat([self.df, amount_dummies], axis=1)
            
            # Drop intermediate column
            self.df = self.df.drop(['Amount_Category'], axis=1)
            
            print(f"  ✓ Created 'Amount_Log' feature")
            print(f"  ✓ Created amount category indicators")
        
        print(f"\n📊 Feature Engineering Summary:")
        print(f"  • Total features: {self.df.shape[1]} (excluding target)")
        
        self.preprocessing_report['feature_engineering'] = {
            'total_features': self.df.shape[1] - 1
        }
        
        return self.df
    
    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target arrays."""
        print("\n" + "=" * 60)
        print("STEP 5: Preparing Features")
        print("=" * 60)
        
        target_col = self._detect_target_column()
        
        # Separate features and target
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"\n📊 Features prepared:")
        print(f"  • Feature matrix shape: {X.shape}")
        print(f"  • Target vector shape: {y.shape}")
        print(f"  • Feature names: {len(self.feature_names)} features")
        
        return X.values, y.values
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Split data into train, validation, and test sets with stratification."""
        print("\n" + "=" * 60)
        print("STEP 6: Train/Validation/Test Split")
        print("=" * 60)
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Second split: separate validation from training
        val_ratio = self.config.val_size / (1 - self.config.test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        print(f"\n📊 Split Summary:")
        print(f"  • Training set:   {self.X_train.shape[0]:,} samples ({(1-self.config.test_size-self.config.val_size)*100:.0f}%)")
        print(f"  • Validation set: {self.X_val.shape[0]:,} samples ({self.config.val_size*100:.0f}%)")
        print(f"  • Test set:       {self.X_test.shape[0]:,} samples ({self.config.test_size*100:.0f}%)")
        
        # Class distribution in each split
        print(f"\n⚖️ Class Distribution per Split:")
        for name, y_split in [('Train', self.y_train), ('Val', self.y_val), ('Test', self.y_test)]:
            fraud_pct = (y_split == 1).sum() / len(y_split) * 100
            print(f"  • {name}: {fraud_pct:.2f}% fraud")
        
        self.preprocessing_report['split'] = {
            'train_size': self.X_train.shape[0],
            'val_size': self.X_val.shape[0],
            'test_size': self.X_test.shape[0]
        }
    
    def scale_features(self) -> None:
        """Scale features using the configured scaler."""
        print("\n" + "=" * 60)
        print("STEP 7: Feature Scaling")
        print("=" * 60)
        
        # Select scaler
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        self.scaler = scalers.get(self.config.scaler_type, RobustScaler())
        scaler_name = type(self.scaler).__name__
        
        print(f"\n🔧 Using {scaler_name}...")
        
        # Fit on training data only, transform all sets
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"  ✓ Scaler fitted on training data")
        print(f"  ✓ All sets transformed")
        
        self.preprocessing_report['scaling'] = {
            'scaler': scaler_name
        }
    
    def handle_class_imbalance(self) -> None:
        """Handle class imbalance using various resampling techniques."""
        print("\n" + "=" * 60)
        print("STEP 8: Handling Class Imbalance")
        print("=" * 60)
        
        if not self.config.handle_imbalance:
            print("\n⏭️ Skipping class imbalance handling (disabled in config)")
            return
        
        if not IMBLEARN_AVAILABLE:
            print("\n⚠️ imbalanced-learn not installed. Skipping resampling.")
            return
        
        # Store original counts
        original_train_size = len(self.y_train)
        original_fraud = (self.y_train == 1).sum()
        original_legit = (self.y_train == 0).sum()
        
        print(f"\n📊 Before Resampling:")
        print(f"  • Legitimate: {original_legit:,}")
        print(f"  • Fraud: {original_fraud:,}")
        print(f"  • Ratio: {original_legit/original_fraud:.2f}:1")
        
        # Select resampling method
        methods = {
            'smote': SMOTE(sampling_strategy=self.config.sampling_strategy, random_state=self.config.random_state),
            'adasyn': ADASYN(sampling_strategy=self.config.sampling_strategy, random_state=self.config.random_state),
            'undersample': RandomUnderSampler(sampling_strategy=self.config.sampling_strategy, random_state=self.config.random_state),
            'smote_tomek': SMOTETomek(sampling_strategy=self.config.sampling_strategy, random_state=self.config.random_state),
            'smote_enn': SMOTEENN(sampling_strategy=self.config.sampling_strategy, random_state=self.config.random_state),
            'oversample': RandomOverSampler(sampling_strategy=self.config.sampling_strategy, random_state=self.config.random_state)
        }
        
        resampler = methods.get(self.config.imbalance_method)
        if resampler is None:
            print(f"\n⚠️ Unknown method '{self.config.imbalance_method}'. Using SMOTE.")
            resampler = SMOTE(sampling_strategy=self.config.sampling_strategy, random_state=self.config.random_state)
        
        method_name = type(resampler).__name__
        print(f"\n🔧 Applying {method_name}...")
        
        # Apply resampling only to training data
        self.X_train, self.y_train = resampler.fit_resample(self.X_train, self.y_train)
        
        # New counts
        new_fraud = (self.y_train == 1).sum()
        new_legit = (self.y_train == 0).sum()
        
        print(f"\n📊 After Resampling:")
        print(f"  • Legitimate: {new_legit:,}")
        print(f"  • Fraud: {new_fraud:,}")
        print(f"  • New Ratio: {new_legit/new_fraud:.2f}:1")
        print(f"  • Training set size: {len(self.y_train):,} (was {original_train_size:,})")
        
        self.preprocessing_report['resampling'] = {
            'method': method_name,
            'original_size': original_train_size,
            'new_size': len(self.y_train),
            'new_fraud_count': int(new_fraud),
            'new_legit_count': int(new_legit)
        }
    
    def save_processed_data(self) -> None:
        """Save processed data to files."""
        print("\n" + "=" * 60)
        print("STEP 9: Saving Processed Data")
        print("=" * 60)
        
        # Create output directory
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save numpy arrays
        np.save(output_path / 'X_train.npy', self.X_train)
        np.save(output_path / 'X_val.npy', self.X_val)
        np.save(output_path / 'X_test.npy', self.X_test)
        np.save(output_path / 'y_train.npy', self.y_train)
        np.save(output_path / 'y_val.npy', self.y_val)
        np.save(output_path / 'y_test.npy', self.y_test)
        
        print(f"\n✓ Saved processed arrays to '{output_path}':")
        print(f"  • X_train.npy: {self.X_train.shape}")
        print(f"  • X_val.npy: {self.X_val.shape}")
        print(f"  • X_test.npy: {self.X_test.shape}")
        print(f"  • y_train.npy: {self.y_train.shape}")
        print(f"  • y_val.npy: {self.y_val.shape}")
        print(f"  • y_test.npy: {self.y_test.shape}")
        
        # Save feature names
        feature_df = pd.DataFrame({'feature_name': self.feature_names})
        feature_df.to_csv(output_path / 'feature_names.csv', index=False)
        print(f"  • feature_names.csv: {len(self.feature_names)} features")
        
        # Save scaler for inference
        import joblib
        joblib.dump(self.scaler, output_path / 'scaler.joblib')
        print(f"  • scaler.joblib: {type(self.scaler).__name__}")
        
        # Save preprocessing report
        report_df = pd.DataFrame([self.preprocessing_report])
        report_df.to_json(output_path / 'preprocessing_report.json', indent=2)
        print(f"  • preprocessing_report.json")
        
        # Also save as CSV for easy loading
        train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        train_df['target'] = self.y_train
        train_df.to_csv(output_path / 'train_processed.csv', index=False)
        
        val_df = pd.DataFrame(self.X_val, columns=self.feature_names)
        val_df['target'] = self.y_val
        val_df.to_csv(output_path / 'val_processed.csv', index=False)
        
        test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        test_df['target'] = self.y_test
        test_df.to_csv(output_path / 'test_processed.csv', index=False)
        
        print(f"  • train_processed.csv")
        print(f"  • val_processed.csv")
        print(f"  • test_processed.csv")
    
    def generate_report(self) -> None:
        """Generate final preprocessing report."""
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        
        print(f"""
📋 FINAL SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Original Dataset:
  • Shape: {self.preprocessing_report.get('original_shape', 'N/A')}
  
After Cleaning:
  • Rows removed: {self.preprocessing_report.get('cleaning', {}).get('rows_removed', 0):,}
  
Feature Engineering:
  • Total features: {self.preprocessing_report.get('feature_engineering', {}).get('total_features', 'N/A')}
  
Data Split:
  • Training: {self.preprocessing_report.get('split', {}).get('train_size', 0):,} samples
  • Validation: {self.preprocessing_report.get('split', {}).get('val_size', 0):,} samples
  • Test: {self.preprocessing_report.get('split', {}).get('test_size', 0):,} samples

Scaling:
  • Method: {self.preprocessing_report.get('scaling', {}).get('scaler', 'N/A')}

Class Imbalance:
  • Method: {self.preprocessing_report.get('resampling', {}).get('method', 'N/A')}
  • Final training size: {self.preprocessing_report.get('resampling', {}).get('new_size', 'N/A'):,}

Output Directory: {self.config.output_dir}/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
    
    def run_pipeline(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute the complete preprocessing pipeline."""
        
        print("\n" + "🚀" * 30)
        print("CREDIT CARD FRAUD DETECTION - DATA PREPROCESSING PIPELINE")
        print("🚀" * 30 + "\n")
        
        # Execute pipeline steps
        self.load_data()
        self.explore_data()
        self.clean_data()
        self.feature_engineering()
        X, y = self.prepare_features()
        self.split_data(X, y)
        self.scale_features()
        self.handle_class_imbalance()
        self.save_processed_data()
        self.generate_report()
        
        return (
            self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test
        )


def load_processed_data(output_dir: str = "processed_data") -> Tuple[np.ndarray, ...]:
    """
    Utility function to load processed data for model training.
    
    Usage:
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    """
    path = Path(output_dir)
    
    X_train = np.load(path / 'X_train.npy')
    X_val = np.load(path / 'X_val.npy')
    X_test = np.load(path / 'X_test.npy')
    y_train = np.load(path / 'y_train.npy')
    y_val = np.load(path / 'y_val.npy')
    y_test = np.load(path / 'y_test.npy')
    
    print(f"✓ Loaded processed data from '{output_dir}'")
    print(f"  • X_train: {X_train.shape}")
    print(f"  • X_val: {X_val.shape}")
    print(f"  • X_test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Configure preprocessing
    config = PreprocessingConfig(
        data_path="creditcard.csv",           # Input data path
        output_dir="processed_data",          # Output directory
        test_size=0.2,                        # 20% test split
        val_size=0.1,                         # 10% validation split
        random_state=42,                      # Reproducibility
        scaler_type="robust",                 # RobustScaler (handles outliers well)
        handle_imbalance=True,                # Enable SMOTE
        imbalance_method="smote",             # SMOTE for oversampling
        sampling_strategy=0.5,                # Target 50% ratio
        remove_outliers=False,                # Keep outliers (fraud may look like outliers)
        outlier_threshold=3.0                 # Z-score threshold if enabled
    )
    
    # Initialize and run pipeline
    preprocessor = CreditCardPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.run_pipeline()
    
    # Data is now ready for ML modeling!
    print("\n✅ Data is ready for model training!")
    print(f"   Load with: X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()")