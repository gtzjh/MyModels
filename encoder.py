"""
Category variable encoding module
Supported methods: OneHot, Label, Target, Frequency, Binary, Ordinal

Note:
1. In machine learning, the encoder from the training set should also be used during the testing phase to avoid data leakage.
2. Similarly, during cross-validation, a separate encoder should be constructed for each fold.
"""


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import category_encoders as ce
import joblib
import os
import json
import numpy as np


class Encoder:
    def __init__(self, method='onehot', target_col=None):
        """
        :param method: Encoding method ['onehot','label','target','frequency','binary','ordinal']
        :param target_col: Target column name required for target encoding
        """
        self.version = '1.0.0'
        self.VALID_METHODS = ['onehot', 'label', 'target', 'frequency', 'binary', 'ordinal']
        
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method. Choose from {self.VALID_METHODS}")
        if method == 'target' and target_col is None:
            raise ValueError("target_col must be specified for target encoding")

        self.method = method
        self.target_col = target_col
        self.encoder = None
        self.feature_names = None

    def fit(self, df, cat_cols):
        """Fit the encoder"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(cat_cols, (list, tuple)):
            raise TypeError("cat_cols must be a list or tuple")
        
        if df[cat_cols].isnull().any().any():
            raise ValueError("Input contains null values. Please handle missing values before encoding.")

        if not cat_cols:
            raise ValueError("cat_cols cannot be empty")
        missing_cols = [col for col in cat_cols if col not in df]
        if missing_cols:
            raise KeyError(f"Columns {missing_cols} not found in dataframe")

        if self.method == 'onehot':
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.encoder.fit(df[cat_cols])
            
        elif self.method == 'label':
            self.encoder = {col: LabelEncoder() for col in cat_cols}
            for col in cat_cols:
                self.encoder[col].fit(df[col])
                
        elif self.method == 'target':
            if self.target_col not in df.columns:
                raise KeyError(f"Target column {self.target_col} not found in dataframe")
            self.encoder = ce.TargetEncoder(cols=cat_cols)
            self.encoder.fit(df[cat_cols], df[self.target_col])
            
        elif self.method == 'frequency':
            self.encoder = {col: df[col].value_counts(normalize=True).to_dict() 
                          for col in cat_cols}
            
        elif self.method == 'binary':
            self.encoder = ce.BinaryEncoder(cols=cat_cols)
            self.encoder.fit(df[cat_cols])
            
        elif self.method == 'ordinal':
            self.encoder = ce.OrdinalEncoder(cols=cat_cols)
            self.encoder.fit(df[cat_cols])
            
        return self

    def transform(self, df, cat_cols):
        """Apply encoding"""
        if self.encoder is None:
            raise RuntimeError("Encoder not fitted. Call fit() first")

        missing_cols = [col for col in cat_cols if col not in df]
        if missing_cols:
            raise KeyError(f"Columns {missing_cols} not found in dataframe")

        df = df.copy()
        
        if len(df) * len(cat_cols) > 1e7:  # Adjust threshold based on actual needs
            import warnings
            warnings.warn("Large dataset detected. This operation may consume significant memory.")
        
        if self.method == 'onehot':
            encoded = self.encoder.transform(df[cat_cols])
            encoded_df = pd.DataFrame(encoded,
                                    columns=self.encoder.get_feature_names_out(cat_cols),
                                    index=df.index)
            df = pd.concat([df.drop(cat_cols, axis=1), encoded_df], axis=1)
            
        elif self.method == 'label':
            for col in cat_cols:
                df[col] = self.encoder[col].transform(df[col])
                
        elif self.method == 'target':
            df[cat_cols] = self.encoder.transform(df[cat_cols])
            
        elif self.method == 'frequency':
            for col in cat_cols:
                df[col+'_freq'] = df[col].map(self.encoder[col]).fillna(0)
            df = df.drop(cat_cols, axis=1)
        
        elif self.method == 'binary':
            binary_cols = self.encoder.transform(df[cat_cols])
            # Delete the original columns, and concatenate the new binary columns
            df = pd.concat([df.drop(cat_cols, axis=1), binary_cols], axis=1)
            
        elif self.method in ['ordinal']:
            df[cat_cols] = self.encoder.transform(df[cat_cols])
            
        return df

    def save(self, path='encoding/encoder.pkl'):
        """Save the encoder"""
        if os.path.exists(path):
            import warnings
            warnings.warn(f"File {path} already exists and will be overwritten")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            joblib.dump({
                'method': self.method,
                'target_col': self.target_col,
                'encoder': self.encoder,
                'feature_names': self.feature_names
            }, path)
        except Exception as e:
            raise IOError(f"Failed to save encoder: {str(e)}")

    @classmethod
    def load(cls, path='encoding/encoder.pkl'):
        """Load the encoder"""
        data = joblib.load(path)
        obj = cls(method=data['method'], target_col=data['target_col'])
        obj.encoder = data['encoder']
        obj.feature_names = data['feature_names']
        return obj

    def validate_encoding(self, encoded_df, original_df):
        """Validate the encoding results"""
        # Check if number of rows matches
        if len(encoded_df) != len(original_df):
            raise ValueError("Number of rows changed after encoding")
        
        # Check for invalid values
        if encoded_df.isnull().any().any():
            raise ValueError("Encoding produced null values")
        
        # For onehot encoding, verify binary values
        if self.method == 'onehot':
            encoded_cols = [col for col in encoded_df.columns 
                           if col not in original_df.columns]
            if not encoded_df[encoded_cols].isin([0, 1]).all().all():
                raise ValueError("OneHot encoding produced non-binary values")

    def get_mapping(self, df, cat_cols):
        """
        Get encoding mapping relationships
        
        :param df: Original dataframe
        :param cat_cols: Category columns to get mapping for
        :return: Dictionary containing mapping relationships
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not fitted. Call fit() first")
        
        mapping = {}
        
        if self.method == 'onehot':
            for col in cat_cols:
                unique_values = df[col].unique()
                encoded = self.encoder.transform(df[cat_cols].loc[df[col].isin(unique_values)].head(len(unique_values)))
                feature_names = self.encoder.get_feature_names_out()
                col_mapping = {
                    val: dict(zip(feature_names, row))
                    for val, row in zip(unique_values, encoded)
                }
                mapping[col] = col_mapping
            
        elif self.method == 'label':
            for col in cat_cols:
                label_encoder = self.encoder[col]
                mapping[col] = dict(zip(label_encoder.classes_, 
                                      label_encoder.transform(label_encoder.classes_)))
            
        elif self.method == 'target':
            # Target encoding mapping is dynamic, return mean target value for each category
            for col in cat_cols:
                mapping[col] = df.groupby(col)[self.target_col].mean().to_dict()
            
        elif self.method == 'frequency':
            mapping = self.encoder  # frequency encoder itself stores the mapping
            
        elif self.method == 'binary':
            # For binary encoding, show binary representation for each category value
            for col in cat_cols:
                unique_values = df[col].unique()
                temp_df = pd.DataFrame({c: [unique_values[0]] * len(unique_values) for c in cat_cols})
                temp_df[col] = unique_values
                encoded = self.encoder.transform(temp_df)
                binary_cols = [c for c in encoded.columns if c.startswith(f"{col}_")]
                mapping[col] = {
                    val: dict(zip(binary_cols, row))
                    for val, row in zip(unique_values, encoded[binary_cols].values)
                }
            
        elif self.method == 'ordinal':
            # For ordinal encoding, return category to ordinal mapping
            for col in cat_cols:
                # Get unique values from the original data
                unique_values = df[col].unique()
                # Create a temporary dataframe with all category columns
                temp_df = pd.DataFrame({c: [unique_values[0]] * len(unique_values) for c in cat_cols})
                # Set the current column's values
                temp_df[col] = unique_values
                # Transform the values to get their ordinal encoding
                encoded_values = self.encoder.transform(temp_df)[col].values
                # Create mapping dictionary
                mapping[col] = dict(zip(unique_values, encoded_values))
        
        return mapping

    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {self.convert_numpy_types(key): self.convert_numpy_types(value) 
                    for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_numpy_types(item) for item in obj]
        return obj


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data.csv')
    
    # Create encoding directory if it doesn't exist
    os.makedirs('encoding', exist_ok=True)
    
    # Test all encoding methods and save mapping relationships
    cat_cols = ['x16', 'x17']
    
    # Define encoding methods to test
    methods = ['onehot', 'label', 'frequency', 'binary', 'ordinal']
    
    for method in methods:
        # Initialize encoder
        encoder = Encoder(method=method)
        
        # Fit data
        encoder.fit(df, cat_cols)
        
        # Get mapping relationships
        mapping = encoder.get_mapping(df, cat_cols)
        
        # Convert numpy types to native Python types
        mapping = encoder.convert_numpy_types(mapping)
        
        # Save to JSON file
        json_path = f'encoding/{method}_mapping.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)
        
        print(f"Saved {method} encoding mapping to {json_path}")
    
    # Handle target encoding separately as it requires target column
    if 'target' in df.columns:  # Assuming target column name is 'target'
        encoder = Encoder(method='target', target_col='target')
        encoder.fit(df, cat_cols)
        mapping = encoder.get_mapping(df, cat_cols)
        
        # Convert numpy types to native Python types
        mapping = encoder.convert_numpy_types(mapping)
        
        json_path = 'encoding/target_mapping.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)
        
        print(f"Saved target encoding mapping to {json_path}")