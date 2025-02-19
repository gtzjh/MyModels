"""
Category variable encoding module
Supported methods: OneHot, Label, Target, Frequency, Binary, Ordinal
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import category_encoders as ce
import joblib

class Encoder:
    def __init__(self, method='onehot', target_col=None):
        """
        :param method: Encoding method ['onehot','label','target','frequency','binary','ordinal']
        :param target_col: Target column name required for target encoding
        """
        self.method = method
        self.target_col = target_col
        self.encoder = None
        self.feature_names = None

    def fit(self, df, cat_cols):
        """Fit the encoder"""
        if self.method == 'onehot':
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.encoder.fit(df[cat_cols])
            
        elif self.method == 'label':
            self.encoder = {col: LabelEncoder() for col in cat_cols}
            for col in cat_cols:
                self.encoder[col].fit(df[col])
                
        elif self.method == 'target':
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

    def transform(self, df, cat_cols=['x16', 'x17']):
        """Apply encoding"""
        if self.method == 'onehot':
            encoded = self.encoder.transform(df[cat_cols])
            df = pd.concat([df.drop(cat_cols, axis=1), encoded], axis=1)
            
        elif self.method == 'label':
            for col in cat_cols:
                df[col] = self.encoder[col].transform(df[col])
                
        elif self.method == 'target':
            df = self.encoder.transform(df)
            
        elif self.method == 'frequency':
            for col in cat_cols:
                df[col+'_freq'] = df[col].map(self.encoder[col])
            df = df.drop(cat_cols, axis=1)
            
        elif self.method in ['binary', 'ordinal']:
            df = self.encoder.transform(df)
            
        return df

    def save(self, path='model/encoder.pkl'):
        """Save the encoder"""
        joblib.dump(self.encoder, path)

    @classmethod
    def load(cls, path='model/encoder.pkl'):
        """Load the encoder"""
        return joblib.load(path)

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data.csv')
    
    # Initialize encoder
    encoder = Encoder(method='target', target_col='y')
    
    # Fit and transform
    df_encoded = encoder.fit(df, ['x16', 'x17']).transform(df)
    
    # Save encoder
    encoder.save()

    print(df_encoded.head())