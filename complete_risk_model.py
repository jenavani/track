import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class RiskTypeModel:
    def __init__(self, data_path=None, model_path=None):
        """
        Initialize the Risk Type prediction model
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file containing the data
        model_path : str, optional
            Path to the saved model file (for loading an existing model)
        """
        self.data_path = data_path
        self.model_path = model_path
        self.df = None
        self.df_processed = None
        self.X = None
        self.y = None
        self.model = None
        self.numeric_features = None
        self.categorical_features = None
        self.preprocessor = None
        self.feature_names = None
        
        # If model path is provided, load the model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_data(self):
        """Load and prepare the dataset"""
        if not self.data_path:
            raise ValueError("Data path not provided")
        
        try:
            # Load data
            self.df = pd.read_csv(self.data_path, sep=';')
            print(f"Dataset loaded with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
            
            # Display basic information
            print("\nTarget variable distribution:")
            print(self.df['Type_risk'].value_counts())
            print("\nTarget variable distribution (%):")
            print(self.df['Type_risk'].value_counts(normalize=True) * 100)
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Preprocess the data by handling dates, missing values, and creating new features"""
        if self.df is None:
            raise ValueError("Data not loaded yet")
        
        try:
            # Create a copy of the dataframe
            self.df_processed = self.df.copy()
            
            # Convert date columns to datetime
            date_columns = ['Date_start_contract', 'Date_last_renewal', 'Date_next_renewal', 
                          'Date_birth', 'Date_driving_licence', 'Date_lapse']
            
            for col in date_columns:
                if col in self.df_processed.columns:
                    self.df_processed[col] = pd.to_datetime(self.df_processed[col], format='%d/%m/%Y', errors='coerce')
            
            # Calculate age from birth date
            current_year = datetime.now().year
            if 'Date_birth' in self.df_processed.columns:
                self.df_processed['Age'] = current_year - self.df_processed['Date_birth'].dt.year
            
            # Calculate driving experience
            if 'Date_driving_licence' in self.df_processed.columns:
                self.df_processed['Driving_Experience'] = current_year - self.df_processed['Date_driving_licence'].dt.year
            
            # Calculate vehicle age
            if 'Year_matriculation' in self.df_processed.columns:
                self.df_processed['Vehicle_Age'] = current_year - self.df_processed['Year_matriculation']
            
            # Calculate contract duration
            if all(col in self.df_processed.columns for col in ['Date_start_contract', 'Date_next_renewal']):
                self.df_processed['Contract_Duration_Days'] = (
                    self.df_processed['Date_next_renewal'] - self.df_processed['Date_start_contract']
                ).dt.days
            
            # Handle 'NA' values in Length and Weight columns
            if 'Length' in self.df_processed.columns:
                self.df_processed['Length'] = pd.to_numeric(
                    self.df_processed['Length'].replace('NA', np.nan), errors='coerce'
                )
            
            if 'Weight' in self.df_processed.columns:
                self.df_processed['Weight'] = pd.to_numeric(self.df_processed['Weight'], errors='coerce')
            
            # Create policy to claims ratio features if possible
            if all(col in self.df_processed.columns for col in ['N_claims_year', 'Policies_in_force']):
                self.df_processed['Claims_per_Policy'] = (
                    self.df_processed['N_claims_year'] / 
                    self.df_processed['Policies_in_force'].replace(0, 1)  # Avoid division by zero
                )
            
            # Create premium to value ratio for vehicle policies
            if all(col in self.df_processed.columns for col in ['Premium', 'Value_vehicle']):
                self.df_processed['Premium_Value_Ratio'] = (
                    self.df_processed['Premium'] / 
                    self.df_processed['Value_vehicle'].replace(0, 1)  # Avoid division by zero
                ) * 1000  # Scale factor for better readability
            
            return True
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return False
    
    def prepare_for_modeling(self):
        """Prepare data for modeling by selecting features and preparing X and y"""
        if self.df_processed is None:
            raise ValueError("Data not preprocessed yet")
        
        try:
            # Columns to drop
            columns_to_drop = [
                'ID',  # Identifier
                'Date_start_contract', 'Date_last_renewal', 'Date_next_renewal',
                'Date_birth', 'Date_driving_licence', 'Date_lapse',
                'Length'  # Too many missing or NA values
            ]
            
            # Drop unnecessary columns
            df_model = self.df_processed.drop(
                columns=[col for col in columns_to_drop if col in self.df_processed.columns], 
                errors='ignore'
            )
            
            # Convert categorical columns to string type to avoid mixed type issues
            categorical_columns = [
                'Distribution_channel', 'Payment', 'Lapse', 'Area', 
                'Second_driver', 'Type_fuel'
            ]
            
            for col in categorical_columns:
                if col in df_model.columns:
                    df_model[col] = df_model[col].astype(str)
            
            # Identify numeric and categorical features
            self.numeric_features = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.categorical_features = df_model.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            
            # Remove the target from feature lists
            if 'Type_risk' in self.numeric_features:
                self.numeric_features.remove('Type_risk')
            
            # Prepare X and y
            self.X = df_model.drop(columns=['Type_risk'])
            self.y = df_model['Type_risk']
            
            print(f"\nFeatures prepared for modeling:")
            print(f"  - Numeric features ({len(self.numeric_features)}): {self.numeric_features}")
            print(f"  - Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
            print(f"  - Total samples: {len(self.X)}")
            
            return True
        except Exception as e:
            print(f"Error preparing data for modeling: {e}")
            return False
    
    def build_preprocessor(self):
        """Build the data preprocessing pipeline"""
        # Create preprocessor for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough'
        )
        
        return self.preprocessor
    
    def train_model(self, model_type='rf', cv=5, save_path='risk_type_model.pkl'):
        """
        Train the machine learning model
        
        Parameters:
        -----------
        model_type : str, default='rf'
            Type of model to train ('rf' for Random Forest, 'gb' for Gradient Boosting)
        cv : int, default=5
            Number of cross-validation folds
        save_path : str, default='risk_type_model.pkl'
            Path to save the trained model
        """
        if self.X is None or self.y is None:
            raise ValueError("Features and target not prepared yet")
        
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.25, random_state=42, stratify=self.y
            )
            
            # Create preprocessor if not already created
            if self.preprocessor is None:
                self.build_preprocessor()
            
            # Select the model
            if model_type == 'rf':
                classifier = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10, 
                    min_samples_split=2, 
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'gb':
                classifier = GradientBoostingClassifier(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=5, 
                    random_state=42
                )
            else:
                raise ValueError(f"Invalid model type: {model_type}")
            
            # Create a pipeline with preprocessing and model
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', classifier)
            ])
            
            # Cross-validation
            print("\nPerforming cross-validation...")
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), 
                scoring='accuracy'
            )
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Train the full model
            print("\nTraining the final model...")
            self.model = pipeline
            self.model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\nTest set accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png')
            plt.close()
            
            # Save the model
            self.save_model(save_path)
            
            # Analyze feature importance
            self.analyze_feature_importance()
            
            return accuracy
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def analyze_feature_importance(self):
        """Analyze and plot feature importance"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            # Check if model has feature_importances_ attribute (Random Forest and Gradient Boosting do)
            if hasattr(self.model[-1], 'feature_importances_'):
                feature_importance = self.model[-1].feature_importances_
                
                # Get feature names after preprocessing
                # For preprocessed data, we can't easily get the exact feature names
                # So we'll use generic names
                n_features = len(feature_importance)
                feature_names = [f'Feature_{i}' for i in range(n_features)]
                
                # Create DataFrame for easier handling
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Plot top 20 features
                plt.figure(figsize=(12, 8))
                plt.title('Top 20 Feature Importances')
                top_features = importance_df.head(20)
                sns.barplot(x='Importance', y='Feature', data=top_features)
                plt.tight_layout()
                plt.savefig('feature_importance.png')
                plt.close()
                
                print(f"\nTop 10 most important features:")
                for i, row in importance_df.head(10).iterrows():
                    print(f"  - {row['Feature']}: {row['Importance']:.4f}")
                
                print("\nFeature importance plot saved to 'feature_importance.png'")
                
                return importance_df
            else:
                print("Model doesn't provide feature importance information")
                return None
        except Exception as e:
            print(f"Error analyzing feature importance: {e}")
            return None
    
    def save_model(self, file_path='risk_type_model.pkl'):
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            # Create a dictionary with the model and important metadata
            model_data = {
                'model': self.model,
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features,
                'model_info': {
                    'target': 'Type_risk',
                    'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            # Save to disk
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"\nModel saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, file_path):
        """Load a saved model from disk"""
        try:
            # Load the model data
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract components
            self.model = model_data['model']
            self.numeric_features = model_data.get('numeric_features', [])
            self.categorical_features = model_data.get('categorical_features', [])
            
            print(f"Model loaded from {file_path}")
            
            # Print model information if available
            if 'model_info' in model_data:
                info = model_data['model_info']
                print(f"Target variable: {info.get('target', 'Unknown')}")
                print(f"Model created on: {info.get('created_date', 'Unknown')}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, data, return_probabilities=False):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        data : DataFrame
            Data to make predictions on
        return_probabilities : bool, default=False
            Whether to return class probabilities
        
        Returns:
        --------
        DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded yet")
        
        try:
            # Create a copy of the data
            df_pred = data.copy()
            
            # Preprocess the data (similar to preprocess_data but for new data)
            # Convert date columns to datetime
            date_columns = ['Date_start_contract', 'Date_last_renewal', 'Date_next_renewal', 
                          'Date_birth', 'Date_driving_licence', 'Date_lapse']
            
            for col in date_columns:
                if col in df_pred.columns:
                    df_pred[col] = pd.to_datetime(df_pred[col], format='%d/%m/%Y', errors='coerce')
            
            # Calculate age from birth date
            current_year = datetime.now().year
            if 'Date_birth' in df_pred.columns:
                df_pred['Age'] = current_year - df_pred['Date_birth'].dt.year
            
            # Calculate driving experience
            if 'Date_driving_licence' in df_pred.columns:
                df_pred['Driving_Experience'] = current_year - df_pred['Date_driving_licence'].dt.year
            
            # Calculate vehicle age
            if 'Year_matriculation' in df_pred.columns:
                df_pred['Vehicle_Age'] = current_year - df_pred['Year_matriculation']
            
            # Calculate contract duration
            if all(col in df_pred.columns for col in ['Date_start_contract', 'Date_next_renewal']):
                df_pred['Contract_Duration_Days'] = (
                    df_pred['Date_next_renewal'] - df_pred['Date_start_contract']
                ).dt.days
            
            # Create policy to claims ratio features if possible
            if all(col in df_pred.columns for col in ['N_claims_year', 'Policies_in_force']):
                df_pred['Claims_per_Policy'] = (
                    df_pred['N_claims_year'] / 
                    df_pred['Policies_in_force'].replace(0, 1)
                )
            
            # Create premium to value ratio for vehicle policies
            if all(col in df_pred.columns for col in ['Premium', 'Value_vehicle']):
                df_pred['Premium_Value_Ratio'] = (
                    df_pred['Premium'] / 
                    df_pred['Value_vehicle'].replace(0, 1)
                ) * 1000
            
            # Convert categorical columns to string
            for col in self.categorical_features:
                if col in df_pred.columns:
                    df_pred[col] = df_pred[col].astype(str)
            
            # Prepare X for prediction
            # Select only the columns used for training (numeric and categorical features)
            prediction_columns = self.numeric_features + self.categorical_features
            X_pred = df_pred[prediction_columns].copy()
            
            # Make predictions
            predictions = self.model.predict(X_pred)
            
            # Add predictions to the original data
            df_pred['Predicted_Type_risk'] = predictions
            
            # Add class probabilities if requested
            if return_probabilities:
                probabilities = self.model.predict_proba(X_pred)
                for i, class_label in enumerate(self.model.classes_):
                    df_pred[f'Probability_Class_{class_label}'] = probabilities[:, i]
            
            return df_pred
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def evaluate_on_test_data(self, test_data_path, true_label_col='Type_risk'):
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        test_data_path : str
            Path to the test data file
        true_label_col : str, default='Type_risk'
            Name of the column with true labels
        
        Returns:
        --------
        float : Accuracy on test data
        """
        if self.model is None:
            raise ValueError("Model not loaded yet")
        
        try:
            # Load test data
            test_df = pd.read_csv(test_data_path, sep=';')
            print(f"Test data loaded with {test_df.shape[0]} rows")
            
            # Check if true label column exists
            if true_label_col not in test_df.columns:
                raise ValueError(f"True label column '{true_label_col}' not found in test data")
            
            # Make predictions with probabilities
            predictions_df = self.predict(test_df, return_probabilities=True)
            
            # Extract true labels and predictions
            y_true = test_df[true_label_col]
            y_pred = predictions_df['Predicted_Type_risk']
            
            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred)
            
            # Generate classification report
            print("\nTest Data Evaluation Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred))
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix (Test Data)')
            plt.savefig('test_confusion_matrix.png')
            plt.close()
            
            return accuracy
        except Exception as e:
            print(f"Error evaluating model on test data: {e}")
            return None


def main():
    """Main function to demonstrate the full workflow"""
    # Set the data path
    data_path = 'data.csv'
    
    # Create the model
    print("Initializing Risk Type Prediction Model...")
    model = RiskTypeModel(data_path=data_path)
    
    # Load and preprocess data
    print("\nLoading data...")
    if model.load_data():
        print("\nPreprocessing data...")
        if model.preprocess_data():
            print("\nPreparing data for modeling...")
            if model.prepare_for_modeling():
                # Train the model
                print("\nTraining model...")
                model.train_model(model_type='rf', cv=3, save_path='risk_type_model.pkl')
                
                # If there's test data, evaluate on it
                test_data_path = input("\nEnter path to test data (leave empty to skip): ")
                if test_data_path and os.path.exists(test_data_path):
                    model.evaluate_on_test_data(test_data_path)
                
                print("\nCompleted! The model is ready for use.")


# Demonstrate how to use the trained model for predictions
def load_and_predict():
    """Load a saved model and make predictions on new data"""
    # Path to the saved model
    model_path = 'risk_type_model.pkl'
    
    # Path to new data for prediction
    new_data_path = input("Enter path to new data for prediction: ")
    
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train the model first.")
        return
    
    # Check if the new data exists
    if not os.path.exists(new_data_path):
        print(f"Data file not found at {new_data_path}.")
        return
    
    # Load the model
    model = RiskTypeModel(model_path=model_path)
    
    # Load new data
    try:
        new_data = pd.read_csv(new_data_path, sep=';')
        print(f"Loaded new data with {new_data.shape[0]} rows and {new_data.shape[1]} columns")
    except Exception as e:
        print(f"Error loading new data: {e}")
        return
    
    # Make predictions with probabilities
    predictions = model.predict(new_data, return_probabilities=True)
    
    # Display sample predictions
    print("\nSample predictions:")
    display_cols = ['ID', 'Predicted_Type_risk'] + [col for col in predictions.columns if 'Probability_Class_' in col]
    display_cols = [col for col in display_cols if col in predictions.columns]
    print(predictions[display_cols].head())
    
    # Save predictions
    output_path = input("Enter path to save predictions (default: risk_predictions.csv): ") or 'risk_predictions.csv'
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # Choose mode
    mode = input("Choose mode (1: Train new model, 2: Load model and predict): ")
    
    if mode == '1':
        main()
    elif mode == '2':
        load_and_predict()
    else:
        print("Invalid mode. Please choose 1 or 2.")