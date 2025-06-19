"""
Risk Type Prediction - Usage Examples

This script demonstrates various ways to use the RiskTypeModel class for common tasks.
"""

from complete_risk_model import RiskTypeModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def example_1_train_new_model():
    """Example 1: Train a new model from scratch"""
    print("\n=== Example 1: Training a new model ===")
    
    # Initialize model with data path
    model = RiskTypeModel(data_path='data.csv')
    
    # Load and process data
    model.load_data()
    model.preprocess_data()
    model.prepare_for_modeling()
    
    # Train model with default Random Forest
    model.train_model(save_path='my_risk_model.pkl')
    
    print("Model training complete!")

def example_2_load_and_predict():
    """Example 2: Load an existing model and make predictions"""
    print("\n=== Example 2: Loading model and making predictions ===")
    
    # Load the saved model
    model = RiskTypeModel(model_path='risk_type_model.pkl')
    
    # Load data for prediction
    new_data = pd.read_csv('data.csv', sep=';')
    print(f"Loaded data with {new_data.shape[0]} rows")
    
    # Use a sample of the data for prediction
    sample_data = new_data.sample(5, random_state=42)
    
    # Make predictions with probabilities
    predictions = model.predict(sample_data, return_probabilities=True)
    
    # Display results
    print("\nSample predictions:")
    columns_to_show = ['ID', 'Predicted_Type_risk'] + [col for col in predictions.columns if 'Probability_Class_' in col]
    print(predictions[columns_to_show])
    
    print("Prediction complete!")

def example_3_compare_models():
    """Example 3: Compare different model types"""
    print("\n=== Example 3: Comparing different models ===")
    
    # Initialize model with data path
    model = RiskTypeModel(data_path='data.csv')
    
    # Load and process data
    model.load_data()
    model.preprocess_data()
    model.prepare_for_modeling()
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    rf_accuracy = model.train_model(model_type='rf', save_path='rf_model.pkl')
    
    # Train Gradient Boosting model
    print("\nTraining Gradient Boosting model...")
    gb_accuracy = model.train_model(model_type='gb', save_path='gb_model.pkl')
    
    # Compare results
    print("\nModel Comparison:")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.bar(['Random Forest', 'Gradient Boosting'], [rf_accuracy, gb_accuracy])
    plt.ylim(0.8, 1.0)  # Adjust as needed
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.savefig('model_comparison.png')
    
    print("Model comparison complete! Results saved to 'model_comparison.png'")

def example_4_feature_engineering():
    """Example 4: Demonstrate advanced feature engineering"""
    print("\n=== Example 4: Advanced Feature Engineering ===")
    
    # Load data
    df = pd.read_csv('data.csv', sep=';')
    print(f"Loaded data with {df.shape[0]} rows")
    
    # Convert date columns to datetime
    date_columns = ['Date_start_contract', 'Date_last_renewal', 'Date_next_renewal', 
                  'Date_birth', 'Date_driving_licence']
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
    
    # Create advanced features
    
    # 1. Age bands
    if 'Date_birth' in df.columns:
        df['Age'] = 2024 - df['Date_birth'].dt.year
        df['Age_Band'] = pd.cut(df['Age'], 
                             bins=[0, 25, 35, 45, 55, 65, 100],
                             labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])
    
    # 2. Vehicle power to weight ratio
    if 'Power' in df.columns and 'Weight' in df.columns:
        df['Power_Weight_Ratio'] = df['Power'] / df['Weight'].replace(0, np.nan)
        df['Power_Weight_Ratio'] = df['Power_Weight_Ratio'].fillna(df['Power_Weight_Ratio'].median())
    
    # 3. Claims frequency
    if 'N_claims_history' in df.columns and 'Seniority' in df.columns:
        df['Claims_per_Year'] = df['N_claims_history'] / df['Seniority'].replace(0, 1)
    
    # 4. Premium per policy
    if 'Premium' in df.columns and 'Policies_in_force' in df.columns:
        df['Premium_per_Policy'] = df['Premium'] / df['Policies_in_force'].replace(0, 1)
    
    # 5. Vehicle age bands
    if 'Year_matriculation' in df.columns:
        df['Vehicle_Age'] = 2024 - df['Year_matriculation']
        df['Vehicle_Age_Band'] = pd.cut(df['Vehicle_Age'],
                                    bins=[0, 3, 6, 10, 15, 100],
                                    labels=['New', '3-6', '6-10', '10-15', '15+'])
    
    # Analyze feature relationships with Type_risk
    if 'Type_risk' in df.columns:
        # Plot relationship between vehicle age and risk type
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Type_risk', y='Vehicle_Age', data=df)
        plt.title('Vehicle Age by Risk Type')
        plt.savefig('vehicle_age_by_risk.png')
        plt.close()
        
        # Plot relationship between power and risk type
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Type_risk', y='Power', data=df)
        plt.title('Vehicle Power by Risk Type')
        plt.savefig('vehicle_power_by_risk.png')
        plt.close()
        
        # Plot age band distribution by risk type
        if 'Age_Band' in df.columns:
            plt.figure(figsize=(12, 8))
            sns.countplot(data=df, x='Age_Band', hue='Type_risk')
            plt.title('Age Band Distribution by Risk Type')
            plt.savefig('age_band_by_risk.png')
            plt.close()
    
    print("Advanced feature engineering complete!")
    print("Created features: Age_Band, Power_Weight_Ratio, Claims_per_Year, Premium_per_Policy, Vehicle_Age_Band")
    print("Visualizations saved as PNG files")

def example_5_model_deployment():
    """Example 5: Demonstrate model deployment workflow"""
    print("\n=== Example 5: Model Deployment Workflow ===")
    
    # Step 1: Load the trained model
    print("Step 1: Loading model...")
    model = RiskTypeModel(model_path='risk_type_model.pkl')
    
    # Step 2: Create a simplified prediction function
    def predict_risk_type(new_data):
        """Simplified prediction function for deployment"""
        # Preprocess the input data
        predictions = model.predict(new_data, return_probabilities=True)
        
        # Format the output in a clean format
        results = []
        for i, row in predictions.iterrows():
            result = {
                'id': row.get('ID', i),
                'predicted_risk_type': int(row['Predicted_Type_risk']),
                'probabilities': {}
            }
            
            # Add probabilities for each class
            for col in predictions.columns:
                if 'Probability_Class_' in col:
                    class_num = col.split('_')[-1]
                    result['probabilities'][class_num] = float(row[col])
            
            results.append(result)
        
        return results
    
    # Step 3: Demonstrate with sample data
    print("Step 2: Creating sample prediction API...")
    
    # Sample data
    sample_data = pd.read_csv('data.csv', sep=';').sample(3, random_state=42)
    
    # Make predictions
    print("Step 3: Making predictions...")
    prediction_results = predict_risk_type(sample_data)
    
    # Show formatted output
    print("\nAPI Response Format:")
    import json
    print(json.dumps(prediction_results[0], indent=2))
    
    print("\nDeployment workflow complete!")

def example_6_batch_processing():
    """Example 6: Batch processing for large datasets"""
    print("\n=== Example 6: Batch Processing ===")
    
    # Load model
    model = RiskTypeModel(model_path='risk_type_model.pkl')
    
    # Load data
    try:
        data = pd.read_csv('data.csv', sep=';')
        print(f"Loaded {data.shape[0]} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Define batch size
    batch_size = 10000
    
    # Calculate number of batches
    n_batches = (data.shape[0] // batch_size) + (1 if data.shape[0] % batch_size != 0 else 0)
    
    print(f"Processing data in {n_batches} batches of {batch_size} records each")
    
    # Create empty DataFrame for results
    all_predictions = []
    
    # Process in batches
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, data.shape[0])
        
        print(f"Processing batch {i+1}/{n_batches} (records {start_idx} to {end_idx})")
        
        # Get current batch
        batch = data.iloc[start_idx:end_idx].copy()
        
        # Make predictions
        batch_predictions = model.predict(batch)
        
        # Store only necessary columns
        columns_to_keep = ['ID', 'Predicted_Type_risk']
        batch_results = batch_predictions[columns_to_keep]
        
        # Append to results
        all_predictions.append(batch_results)
    
    # Combine all batches
    final_predictions = pd.concat(all_predictions)
    
    # Save results
    final_predictions.to_csv('batch_predictions.csv', index=False)
    
    print(f"Batch processing complete! Results saved to 'batch_predictions.csv'")
    print(f"Processed {final_predictions.shape[0]} records in total")

if __name__ == "__main__":
    # Show menu
    print("Risk Type Model Usage Examples")
    print("------------------------------")
    print("1. Train new model")
    print("2. Load model and make predictions")
    print("3. Compare different model types")
    print("4. Advanced feature engineering")
    print("5. Model deployment workflow")
    print("6. Batch processing for large datasets")
    
    # Get user choice
    choice = input("\nSelect an example to run (1-6) or 'all' to run all: ")
    
    if choice == '1':
        example_1_train_new_model()
    elif choice == '2':
        example_2_load_and_predict()
    elif choice == '3':
        example_3_compare_models()
    elif choice == '4':
        example_4_feature_engineering()
    elif choice == '5':
        example_5_model_deployment()
    elif choice == '6':
        example_6_batch_processing()
    elif choice.lower() == 'all':
        example_1_train_new_model()
        example_2_load_and_predict()
        example_3_compare_models()
        example_4_feature_engineering()
        example_5_model_deployment()
        example_6_batch_processing()
    else:
        print("Invalid choice. Please select a number from 1 to 6 or 'all'.")