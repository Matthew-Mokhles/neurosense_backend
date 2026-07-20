#!/usr/bin/env python3
"""
XGBoost + Random Forest Enhanced Autism Screening Model
Advanced ensemble with XGBoost and Random Forest
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def _patch_numpy2_pickle_compat():
    """Map numpy._core so NumPy 1.x can load pickles saved with NumPy 2.x."""
    import sys
    if 'numpy._core' in sys.modules:
        return
    if hasattr(np, '_core'):
        return
    core = np.core
    sys.modules['numpy._core'] = core
    sys.modules['numpy._core.multiarray'] = core.multiarray
    sys.modules['numpy._core.umath'] = core.umath


def _patch_xgb_sklearn_compat(estimator):
    """Backfill attrs removed in newer XGBoost pickles for older xgboost sklearn wrappers."""
    try:
        import xgboost as xgb
    except ImportError:
        return

    from inspect import signature

    try:
        needs_label_encoder = 'use_label_encoder' in signature(xgb.XGBClassifier.__init__).parameters
    except (TypeError, ValueError):
        needs_label_encoder = False

    if not needs_label_encoder:
        return

    def _fix(model):
        if isinstance(model, xgb.XGBClassifier):
            if not hasattr(model, 'use_label_encoder'):
                model.use_label_encoder = False

    _fix(estimator)
    if hasattr(estimator, 'estimators'):
        for _, est in estimator.estimators:
            _fix(est)
    if hasattr(estimator, 'estimators_'):
        for est in estimator.estimators_:
            _fix(est)


class XGBoostRandomForestModel:
    """
    XGBoost + Random Forest Enhanced Autism Screening Model
    """
    
    def __init__(self, use_xgboost=True, use_random_forest=True, 
                 use_ensemble=True, random_state=42):
        """
        Initialize the XGBoost + Random Forest model
        
        Args:
            use_xgboost (bool): Whether to use XGBoost
            use_random_forest (bool): Whether to use Random Forest
            use_ensemble (bool): Whether to use ensemble of both
            random_state (int): Random state for reproducibility
        """
        self.use_xgboost = use_xgboost
        self.use_random_forest = use_random_forest
        self.use_ensemble = use_ensemble
        self.random_state = random_state
        
        # Model components
        self.xgb_model = None
        self.rf_model = None
        self.ensemble_model = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        # Performance tracking
        self.performance_history = []
        self.feature_importance_history = []
        
    def preprocess_data(self, df, is_training=True):
        """
        Enhanced preprocessing with scaling and feature engineering
        
        Args:
            df (DataFrame): Input data
            is_training (bool): Whether this is training data
            
        Returns:
            DataFrame: Preprocessed data
        """
        df_processed = df.copy()
        
        # Handle missing values
        if is_training:
            print("XGBoost+RF preprocessing: Handling missing values...")
        
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].fillna('unknown')
            elif df_processed[col].dtype in ['float64', 'int64']:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Advanced feature engineering
        if is_training:
            print("XGBoost+RF preprocessing: Feature engineering...")
        
        # Create autism score (sum of A1-A10)
        autism_columns = [col for col in df_processed.columns if col.startswith('A') and col[1:].isdigit()]
        if autism_columns:
            df_processed['autism_score'] = df_processed[autism_columns].sum(axis=1)
        
        # Create age groups (numerical encoding)
        if 'age_months' in df_processed.columns:
            df_processed['age_group'] = pd.cut(df_processed['age_months'], 
                                             bins=[0, 12, 24, 36, 48, 100], 
                                             labels=[0, 1, 2, 3, 4]).astype(int)
        
        # Create risk factors count
        risk_factors = ['jaundice', 'family_asd']
        risk_factors_present = [col for col in risk_factors if col in df_processed.columns]
        if risk_factors_present:
            df_processed['risk_factors_count'] = df_processed[risk_factors_present].apply(
                lambda x: sum(1 for val in x if str(val).lower() in ['yes', '1', 'true']), axis=1)
        
        # Create interaction features
        if 'autism_score' in df_processed.columns and 'age_months' in df_processed.columns:
            df_processed['autism_age_interaction'] = df_processed['autism_score'] * df_processed['age_months']
        
        # Create severity indicators
        if 'autism_score' in df_processed.columns:
            df_processed['high_autism_score'] = (df_processed['autism_score'] >= 7).astype(int)
            df_processed['moderate_autism_score'] = ((df_processed['autism_score'] >= 4) & 
                                                   (df_processed['autism_score'] < 7)).astype(int)
        
        # Encode categorical variables
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                if is_training:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                    else:
                        df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
                else:
                    # Handle unseen labels during prediction
                    unique_values = set(df_processed[col].astype(str).unique())
                    trained_values = set(self.label_encoders[col].classes_)
                    unseen_values = unique_values - trained_values
                    
                    if unseen_values:
                        if is_training:
                            print(f"Warning: Found unseen labels in {col}: {unseen_values}")
                        df_processed[col] = df_processed[col].astype(str)
                        df_processed[col] = df_processed[col].replace(list(unseen_values), 'unknown')
                    
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Scale numerical features
        if is_training:
            print("XGBoost+RF preprocessing: Scaling numerical features...")
            numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
            self.scaler = RobustScaler()
            df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
        else:
            numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numerical_cols] = self.scaler.transform(df_processed[numerical_cols])
        
        return df_processed
    
    def prepare_features(self, df):
        """
        Prepare features for model training/prediction
        
        Args:
            df (DataFrame): Preprocessed data
            
        Returns:
            DataFrame: Features ready for ML
        """
        df_features = df.copy()
        
        # Remove non-predictive columns
        columns_to_remove = ['case_id', 'dataset_source', 'qchat_score', 'total_autism_score']
        for col in columns_to_remove:
            if col in df_features.columns:
                df_features = df_features.drop(col, axis=1)
        
        return df_features
    
    def tune_xgboost_hyperparameters(self, X_train, y_train):
        """
        Tune XGBoost hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            dict: Best parameters for XGBoost
        """
        print("XGBoost tuning: Hyperparameter optimization...")
        
        # XGBoost parameter grid (simplified for speed)
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2]
        }
        
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
            xgb_params, cv=3, scoring='accuracy', n_jobs=-1
        )
        xgb_grid.fit(X_train, y_train)
        
        print(f"Best XGBoost parameters: {xgb_grid.best_params_}")
        return xgb_grid.best_params_
    
    def tune_random_forest_hyperparameters(self, X_train, y_train):
        """
        Tune Random Forest hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            dict: Best parameters for Random Forest
        """
        print("Random Forest tuning: Hyperparameter optimization...")
        
        # Random Forest parameter grid (simplified for speed)
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15],
            'min_samples_split': [2, 5]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state),
            rf_params, cv=3, scoring='accuracy', n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        
        print(f"Best Random Forest parameters: {rf_grid.best_params_}")
        return rf_grid.best_params_
    
    def train_xgboost(self, X_train, y_train, best_params=None):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            best_params: Best parameters from tuning
            
        Returns:
            XGBClassifier: Trained XGBoost model
        """
        print("XGBoost training: Training XGBoost model...")
        
        # Default parameters
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'random_state': self.random_state,
            'eval_metric': 'logloss'
        }
        
        # Use tuned parameters if available
        if best_params:
            default_params.update(best_params)
        
        # Create and train XGBoost model
        xgb_model = xgb.XGBClassifier(**default_params)
        xgb_model.fit(X_train, y_train)
        
        print("XGBoost model trained successfully")
        return xgb_model
    
    def train_random_forest(self, X_train, y_train, best_params=None):
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            best_params: Best parameters from tuning
            
        Returns:
            RandomForestClassifier: Trained Random Forest model
        """
        print("Random Forest training: Training Random Forest model...")
        
        # Default parameters
        default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': self.random_state,
            'class_weight': 'balanced'
        }
        
        # Use tuned parameters if available
        if best_params:
            default_params.update(best_params)
        
        # Create and train Random Forest model
        rf_model = RandomForestClassifier(**default_params)
        rf_model.fit(X_train, y_train)
        
        print("Random Forest model trained successfully")
        return rf_model
    
    def create_ensemble(self, xgb_model, rf_model):
        """
        Create ensemble of XGBoost and Random Forest
        
        Args:
            xgb_model: Trained XGBoost model
            rf_model: Trained Random Forest model
            
        Returns:
            VotingClassifier: Ensemble model
        """
        if not self.use_ensemble:
            return xgb_model  # Return XGBoost as default
        
        print("Ensemble creation: Creating XGBoost + Random Forest ensemble...")
        
        from sklearn.ensemble import VotingClassifier
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model)
            ],
            voting='soft'  # Use probabilities
        )
        
        return ensemble
    
    def train(self, X, y):
        """
        Train XGBoost + Random Forest models
        
        Args:
            X (DataFrame): Features
            y (Series): Target variable
        """
        print("="*80)
        print("XGBOOST + RANDOM FOREST MODEL TRAINING")
        print("="*80)
        
        print(f"Training samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Apply SMOTE for better balance
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=self.random_state)
        X_train_sampled, y_train_sampled = smote.fit_resample(X_train, y_train)
        
        print(f"Original training: {len(X_train)} samples")
        print(f"After SMOTE: {len(X_train_sampled)} samples")
        print(f"SMOTE class distribution: {pd.Series(y_train_sampled).value_counts().to_dict()}")
        
        # Tune hyperparameters
        xgb_best_params = self.tune_xgboost_hyperparameters(X_train_sampled, y_train_sampled)
        rf_best_params = self.tune_random_forest_hyperparameters(X_train_sampled, y_train_sampled)
        
        # Train individual models
        if self.use_xgboost:
            self.xgb_model = self.train_xgboost(X_train_sampled, y_train_sampled, xgb_best_params)
        
        if self.use_random_forest:
            self.rf_model = self.train_random_forest(X_train_sampled, y_train_sampled, rf_best_params)
        
        # Create ensemble
        if self.use_xgboost and self.use_random_forest:
            self.ensemble_model = self.create_ensemble(self.xgb_model, self.rf_model)
            # Train ensemble
            print("Ensemble training: Training ensemble model...")
            self.ensemble_model.fit(X_train_sampled, y_train_sampled)
        
        self.feature_names = X.columns.tolist()
        self.is_trained = True
        
        # Evaluate all models
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        results = {}
        
        # Evaluate XGBoost
        if self.xgb_model is not None:
            y_pred_xgb = self.xgb_model.predict(X_test)
            accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
            
            cm_xgb = confusion_matrix(y_test, y_pred_xgb)
            tn, fp, fn, tp = cm_xgb.ravel()
            specificity_xgb = tn / (tn + fp)
            sensitivity_xgb = tp / (tp + fn)
            bias_xgb = abs(specificity_xgb - sensitivity_xgb)
            
            results['xgboost'] = {
                'accuracy': accuracy_xgb,
                'specificity': specificity_xgb,
                'sensitivity': sensitivity_xgb,
                'bias': bias_xgb
            }
            
            print(f"\nXGBOOST Model:")
            print(f"  Accuracy: {accuracy_xgb:.4f}")
            print(f"  Specificity: {specificity_xgb:.3f}")
            print(f"  Sensitivity: {sensitivity_xgb:.3f}")
            print(f"  Bias: {bias_xgb:.3f}")
        
        # Evaluate Random Forest
        if self.rf_model is not None:
            y_pred_rf = self.rf_model.predict(X_test)
            accuracy_rf = accuracy_score(y_test, y_pred_rf)
            
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            tn, fp, fn, tp = cm_rf.ravel()
            specificity_rf = tn / (tn + fp)
            sensitivity_rf = tp / (tp + fn)
            bias_rf = abs(specificity_rf - sensitivity_rf)
            
            results['random_forest'] = {
                'accuracy': accuracy_rf,
                'specificity': specificity_rf,
                'sensitivity': sensitivity_rf,
                'bias': bias_rf
            }
            
            print(f"\nRANDOM FOREST Model:")
            print(f"  Accuracy: {accuracy_rf:.4f}")
            print(f"  Specificity: {specificity_rf:.3f}")
            print(f"  Sensitivity: {sensitivity_rf:.3f}")
            print(f"  Bias: {bias_rf:.3f}")
        
        # Evaluate Ensemble
        if self.ensemble_model is not None:
            y_pred_ensemble = self.ensemble_model.predict(X_test)
            accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
            
            cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
            tn, fp, fn, tp = cm_ensemble.ravel()
            specificity_ensemble = tn / (tn + fp)
            sensitivity_ensemble = tp / (tp + fn)
            bias_ensemble = abs(specificity_ensemble - sensitivity_ensemble)
            
            results['ensemble'] = {
                'accuracy': accuracy_ensemble,
                'specificity': specificity_ensemble,
                'sensitivity': sensitivity_ensemble,
                'bias': bias_ensemble
            }
            
            print(f"\nENSEMBLE Model:")
            print(f"  Accuracy: {accuracy_ensemble:.4f}")
            print(f"  Specificity: {specificity_ensemble:.3f}")
            print(f"  Sensitivity: {sensitivity_ensemble:.3f}")
            print(f"  Bias: {bias_ensemble:.3f}")
        
        # Store performance history
        self.performance_history.append(results)
        
        return results
    
    def predict(self, X):
        """
        Make predictions using the best model
        
        Args:
            X (DataFrame): Features
            
        Returns:
            array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess the input data
        X_processed = self.preprocess_data(X, is_training=False)
        X_features = self.prepare_features(X_processed)
        
        # Use ensemble if available, otherwise use XGBoost
        if self.ensemble_model is not None:
            return self.ensemble_model.predict(X_features)
        elif self.xgb_model is not None:
            return self.xgb_model.predict(X_features)
        else:
            return self.rf_model.predict(X_features)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X (DataFrame): Features
            
        Returns:
            array: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess the input data
        X_processed = self.preprocess_data(X, is_training=False)
        X_features = self.prepare_features(X_processed)
        
        # Use ensemble if available, otherwise use XGBoost
        if self.ensemble_model is not None:
            return self.ensemble_model.predict_proba(X_features)
        elif self.xgb_model is not None:
            return self.xgb_model.predict_proba(X_features)
        else:
            return self.rf_model.predict_proba(X_features)
    
    def get_feature_importance(self):
        """
        Get feature importance from XGBoost model
        
        Returns:
            DataFrame: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Use XGBoost for feature importance
        model = self.xgb_model if self.xgb_model is not None else self.rf_model
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def cross_validate(self, X, y, cv_folds=5):
        """
        Perform cross-validation on all models
        
        Args:
            X (DataFrame): Features
            y (Series): Target variable
            cv_folds (int): Number of CV folds
            
        Returns:
            dict: CV results for all models
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before cross-validation")
        
        print("Cross-validation: Performing cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_results = {}
        
        # Cross-validate XGBoost
        if self.xgb_model is not None:
            cv_scores = cross_val_score(self.xgb_model, X, y, cv=cv, scoring='accuracy')
            cv_results['xgboost'] = {
                'cv_scores': cv_scores,
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std()
            }
        
        # Cross-validate Random Forest
        if self.rf_model is not None:
            cv_scores = cross_val_score(self.rf_model, X, y, cv=cv, scoring='accuracy')
            cv_results['random_forest'] = {
                'cv_scores': cv_scores,
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std()
            }
        
        # Cross-validate Ensemble
        if self.ensemble_model is not None:
            cv_scores = cross_val_score(self.ensemble_model, X, y, cv=cv, scoring='accuracy')
            cv_results['ensemble'] = {
                'cv_scores': cv_scores,
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std()
            }
        
        return cv_results
    
    def save_model(self, filepath):
        """
        Save the XGBoost + Random Forest model
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'xgb_model': self.xgb_model,
            'rf_model': self.rf_model,
            'ensemble_model': self.ensemble_model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'use_xgboost': self.use_xgboost,
            'use_random_forest': self.use_random_forest,
            'use_ensemble': self.use_ensemble,
            'random_state': self.random_state,
            'performance_history': self.performance_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"XGBoost + Random Forest model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load the XGBoost + Random Forest model
        
        Args:
            filepath (str): Path to the saved model
        """
        _patch_numpy2_pickle_compat()
        model_data = joblib.load(filepath)
        
        self.xgb_model = model_data['xgb_model']
        self.rf_model = model_data['rf_model']
        self.ensemble_model = model_data['ensemble_model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.use_xgboost = model_data['use_xgboost']
        self.use_random_forest = model_data['use_random_forest']
        self.use_ensemble = model_data['use_ensemble']
        self.random_state = model_data['random_state']
        self.performance_history = model_data.get('performance_history', [])
        self.is_trained = True

        if self.xgb_model is not None:
            _patch_xgb_sklearn_compat(self.xgb_model)
        if self.ensemble_model is not None:
            _patch_xgb_sklearn_compat(self.ensemble_model)
        
        print(f"XGBoost + Random Forest model loaded from: {filepath}")

def main():
    """
    Main function to demonstrate the XGBoost + Random Forest model
    """
    print("="*80)
    print("XGBOOST + RANDOM FOREST AUTISM SCREENING MODEL")
    print("="*80)
    
    # Load the clean dataset
    print("\n1. Loading Clean Dataset...")
    df = pd.read_csv('Clean_Autism_Dataset.csv')
    print(f"Clean dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Initialize the XGBoost + Random Forest model
    print("\n2. Initializing XGBoost + Random Forest Model...")
    model = XGBoostRandomForestModel(
        use_xgboost=True,
        use_random_forest=True,
        use_ensemble=True,
        random_state=42
    )
    
    # Preprocess the data
    print("\n3. Preprocessing Data...")
    df_processed = model.preprocess_data(df, is_training=True)
    df_features = model.prepare_features(df_processed)
    
    # Separate features and target
    X = df_features.drop('asd_class', axis=1)
    y = df_features['asd_class']
    
    print(f"Features for training: {list(X.columns)}")
    
    # Train the model
    print("\n4. Training XGBoost + Random Forest Model...")
    results = model.train(X, y)
    
    # Get feature importance
    print("\n5. Feature Importance Analysis...")
    feature_importance = model.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Cross-validation
    print("\n6. Cross-Validation...")
    cv_results = model.cross_validate(X, y)
    
    print("\nCross-Validation Results:")
    for name, cv_result in cv_results.items():
        print(f"{name.upper()}: {cv_result['mean_cv_score']:.4f} ± {cv_result['std_cv_score']:.4f}")
    
    # Save the model
    print("\n7. Saving XGBoost + Random Forest Model...")
    model.save_model('xgboost_random_forest_model.pkl')
    
    # Demonstrate prediction
    print("\n8. Model Prediction Example...")
    sample_data = pd.DataFrame({
        'A1': [1, 0, 1, 0, 1],
        'A2': [0, 1, 1, 0, 1],
        'A3': [1, 0, 1, 1, 0],
        'A4': [1, 1, 0, 1, 1],
        'A5': [0, 0, 1, 1, 1],
        'A6': [1, 0, 0, 1, 1],
        'A7': [0, 1, 1, 0, 1],
        'A8': [1, 0, 1, 1, 0],
        'A9': [0, 1, 0, 1, 1],
        'A10': [1, 0, 1, 0, 1],
        'age_months': [24, 18, 30, 12, 36],
        'gender': ['male', 'female', 'male', 'female', 'male'],
        'jaundice': ['no', 'yes', 'no', 'no', 'yes'],
        'family_asd': ['no', 'no', 'yes', 'no', 'yes']
    })
    
    predictions = model.predict(sample_data)
    probabilities = model.predict_proba(sample_data)
    
    print("Sample predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Sample {i+1}: Prediction={pred}, Probability=[No ASD: {prob[0]:.3f}, ASD: {prob[1]:.3f}]")
    
    # Final summary
    print("\n" + "="*80)
    print("XGBOOST + RANDOM FOREST MODEL SUMMARY")
    print("="*80)
    
    # Find best model
    best_model = None
    best_accuracy = 0
    for name, result in results.items():
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_model = name
    
    print(f"Best Model: {best_model.upper()}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Dataset: Clean_Autism_Dataset.csv")
    print(f"Features: {len(model.feature_names)}")
    print(f"Training Samples: {len(X)}")
    print(f"Missing Values: 0 (clean dataset)")
    
    print(f"\nAdvanced Features Applied:")
    print(f"OK: XGBoost: {model.use_xgboost}")
    print(f"OK: Random Forest: {model.use_random_forest}")
    print(f"OK: Ensemble: {model.use_ensemble}")
    print(f"OK: SMOTE sampling: Yes")
    print(f"OK: Hyperparameter tuning: Yes")
    print(f"OK: Feature engineering: Yes")
    print(f"OK: Robust scaling: Yes")
    print(f"OK: Cross-validation: Yes")
    
    print(f"\nXGBoost + Random Forest Model Ready for Production Use!")
    print("="*80)

if __name__ == "__main__":
    main()
