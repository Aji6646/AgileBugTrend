# -*- coding: utf-8 -*-
"""Amrutha_Diss_v2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15r4WdKD9gf99dHuUSBvzhLDNeml74XwF
"""

# -*- coding: utf-8 -*-
"""
AI-Powered Bug Trend Analysis - Final Corrected Version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import re
import nltk
nltk.download('punkt_tab')
from datetime import datetime
from collections import Counter
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            RandomForestRegressor, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, mean_absolute_error,
                           confusion_matrix, accuracy_score, r2_score)
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_val_score

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download(['stopwords', 'wordnet', 'punkt'])
warnings.filterwarnings('ignore')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def parse_custom_datetime(date_str):
    """Robust datetime parser for JIRA data"""
    if pd.isna(date_str) or str(date_str).lower() in ['none', 'nan', '']:
        return pd.NaT

    try:
        # Try multiple common JIRA datetime formats
        for fmt in ("%d/%b/%Y %I:%M %p",   # "27/Apr/2023 7:35 PM"
                   "%d/%b/%Y %H:%M",       # "27/Apr/2023 19:35"
                   "%Y-%m-%d %H:%M:%S",    # "2023-04-27 19:35:00"
                   "%m/%d/%Y %H:%M",       # "04/27/2023 19:35"
                   "%d-%b-%y %I:%M %p"):   # "27-Apr-23 7:35 PM"
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        return pd.to_datetime(date_str)  # Fallback to pandas parser
    except Exception as e:
        print(f"Warning: Could not parse date {date_str}: {str(e)}")
        return pd.NaT

def load_data(file_path):
    """Load and preprocess the initial dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"\nDataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
        print("\nMissing values per column:")
        print(df.isnull().sum())


        print("-----------------")
        print(df.columns)

        return df
    except Exception as e:
        print(f"\nError loading file: {str(e)}")
        return None

def clean_text(text):
    """Enhanced text cleaning with spam removal"""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs and markdown links
    text = re.sub(r'\[.*?\]\(https?://\S+\)', '', text)  # Markdown links
    text = re.sub(r'https?://\S+', '', text)  # Plain URLs
    text = re.sub(r'\*+', ' ', text)  # Asterisks
    text = re.sub(r'\bclick here\b', '', text)  # Call-to-action phrases

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Custom blocklist for spam terms
    spam_terms = {
        'primalbeast', 'maleenhancement', 'enhancement', 'supplement',
        'sexual', 'wellbeing', 'clinicallytested', 'bloodflow',
        'genitals', 'sideeffects', 'buynow', 'officialwebsite',
        'sharktank', 'legitorscam', 'pill', 'review', 'reviews',
        'clickhere', 'under18', 'h2', 'jimdosite', 'outlookindianewsblog',
        'facebook', 'groupsgoogle', 'sitesgoogle', 'jiraatlassian','natural',
        'primal', 'beast','male','contains','reproduce','clinically','blood','ingredient',
        'men','safe','product'
    }

    # Tokenize and filter
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if (word not in stop_words and
            word not in spam_terms and
            len(word) > 2 and
            not word.isdigit())
    ]

    return ' '.join(tokens)

def handle_symptom_severity(df):
    """
    Handles symptom severity, mapping various string values to a consistent integer encoding.
    Handles only specified values, returning NaN for others.

    Args:
        df: The DataFrame containing the data.

    Returns:
        The DataFrame with an added 'Severity_Encoded' column, or None if errors occur.
    """

    if 'Custom field (Symptom Severity)' not in df.columns:
        print("Error: 'Custom field (Symptom Severity)' column not found.")
        return None

    severity_map = {
        'Major': 2,
        'Minor': 0,
        'Severity1-Critical': 3,
        'Severity 2- Major': 2,
        'Severity-Minor': 0,
    }

    # Use .map with .fillna() to handle values not in the dictionary gracefully.
    df['Severity'] = df['Custom field (Symptom Severity)'].str.strip().map(severity_map).fillna(-1)
    df['Severity_Encoded'] = df['Severity']
    # Explicitly filter out the rows with -1, indicating invalid severity values.
    df = df[df['Severity_Encoded'] != -1]

    return df

def is_useless_column(col):
    unique_vals = col.dropna().unique()
    return len(unique_vals) <= 1

def get_base_name(col_name):
    """Helper to remove .1, .2 suffixes"""
    return re.sub(r"\.\d+$", "", col_name)

def engineer_features(df):
    """Create additional features from the raw data"""
    if df is None or df.empty:
        print("\nNo data to engineer features from")
        return None

    df = df.copy()

    # Drop all duplicate-named variant columns (e.g., 'Comment.1') keeping only the original
    seen_base_cols = set()
    cols_to_keep = []
    for col in df.columns:
        base_col = get_base_name(col)
        if base_col not in seen_base_cols:
            seen_base_cols.add(base_col)
            cols_to_keep.append(col)
    df = df[cols_to_keep]

    # Drop columns with only one unique value or completely blank
    useless_cols = [col for col in df.columns if is_useless_column(df[col])]
    df = df.drop(columns=useless_cols)
    df=handle_symptom_severity(df)

    print(f"🧹 Dropped {len(useless_cols)} columns with only one unique value or entirely blank.")
    print("-----------------")
    print(df.columns.tolist())
    print("****************")

    # Combine text columns safely
    text_cols = ['Summary', 'Description']
    comment_cols = [col for col in df.columns if col.startswith('Comment')]
    for col in text_cols + comment_cols:
        if col not in df.columns:
            df[col] = ''
    df['All_Comments'] = df[comment_cols].astype(str).apply(lambda row: ' '.join(row.values), axis=1)
    df['Text'] = df[text_cols + ['All_Comments']].astype(str).agg(' '.join, axis=1)
    df['Cleaned_Text'] = df['Text'].apply(clean_text)

    # Fill key categorical fields
    df['Priority'] = df['Priority'].fillna('Medium')
    df['Status'] = df['Status'].fillna('Unknown')
    df['Component/s'] = df['Component/s'].fillna('Unknown')
    df['Company'] = df['Custom field (Company)'].fillna('Unknown') if 'Custom field (Company)' in df.columns else 'Unknown'

    # Participant feature
    participant_cols = [col for col in df.columns if "Custom field (Participants" in col]
    if participant_cols:
        df['Participant_Count'] = df[participant_cols].notna().sum(axis=1)
        df['Has_Participants'] = df['Participant_Count'] > 0

   # Handle Symptom Severity
    if 'Custom field (Symptom Severity)' in df.columns:
        df['Severity'] = (
            df['Custom field (Symptom Severity)']
            .str.strip()
            .fillna('Unknown')
        )
        severity_map = {
            'Critical': 3,
            'High': 2,
            'Medium': 1,
            'Low': 0,
            'Unknown': -1
        }
        df['Severity_Encoded'] = df['Severity'].map(severity_map)

    # Parse datetime
    datetime_cols = ['Created', 'Resolved']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_custom_datetime)
        else:
            df[col] = pd.NaT

    # Resolution time
    if 'Created' in df.columns and 'Resolved' in df.columns:
        df['Resolution_Time'] = (df['Resolved'] - df['Created']).dt.total_seconds() / (60 * 60 * 24)

    # Handle resolution outliers
    if 'Resolution_Time' in df.columns:
        q1 = df['Resolution_Time'].quantile(0.25)
        q3 = df['Resolution_Time'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = max(0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        df = df[(df['Resolution_Time'] >= lower_bound) & (df['Resolution_Time'] <= upper_bound)]

    # Temporal features
    if 'Created' in df.columns:
        df['Created_Year'] = df['Created'].dt.year
        df['Created_Month'] = df['Created'].dt.month
        df['Created_Day'] = df['Created'].dt.day
        df['Created_Weekday'] = df['Created'].dt.weekday
        df['Created_Hour'] = df['Created'].dt.hour

    # Categorical encodings
    categorical_cols = ['Priority', 'Status', 'Component/s', 'Company', 'Issue Type']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))

    # Drop rows with critical missing encodings
    required_cols = ['Issue Type_Encoded', 'Priority_Encoded']
    df.dropna(subset=[col for col in required_cols if col in df.columns], inplace=True)

    return df



def vectorize_text(df, text_col='Cleaned_Text', max_features=1000):
    """Vectorize text data using TF-IDF with NaN handling"""
    if df is None or df.empty:
        print("\nNo data to vectorize")
        return np.array([]), None

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    try:
        X_text = tfidf.fit_transform(df[text_col].fillna('')).toarray()
        print(f"\nText vectorization complete. Shape: {X_text.shape}")
        return X_text, tfidf
    except Exception as e:
        print(f"\nError in text vectorization: {str(e)}")
        return np.array([]), None

def perform_topic_modeling(df, text_col='Cleaned_Text', n_topics=5):
    """Identify latent topics in bug reports with error handling"""
    if df is None or text_col not in df.columns:
        print("\nNo data for topic modeling")
        return []

    try:
        count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = count_vectorizer.fit_transform(df[text_col].fillna(''))

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)

        feature_names = count_vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            topics.append([feature_names[i] for i in topic.argsort()[-10:]])

        print("\nTopic modeling completed successfully")
        return topics
    except Exception as e:
        print(f"\nError in topic modeling: {str(e)}")
        return []

def train_classification_models(X, y):
    """Train and evaluate classification models with enhanced validation"""
    if X.size == 0 or len(y) == 0:
        print("\nNo data for classification training")
        return {}

    # Check class distribution
    class_dist = pd.Series(y).value_counts()
    print("\nClass Distribution:")
    print(class_dist)

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs'),
        'Baseline (Most Frequent)': DummyClassifier(strategy='most_frequent')
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.2f}")

            # Final training
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Store results
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'report': classification_report(y_test, y_pred, output_dict=True),
                'cv_scores': cv_scores
            }

            # Print results
            print(f"\n{name} Classification Report:")
            print(classification_report(y_test, y_pred))

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred),
                       annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

        except Exception as e:
            print(f"Error training {name}: {str(e)}")

    return results

def train_regression_models(X, y):
    """Train and evaluate regression models with diagnostics"""
    if X.size == 0 or len(y) == 0:
        print("\nNo data for regression training")
        return {}

    # Check target distribution
    print("\nTarget Variable Statistics:")
    print(f"Mean: {np.mean(y):.2f}")
    print(f"Median: {np.median(y):.2f}")
    print(f"Std Dev: {np.std(y):.2f}")
    print(f"Min: {np.min(y):.2f}")
    print(f"Max: {np.max(y):.2f}")

    # Plot distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(y, bins=50)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.show()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Baseline (Median)': DummyRegressor(strategy='median')
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            print(f"Cross-validation MAE scores: {-cv_scores}")
            print(f"Mean CV MAE: {-cv_scores.mean():.2f}")

            # Final training
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Store results
            results[name] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'cv_scores': cv_scores
            }

            # Print results
            print(f"\n{name} Performance:")
            print(f"MAE: {results[name]['mae']:.2f}")
            print(f"R2 Score: {results[name]['r2']:.2f}")

            # Plot actual vs predicted
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.3)
            plt.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()], 'k--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{name} - Actual vs Predicted')
            plt.show()

        except Exception as e:
            print(f"Error training {name}: {str(e)}")

    return results

def plot_issue_trends(df):
    """Plot various trends in the bug data with error handling"""
    if df is None or df.empty:
        return

    try:
        plt.figure(figsize=(14, 10))

        # Bug trend over time
        if 'Created' in df.columns:
            plt.subplot(2, 2, 1)
            df['YearMonth'] = df['Created'].dt.to_period('M')
            trend = df.groupby('YearMonth').size()
            trend.plot(kind='line', marker='o', title='Monthly Bug Report Trend')
            plt.ylabel('Number of Issues')
            plt.grid(True)

        # Issue type distribution
        if 'Issue Type' in df.columns:
            plt.subplot(2, 2, 2)
            sns.countplot(y='Issue Type', data=df, order=df['Issue Type'].value_counts().index)
            plt.title("Distribution of Issue Types")

        # Resolution time distribution
        if 'Resolution_Time' in df.columns:
            plt.subplot(2, 2, 3)
            sns.histplot(df['Resolution_Time'], bins=30, kde=True)
            plt.title("Distribution of Resolution Time (Days)")
            plt.xlabel("Days")
            plt.ylabel("Number of Bugs")

        # Priority vs Resolution time
        if 'Priority' in df.columns and 'Resolution_Time' in df.columns:
            plt.subplot(2, 2, 4)
            sns.boxplot(x='Priority', y='Resolution_Time', data=df)
            plt.title("Resolution Time by Priority")
            plt.yscale('log')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plotting trends: {str(e)}")

    # Word cloud for common terms
    try:
        if 'Cleaned_Text' in df.columns:
            plt.figure(figsize=(12, 6))
            wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(
                ' '.join(df['Cleaned_Text'].fillna('')))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Terms in Bug Reports')
            plt.show()
    except Exception as e:
        print(f"Error generating word cloud: {str(e)}")

    # Component-wise analysis
    try:
        if 'Component/s' in df.columns:
            plt.figure(figsize=(12, 6))
            component_counts = df['Component/s'].value_counts().nlargest(10)
            component_counts.plot(kind='barh')
            plt.title('Top 10 Components with Most Bugs')
            plt.xlabel('Number of Bugs')
            plt.show()

            if 'Resolution_Time' in df.columns:
                plt.figure(figsize=(12, 6))
                top_components = df['Component/s'].value_counts().nlargest(5).index
                df_top = df[df['Component/s'].isin(top_components)]
                sns.boxplot(x='Component/s', y='Resolution_Time', data=df_top)
                plt.title('Resolution Time by Top Components')
                plt.xticks(rotation=45)
                plt.yscale('log')
                plt.show()
    except Exception as e:
        print(f"Error in component analysis: {str(e)}")

def prepare_model_features(df):
    """Prepare features for modeling using participant data"""
    # Vectorize text
    X_text, tfidf = vectorize_text(df)

    # Prepare structured features
    structured_features = ['Priority_Encoded', 'Status_Encoded',
                         'Component/s_Encoded', 'Company_Encoded']

    if 'Participant_Count' in df.columns:
        structured_features.append('Participant_Count')
        structured_features.append('Has_Participants')

    # Scale features
    scaler = StandardScaler()
    X_structured = scaler.fit_transform(df[structured_features])

    # Combine features
    X = np.hstack([X_text, X_structured])

    return X

# currently not in use...
def plot_participant_analysis(df):
    """Visualizations focused on participant data"""
    if 'Participant_Count' not in df.columns:
        return

    plt.figure(figsize=(15, 10))

    # Participant count distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['Participant_Count'], bins=30, kde=True)
    plt.title('Distribution of Participant Count per Issue')
    plt.xlabel('Number of Participants')
    plt.ylabel('Number of Issues')

    # Participant presence by issue type
    plt.subplot(2, 2, 2)
    if 'Issue Type' in df.columns:
        sns.countplot(x='Issue Type', hue='Has_Participants', data=df)
        plt.title('Participant Presence by Issue Type')
        plt.ylabel('Number of Issues')

    # Resolution time by participant count
    plt.subplot(2, 2, 3)
    if 'Resolution_Time' in df.columns:
        sns.boxplot(x='Participant_Count', y='Resolution_Time',
                   data=df[df['Participant_Count'] <= 10])  # Limit for readability
        plt.title('Resolution Time by Participant Count')
        plt.xlabel('Number of Participants')
        plt.ylabel('Resolution Time (days)')
        plt.yscale('log')

    # Participant count trend over time
    plt.subplot(2, 2, 4)
    if 'Created' in df.columns:
        df['Created_Month'] = df['Created'].dt.to_period('M')
        monthly_participants = df.groupby('Created_Month')['Participant_Count'].mean()
        monthly_participants.plot(kind='line', marker='o')
        plt.title('Average Participants per Issue Over Time')
        plt.ylabel('Average Participants')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_cluster_analysis(df):
    """Modified cluster analysis using participants"""
    if df is None or 'Cluster' not in df.columns:
        return

    # Cluster characteristics
    print("\nCluster Characteristics:")
    for cluster in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cluster]
        print(f"\nCluster {cluster}:")
        print(f"Size: {len(cluster_df)} ({len(cluster_df)/len(df)*100:.1f}%)")

        if 'Resolution_Time' in cluster_df.columns:
            print(f"Avg Resolution Time: {cluster_df['Resolution_Time'].mean():.2f} days")

        if 'Participant_Count' in cluster_df.columns:
            print(f"Avg Participants: {cluster_df['Participant_Count'].mean():.2f}")

        if 'Issue Type' in cluster_df.columns:
            print("Top Issue Types:")
            print(cluster_df['Issue Type'].value_counts().head(3))

    # Plot clusters using participant data
    if 'Participant_Count' in df.columns and 'Resolution_Time' in df.columns:
        plt.figure(figsize=(12, 8))

        # Filter extreme outliers
        q_high = df['Resolution_Time'].quantile(0.95)
        df_plot = df[df['Resolution_Time'] <= q_high]

        # Create scatter plot
        sns.scatterplot(
            x='Participant_Count',
            y='Resolution_Time',
            hue='Cluster',
            data=df_plot,
            palette='viridis',
            alpha=0.6,
            s=100
        )

        plt.title('Bug Clusters by Participant Count and Resolution Time')
        plt.xlabel('Number of Participants')
        plt.ylabel('Resolution Time (days)')
        plt.grid(True, alpha=0.3)
        plt.show()

def plot_severity_analysis(df):
    if 'Severity' not in df.columns or 'Resolution_Time' not in df.columns:
        print("Error: Required columns missing.")
        return

    try:
        df['Resolution_Time'] = pd.to_numeric(df['Resolution_Time'])
    except (ValueError, TypeError):
        print("Error: 'Resolution_Time' must be numeric.")
        return

    # Drop NaN and check for valid severities
    unique_severities = df['Severity'].dropna().unique()  # Remove NaN
    if len(unique_severities) == 0:  # Check array length instead of truthiness
        print("Error: No valid severity levels.")
        return

    #  Define severity levels in a way that adapts to the data's values.
    severity_order = sorted(unique_severities)
    available_severities = [s for s in severity_order if s in df['Severity'].unique()]

    if not available_severities:
        print("Error: No valid severity levels found in the data.")
        return


    plt.figure(figsize=(15, 10))

    # Severity distribution
    plt.subplot(2, 2, 1)
    sns.countplot(x='Severity', data=df, order=available_severities)
    plt.title('Bug Severity Distribution')
    plt.xticks(rotation=45)


    # Resolution time by severity (using boxplot)
    plt.subplot(2, 2, 2)
    sns.boxplot(x='Severity', y='Resolution_Time', data=df, order=available_severities)
    plt.title('Resolution Time by Severity')
    plt.ylabel('Resolution Time')
    plt.yscale('log')  # Crucial: Use log scale for better visualization of varying scales
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))  #Format for log scale.
    plt.xticks(rotation=45)


    # Severity trend over time
    plt.subplot(2, 2, 4)
    if 'Created_Month' in df.columns:
        severity_trend = df.groupby(['Created_Month', 'Severity']).size().unstack()
        severity_trend.plot(kind='line', marker='o')
        plt.title('Severity Trend Over Time')
        plt.ylabel('Number of Issues')
        plt.grid(True)

    # Display the plots
    plt.tight_layout()  # Important: Adjust subplot parameters for better layout.
    plt.show()

def main():
    """Main execution function with comprehensive error handling"""
    try:
        # Load and preprocess data
        print("\nLoading data...")
        df = load_data("/content/GFG_FINAL.csv")
        if df is None or df.empty:
            print("\nFailed to load data. Exiting.")
            return None, {}, {}

        print("\nEngineering features...")
        df = engineer_features(df)
        if df is None or df.empty:
            print("\nNo valid data after preprocessing. Exiting.")
            return None, {}, {}

        # Generate participant-focused visualizations
        plot_participant_analysis(df)
        # Visualizations
        print("\nGenerating visualizations...")
        plot_issue_trends(df)
        plot_severity_analysis(df)

        # Prepare features for modeling
        X = prepare_model_features(df)

        # Prepare targets
        y_class = df['Issue Type_Encoded'].values if 'Issue Type_Encoded' in df.columns else None
        y_regress = df['Resolution_Time'].values if 'Resolution_Time' in df.columns else None

        # Perform topic modeling
        print("\nPerforming topic modeling...")
        topics = perform_topic_modeling(df)
        if topics:
            print("\nDiscovered Topics:")
            for i, topic in enumerate(topics):
                print(f"Topic {i+1}: {', '.join(topic)}")

        # Train models
        classification_results = {}
        regression_results = {}

        if y_class is not None:
            print("\nTraining classification models...")
            classification_results = train_classification_models(X, y_class)

        if y_regress is not None:
            print("\nTraining regression models...")
            regression_results = train_regression_models(X, y_regress)



        # Cluster analysis
        if X.shape[1] > 0:
            kmeans = KMeans(n_clusters=3, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)
            plot_cluster_analysis(df)

        return df, classification_results, regression_results

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        return None, {}, {}

if __name__ == "__main__":
    print("Starting AI-Powered Bug Trend Analysis...")
    df, classification_results, regression_results = main()
    # Add this in main() before return statement
    df.to_csv('processed_bug_data.csv', index=False)
    print("\nExported processed data to processed_bug_data.csv")
    print("\nAnalysis complete.")