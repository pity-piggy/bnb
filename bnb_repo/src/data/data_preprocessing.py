import pandas as pd
import numpy as np
from datetime import datetime


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute haversine distance (in km) between two geo coordinates.
    """
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def load_and_clean_data(filepath):
    # Load raw data
    df = pd.read_csv(filepath, low_memory=False)

    # Select relevant features
    selected_columns = [
        'latitude', 'longitude', 'neighbourhood_cleansed',
        'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'property_type', 'host_is_superhost', 'host_since', 'host_listings_count',
        'availability_365', 'minimum_nights', 'maximum_nights',
        'number_of_reviews', 'review_scores_rating', 'review_scores_cleanliness',
        'review_scores_location', 'instant_bookable',
        'reviews_per_month', 'price'  # target
    ]

    df = df[selected_columns].copy()
    # formatting dataset
    # Clean and convert price column
    df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)

    # Convert date columns
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')

    # Feature engineering:
    # host_tenure in days
    today = pd.to_datetime("today")
    df['host_tenure_days'] = (today - df['host_since']).dt.days
    df.drop(columns='host_since', inplace=True)

    # Add distance to McGill University (Downtown Montreal)
    mcgill_lat, mcgill_lon = 45.5048, -73.5772
    df['distance_to_downtown'] = haversine_distance(df['latitude'], df['longitude'], mcgill_lat, mcgill_lon)

    # price per person
    df['price_per_person'] = df['price'] / df['accommodates']
    df['price_per_person'] = df['price_per_person'].replace([np.inf, -np.inf], np.nan)
    df['price_per_person'] = df['price_per_person'].fillna(df['price_per_person'].median())

    # Handle missing values
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Remove listings with uncommon room types
    df = df[~df['room_type'].isin(['Hotel room', 'Shared room'])]

    #cap minimal and maximum nights
    df['maximum_nights_capped'] = df['maximum_nights'].clip(upper=365)
    df['minimum_nights_capped'] = df['minimum_nights'].clip(upper=90)

    # Limit neighbourhood_cleansed to top 10
    top_neighborhoods = df['neighbourhood_cleansed'].value_counts().nlargest(10).index
    df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].apply(
        lambda x: x if x in top_neighborhoods else 'Other'
    )

    # Limit property_type to top 10
    top_property_types = df['property_type'].value_counts().nlargest(10).index
    df['property_type'] = df['property_type'].apply(
        lambda x: x if x in top_property_types else 'Other'
    )


    # One-hot encode categorical variables
    categorical_cols = ['room_type', 'property_type', 'neighbourhood_cleansed']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Binary encoding for host_is_superhost and instant_bookable
    binary_cols = ['host_is_superhost', 'instant_bookable']
    for col in binary_cols:
        df[col] = df[col].map({'t': 1, 'f': 0})

    return df


# Example usage
if __name__ == "__main__":
    input_path = "../../data/raw/listings2.csv"
    output_path_full = "../../data/processed/airbnb_montreal_cleaned.csv"
    output_path_filtered = "../../data/processed/airbnb_montreal_cleaned_1.csv"

    # Load and clean
    df_cleaned = load_and_clean_data(input_path)
    df_cleaned.to_csv(output_path_full, index=False)

    # Apply business rule
    df_filtered = df_cleaned[df_cleaned["price"] < 1000]
    df_filtered.to_csv(output_path_filtered, index=False)

    print("  Data cleaned and saved to:")
    print(f"   Full dataset → {output_path_full}")
    print(f"   Filtered (<$1000) → {output_path_filtered}")
