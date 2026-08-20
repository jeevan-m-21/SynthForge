"""
Phase 4 Dataset Fixture Generator for SynthForge Generalization Testing.
Generates 6 unseen domain datasets with diverse schemas, cardinalities, missingness, and data types.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

FIXTURES_DIR = Path("data/phase4_fixtures")
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

def generate_fixtures():
    # 1. E-commerce Dataset (50 rows, mixed types, text, missing values, booleans, datetime)
    n = 50
    categories = ["Electronics", "Apparel", "Home & Kitchen", "Books", "Beauty & Personal Care"]
    reviews = [
        "Great quality product, fast shipping!",
        "Item arrived damaged and packaging was poor.",
        "Decent value for money, exactly as described.",
        "Exceeded my expectations, will definitely buy again.",
        "Customer service was unhelpful and return process was slow.",
        "Average build quality, works okay for basic tasks.",
        "Super fast delivery and premium materials used.",
        "Not satisfied with the performance, returning immediately.",
    ]
    ecommerce_df = pd.DataFrame({
        "order_id": [f"ORD-{1000 + i}" for i in range(n)],
        "customer_id": [f"CUST-{500 + (i % 35)}" for i in range(n)],
        "category": np.random.choice(categories, size=n),
        "unit_price": np.round(np.random.uniform(9.99, 499.99, size=n), 2),
        "quantity": np.random.randint(1, 6, size=n),
        "is_prime_member": np.random.choice([True, False], size=n, p=[0.6, 0.4]),
        "order_timestamp": pd.date_range("2024-01-01 09:00:00", periods=n, freq="4h30min").astype(str),
        "customer_review": np.random.choice(reviews, size=n),
        "discount_code": np.random.choice(["SAVE10", "SPRING20", "FLASH50", None], size=n, p=[0.3, 0.2, 0.1, 0.4]),
        "delivery_days": [int(x) if x is not None else None for x in np.random.choice([1, 2, 3, 5, None], size=n, p=[0.2, 0.4, 0.2, 0.1, 0.1])],
        "returned": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
    })
    ecommerce_df.to_csv(FIXTURES_DIR / "ecommerce.csv", index=False)

    # 2. HR Analytics Dataset (60 rows, high-cardinality titles, constant company HQ, text notes)
    n = 60
    titles = [
        "Senior Software Engineer", "Junior Frontend Developer", "Staff Data Architect",
        "Lead DevOps Engineer", "Product Manager", "Associate Product Manager",
        "Technical Recruiter", "HR Business Partner", "Financial Analyst",
        "Senior Accounting Manager", "UX Designer", "Principal Security Analyst",
        "Marketing Specialist", "Customer Success Lead", "Operations Associate",
        "Database Administrator", "Site Reliability Engineer", "Machine Learning Specialist",
        "Business Intelligence Analyst", "Director of Product"
    ]
    depts = ["Engineering", "Product", "Human Resources", "Finance", "Marketing", "Operations"]
    exit_notes = [
        "Employee relocated to another city for family reasons.",
        "Left for higher compensation and promotion opportunity.",
        "Career transition to another industry.",
        "Expressed dissatisfaction with work-life balance.",
        "Pursuing higher education full-time.",
    ]
    hr_df = pd.DataFrame({
        "employee_id": [f"EMP-{100 + i}" for i in range(n)],
        "job_title": np.random.choice(titles, size=n),
        "department": np.random.choice(depts, size=n),
        "salary": np.random.randint(55000, 175000, size=n),
        "years_of_experience": np.round(np.random.uniform(1.0, 20.0, size=n), 1),
        "performance_rating": np.random.randint(1, 6, size=n),
        "is_manager": np.random.choice([True, False], size=n, p=[0.25, 0.75]),
        "company_hq": ["New York"] * n,  # Constant column
        "hire_date": pd.date_range("2015-01-15", periods=n, freq="45D").astype(str),
        "exit_notes": np.random.choice(exit_notes + [None], size=n, p=[0.05, 0.05, 0.05, 0.05, 0.05, 0.75]),
        "attrition": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
    })
    hr_df.to_csv(FIXTURES_DIR / "hr_analytics.csv", index=False)

    # 3. Logistics & Supply Chain Dataset (45 rows, correlated distance/fuel/weight, delay target)
    n = 45
    hubs = ["ORD-Chicago", "JFK-New York", "LAX-Los Angeles", "ATL-Atlanta", "DFW-Dallas", "SEA-Seattle"]
    dist = np.random.uniform(200.0, 3000.0, size=n)
    weight = np.random.uniform(10.0, 250.0, size=n)
    fuel = dist * 0.08 + weight * 0.15 + np.random.normal(0, 5, size=n)
    logistics_df = pd.DataFrame({
        "tracking_number": [f"TRK-{90000 + i}" for i in range(n)],
        "origin_hub": np.random.choice(hubs, size=n),
        "destination_hub": np.random.choice(hubs, size=n),
        "distance_km": np.round(dist, 1),
        "weight_kg": np.round(weight, 1),
        "fuel_consumed_liters": np.round(np.clip(fuel, 10.0, 500.0), 1),
        "dispatch_datetime": pd.date_range("2024-03-01 06:00:00", periods=n, freq="8h").astype(str),
        "requires_temperature_control": np.random.choice([1, 0], size=n, p=[0.3, 0.7]),
        "handling_notes": np.random.choice(["Fragile glass components", "Hazardous chemicals tier 1", "Standard freight", None], size=n, p=[0.2, 0.1, 0.5, 0.2]),
        "delayed": np.random.choice([0, 1], size=n, p=[0.75, 0.25]),
    })
    logistics_df.to_csv(FIXTURES_DIR / "logistics.csv", index=False)

    # 4. IoT Sensor Dataset (55 rows with 5 duplicates, high-frequency datetime, constant firmware, anomaly target)
    n = 50
    base_timestamps = pd.date_range("2024-06-01 00:00:00", periods=n, freq="1s").astype(str)
    temp = np.random.normal(45.0, 8.0, size=n)
    vibe = np.random.normal(120.0, 25.0, size=n)
    volt = np.random.normal(3.30, 0.05, size=n)
    iot_df = pd.DataFrame({
        "sensor_id": [f"SN-{1000 + i}" for i in range(n)],
        "timestamp": base_timestamps,
        "firmware_version": ["v3.1.0-prod"] * n,  # Constant column
        "temperature_celsius": np.round(temp, 2),
        "vibration_hz": np.round(vibe, 2),
        "voltage_volts": np.round(volt, 3),
        "operational_mode": np.random.choice(["ECO", "NORMAL", "BOOST"], size=n, p=[0.2, 0.7, 0.1]),
        "error_flag": np.random.choice([True, False], size=n, p=[0.1, 0.9]),
        "is_anomaly": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
    })
    # Add 5 duplicate rows to test duplicate handling
    duplicates = iot_df.iloc[:5].copy()
    iot_df = pd.concat([iot_df, duplicates], ignore_index=True)
    iot_df.to_csv(FIXTURES_DIR / "iot_sensor.csv", index=False)

    # 5. Financial Transactions Dataset (200 rows, medium size, skewed amounts, fraud target)
    n = 200
    amounts = np.exp(np.random.normal(4.5, 1.2, size=n)) # Lognormal skewed distribution
    merchants = ["Grocery & Supermarket", "Airlines & Travel", "Dining & Restaurants", "Online Retail", "Fuel & Gas", "Healthcare & Pharmacy"]
    fin_df = pd.DataFrame({
        "transaction_id": [f"TXN-{100000 + i}" for i in range(n)],
        "account_number": [f"ACC-{80000 + (i % 80)}" for i in range(n)],
        "transaction_amount": np.round(amounts, 2),
        "merchant_category": np.random.choice(merchants, size=n),
        "is_international": np.random.choice([1, 0], size=n, p=[0.15, 0.85]),
        "transaction_time": pd.date_range("2024-05-01 00:05:00", periods=n, freq="18min").astype(str),
        "terminal_city": np.random.choice(["London", "Tokyo", "New York", "Singapore", "Sydney", None], size=n, p=[0.3, 0.2, 0.2, 0.15, 0.05, 0.1]),
        "card_present": np.random.choice([True, False], size=n, p=[0.7, 0.3]),
        "is_fraud": np.random.choice([0, 1], size=n, p=[0.96, 0.04]),
    })
    fin_df.to_csv(FIXTURES_DIR / "financial_transactions.csv", index=False)

    # 6. Movie Recommendations Dataset (50 rows, high-cardinality movie titles, free-form text, liked target)
    n = 50
    movies = [
        "Inception", "Interstellar", "The Dark Knight", "Pulp Fiction", "The Matrix",
        "Fight Club", "Forrest Gump", "Goodfellas", "The Godfather", "Spirited Away",
        "Parasite", "Whiplash", "Gladiator", "Memento", "The Prestige",
        "Alien", "Blade Runner 2049", "Arrival", "Dune", "Oppenheimer"
    ]
    genres = ["Sci-Fi", "Action", "Drama", "Crime", "Thriller", "Animation"]
    comments = [
        "Masterpiece of visual storytelling and sound design.",
        "Great acting but the pacing dragged in the second act.",
        "One of the best movies of the decade, highly recommended.",
        "Overrated plot with predictable twists and clichéd dialogue.",
        "Incredible practical effects and stellar soundtrack.",
        "Confusing plot structure, required a second viewing to appreciate.",
        "Emotional rollercoaster from start to finish.",
    ]
    movie_df = pd.DataFrame({
        "user_id": [f"USR-{1000 + i}" for i in range(n)],
        "movie_title": np.random.choice(movies, size=n),
        "genre": np.random.choice(genres, size=n),
        "release_year": np.random.randint(1990, 2025, size=n),
        "user_rating": [round(x, 1) if not pd.isna(x) else None for x in np.random.choice([1.0, 2.0, 3.5, 4.0, 4.5, 5.0, np.nan], size=n, p=[0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])],
        "watch_duration_minutes": np.random.randint(10, 180, size=n),
        "has_subtitles": np.random.choice([True, False], size=n, p=[0.8, 0.2]),
        "stream_date": pd.date_range("2024-02-01", periods=n, freq="18h").astype(str),
        "user_comment": np.random.choice(comments + [None], size=n, p=[0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.02]),
        "liked": np.random.choice([0, 1], size=n, p=[0.3, 0.7]),
    })
    movie_df.to_csv(FIXTURES_DIR / "movie_recommendations.csv", index=False)
    print("Fixtures generated successfully in data/phase4_fixtures/")

if __name__ == "__main__":
    generate_fixtures()
