import pandas as pd

RAW_PATH = "ML-P2/data/raw/flights_sample_3m.csv"
CLEANED_PATH = "ML-P2/data/processed/flights_sample_3m_cleaned.csv"

def load_raw(path=RAW_PATH):
    return pd.read_csv(path)

#remove cancelled and delayed flights rows
def remove_non_operated_flights(df):
    return df[(df['CANCELLED']==0) & (df['DIVERTED']==0)].copy()


def handle_missing_values(df):
    # these coulums needed to build target 
    core_cols = ['DEP_TIME', 'DEP_DELAY', 'ARR_DELAY', 'CRS_DEP_TIME',
                 'AIRLINE', 'ORIGIN', 'DEST', 'DISTANCE', 'FL_DATE']
    before = len(df)
    df = df.dropna(subset=core_cols)
    print(f"Dropped {before - len(df)} rows with missing core values")
    return df


def create_target(df):
    df['is_delayed'] = (df['DEP_DELAY'] > 15).astype(int)
    return df


def remove_leakage_columns(df):

    leakage_cols = [
        'DEP_TIME',        
        'ARR_TIME',        
        'ARR_DELAY',       
        'TAXI_OUT', 'TAXI_IN', 'WHEELS_OFF', 'WHEELS_ON',
        'ELAPSED_TIME', 'AIR_TIME',    
        'CANCELLATION_CODE',
        'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
        'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
        'DEP_DELAY',
        'CANCELLED','DIVERTED',
    ]

    return df.drop(columns=[c for c in leakage_cols if c in df.columns])


def remove_unnecessary_columns(df):

    useless_cols = [
        'FL_NUMBER', 'AIRLINE_DOT',
        'AIRLINE_CODE','DOT_CODE',
    ]

    return df.drop(columns=[c for c in useless_cols if c in df.columns])


def run():
    df = load_raw()
    print(f"Raw shape: {df.shape}")
 
    df = remove_non_operated_flights(df)
    df = handle_missing_values(df)
    df = create_target(df)      # craete BEFORE dropping DEP_DELAY
    df = remove_leakage_columns(df)
    df = remove_unnecessary_columns(df)
 
    print(f"Cleaned shape: {df.shape}")
    print(f"Final columns: {df.columns.tolist()}")
    print(f"Target balance:\n{df['is_delayed'].value_counts(normalize=True)}")
 
    df.to_csv(CLEANED_PATH, index=False)
    print(f"Saved cleaned data to {CLEANED_PATH}")
 
 
if __name__ == "__main__":
    run()