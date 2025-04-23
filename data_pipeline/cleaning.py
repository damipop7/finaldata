import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def init_spark():
    """Initialize Spark session with Java configurations"""
    # Set Java home
    java_home = os.popen('/usr/libexec/java_home').read().strip()
    os.environ['JAVA_HOME'] = java_home
    
    return SparkSession.builder \
        .appName("PlayerDataCleaning") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.host", "localhost") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1") \
        .getOrCreate()

def clean_player_data(input_path="data/raw/kaggle/player-scores", 
                     output_path="data/processed/player_data"):
    """Clean and centralize player data"""
    print("🧹 Starting data cleaning process...")
    
    # Initialize Spark
    spark = init_spark()
    
    # Get all CSV files in the input directory
    csv_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    # Initialize list to store valid dataframes
    valid_dfs = []
    
    # Process each CSV file
    for file_path in csv_files:
        print(f"📖 Reading {os.path.basename(file_path)}...")
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        
        # Check if player_id exists in columns
        if 'player_id' in df.columns:
            # Basic cleaning steps
            df_cleaned = df \
                .dropDuplicates(['player_id']) \
                .dropna(subset=['player_id']) \
                .filter(col('player_id').isNotNull())
            
            valid_dfs.append(df_cleaned)
            print(f"✅ Added {df_cleaned.count()} records from {os.path.basename(file_path)}")
        else:
            print(f"❌ Skipping {os.path.basename(file_path)} - no player_id column")
    
    if not valid_dfs:
        print("❌ No valid datasets found with player_id!")
        return
    
    # Merge all valid dataframes
    print("🔄 Merging valid datasets...")
    final_df = valid_dfs[0]
    for df in valid_dfs[1:]:
        final_df = final_df.join(df, 'player_id', 'outer')
    
    # Save the cleaned dataset
    print("💾 Saving cleaned dataset...")
    os.makedirs(output_path, exist_ok=True)
    final_df.write.parquet(
        os.path.join(output_path, "player_data.parquet"),
        mode="overwrite"
    )
    
    # Print summary
    print("\n📊 Cleaning Summary:")
    print(f"Total input files processed: {len(csv_files)}")
    print(f"Valid files with player_id: {len(valid_dfs)}")
    print(f"Final dataset rows: {final_df.count()}")
    print(f"Final dataset columns: {len(final_df.columns)}")
    
    spark.stop()