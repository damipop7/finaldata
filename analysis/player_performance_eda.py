import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class PlayerPerformanceEDA:
    def __init__(self):
        self.data = None
        self.results_dir = 'results/PlayerPerf_insight'  # Changed from 'results/EDA'
        self.log_file = os.path.join(self.results_dir, 'analysis_log.txt')
        self._setup_results_directory()

    def _setup_results_directory(self):
        """Setup results directory and log file"""
        # Create directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Clear existing content
        for file in os.listdir(self.results_dir):
            if file != 'analysis_log.txt':  # Keep the log file
                os.remove(os.path.join(self.results_dir, file))
    
    def _log_run(self, message):
        """Log analysis run with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f"\n[{timestamp}] {message}")

    def load_data(self):
        """Load and prepare the combined player data"""
        self.data = pd.read_csv(r'data/processed/compressed_players_single/part-00000-dff7338d-b9bb-41f3-820c-05d6b1a7605a-c000.csv')
        
        try:
            self.data['date_of_birth'] = pd.to_datetime(self.data['date_of_birth'], utc=True)
            self.data['contract_expiration_date'] = pd.to_datetime(self.data['contract_expiration_date'], utc=True)
            
            current_date = pd.Timestamp.now(tz='UTC')
            self.data['age'] = (current_date - self.data['date_of_birth']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
            self.data['age'] = self.data['age'].round().astype('Int64')
            
        except Exception as e:
            print(f"Error processing dates: {e}")
            self.data['age'] = np.nan
        
        self.data['max_transfer_fee'] = self.data['transfer_fee_history'].apply(self._clean_transfer_fee)
        
        print(f"Loaded {len(self.data)} player records")
        self._inspect_data()
    
    def _clean_transfer_fee(self, fee):
        """Helper method to clean transfer fee data"""
        try:
            if isinstance(fee, str):
                return max([float(x) for x in fee.split(', ')])
            return float(fee)
        except:
            return np.nan
    
    def _inspect_data(self):
        """Print basic data inspection results"""
        print("\nDataset Shape:", self.data.shape)
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
    def save_plot(self, plot_name):
        """Save plot to results directory"""
        filename = f"{plot_name}.png"
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        self._log_run(f"Generated plot: {filename}")

    def save_stats(self, stats_df, stats_name):
        """Save statistics to results directory"""
        filename = f"{stats_name}.csv"
        filepath = os.path.join(self.results_dir, filename)
        stats_df.to_csv(filepath, index=True)
        self._log_run(f"Generated stats: {filename}")

    def save_data_info(self):
        """Save dataset information to a text file"""
        info_path = os.path.join(self.results_dir, 'dataset_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Dataset Analysis Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total Records: {len(self.data)}\n\n")
            f.write("Dataset Shape:\n")
            f.write(f"{self.data.shape}\n\n")
            f.write("Data Types:\n")
            f.write(f"{self.data.dtypes}\n\n")
            f.write("Missing Values:\n")
            f.write(f"{self.data.isnull().sum()}")
        self._log_run("Generated dataset info")

    def basic_stats(self):
        """Generate basic statistics about the dataset"""
        stats = {
            'total_players': len(self.data),
            'unique_clubs': self.data['current_club_name'].nunique(),
            'avg_market_value': self.data['market_value_in_eur'].mean(),
            'max_market_value': self.data['market_value_in_eur'].max(),
            'positions': self.data['sub_position'].value_counts()
        }
        
        stats_df = pd.DataFrame([stats])
        self.save_stats(stats_df, 'basic_stats')
        return stats

    def market_value_analysis(self):
        """Analyze player market values"""
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.data, x='market_value_in_eur', bins=50)
        plt.title('Distribution of Player Market Values')
        plt.xlabel('Market Value (EUR)')
        self.save_plot('market_value_dist')
        
        top_players = self.data.nlargest(10, 'market_value_in_eur')[
            ['name', 'current_club_name', 'market_value_in_eur']
        ]
        self.save_stats(top_players, 'top_valuable_players')

    def position_analysis(self):
        """Analyze player positions and their characteristics"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.data, x='sub_position', y='market_value_in_eur')
        plt.xticks(rotation=45)
        plt.title('Market Value by Position')
        self.save_plot('position_market_value')
        
        pos_stats = self.data.groupby('sub_position').agg({
            'market_value_in_eur': ['mean', 'median', 'count']
        }).round(2)
        self.save_stats(pos_stats, 'position_statistics')

    def age_analysis(self):
        """Analyze player age distribution and correlation with market value"""
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=self.data, x='age', y='market_value_in_eur')
        plt.title('Age vs Market Value')
        self.save_plot('age_market_value')
        
        age_stats = self.data.groupby(self.data['age'].round())[['market_value_in_eur']].mean()
        self.save_stats(age_stats, 'age_value_correlation')

    def club_analysis(self):
        """Analyze clubs and their player values"""
        club_stats = self.data.groupby('current_club_name').agg({
            'market_value_in_eur': ['count', 'mean', 'sum']
        }).round(2)
        
        top_clubs = club_stats.nlargest(10, ('market_value_in_eur', 'sum'))
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_clubs.index, 
                   y=top_clubs[('market_value_in_eur', 'sum')])
        plt.xticks(rotation=45)
        plt.title('Top 10 Clubs by Total Player Value')
        self.save_plot('top_clubs_value')
        
        self.save_stats(club_stats, 'club_statistics')

    def nationality_analysis(self):
        """Analyze player nationalities"""
        plt.figure(figsize=(12, 6))
        self.data['country_of_citizenship'].value_counts().head(10).plot(kind='barh')
        plt.title('Top 10 Player Nationalities')
        plt.xlabel('Number of Players')
        plt.ylabel('Country')
        self.save_plot('nationality_distribution')
        
    def correlation_analysis(self):
        """Analyze correlations between numeric features"""
        corr_matrix = self.data[
            ['age', 'height_in_cm', 'market_value_in_eur', 'max_transfer_fee']
        ].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        self.save_plot('correlation_matrix')
        
    def transfer_analysis(self):
        """Analyze top transfers"""
        top_transfers = self.data[['name', 'max_transfer_fee']].dropna().sort_values(
            'max_transfer_fee', ascending=False
        ).head(10)
        self.save_stats(top_transfers, 'top_transfers')

    def run_full_analysis(self):
        """Run all analyses"""
        start_time = datetime.now()
        self._log_run("Starting analysis")
        
        print("Loading data...")
        self.load_data()
        self.save_data_info()
        
        print("Generating basic statistics...")
        self.basic_stats()
        
        print("Analyzing market values...")
        self.market_value_analysis()
        
        print("Analyzing positions...")
        self.position_analysis()
        
        print("Analyzing age relationships...")
        self.age_analysis()
        
        print("Analyzing clubs...")
        self.club_analysis()
        
        print("Analyzing nationalities...")
        self.nationality_analysis()
        
        print("Analyzing correlations...")
        self.correlation_analysis()
        
        print("Analyzing transfers...")
        self.transfer_analysis()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self._log_run(f"Analysis complete. Duration: {duration:.2f} seconds")
        print(f"Analysis complete. Results saved in {self.results_dir}")
        print(f"Check {self.log_file} for run history")

if __name__ == "__main__":
    eda = PlayerPerformanceEDA()
    eda.run_full_analysis()