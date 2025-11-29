import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
from math import pi
import os

plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.1)

# Takım renkleri
COLORS = {"Fenerbahce": "#002d72", "Galatasaray": "#a90432", "League_Average": "gray"}

class DerbyAnalytics:
    def __init__(self, stats_file, times_file):
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        s_path = os.path.join(base_path, stats_file)
        t_path = os.path.join(base_path, times_file)

        try:
            
            self.df_stats = pd.read_csv(s_path, sep=None, engine='python')
            self.df_stats.columns = self.df_stats.columns.str.strip()

            # Times dosyası biraz karışık, onu özel okuyoruz (utf-8 ve noktalı virgül)
            raw_times = pd.read_csv(t_path, sep=';', encoding='utf-8-sig')
            
            self.df_times = self._process_times_data(raw_times)
            
            print("Veriler yüklendi, analize hazır.")
            
        except FileNotFoundError:
            print("Dosyalar bulunamadı! stats.csv ve Times.csv aynı klasörde mi?")
            raise
        except Exception as e:
            print(f"Bir hata oldu: {e}")
            raise

    def _process_times_data(self, df):
        
        df.columns = [c.strip().upper() for c in df.columns]
        
        processed = pd.DataFrame()
        processed['Time_Interval'] = df['DAKIKALAR'].unique()

      
        def get_goals(team, status):
            mask = (df['TAKIM'] == team) & (df['DURUM'] == status)
            return df[mask].set_index('DAKIKALAR')['GOL SAYISI'].reindex(processed['Time_Interval']).fillna(0).values

        processed['FB_Goals_Scored'] = get_goals('Fenerbahce', 'Atılan')
        processed['FB_Goals_Conceded'] = get_goals('Fenerbahce', 'Yenilen')
        processed['GS_Goals_Scored'] = get_goals('Galatasaray', 'Atılan')
        processed['GS_Goals_Conceded'] = get_goals('Galatasaray', 'Yenilen')

        return processed

    def create_radar_chart(self, team1, team2):
        # Radar grafiğinde kullanacağım özellikler
        categories = ['Goals_Scored', 'xG_Total', 'Possession_Percentage', 
                      'Pass_Accuracy', 'SoT_Per_Match', 'Passes_Total']
        
       
        df_norm = self.df_stats.copy()
        for col in categories:
            max_val = self.df_stats[col].max()
            min_val = self.df_stats[col].min()
            if max_val == min_val:
                df_norm[col] = 1.0
            else:
                df_norm[col] = (self.df_stats[col] - min_val) / (max_val - min_val)

        # Grafik ayarları
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        
        
        for team, color in [(team1, 'blue'), (team2, 'red')]:
            vals = df_norm[df_norm['Team'] == team][categories].values.flatten().tolist()
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, linestyle='solid', label=team, color=COLORS.get(team, color))
            ax.fill(angles, vals, COLORS.get(team, color), alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        plt.title(f"Kadro Performans Analizi: {team1} vs {team2}", y=1.08, fontweight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.show()

    def predict_poisson(self, team1, team2):
        # Lig ortalamaların
        avg_stats = self.df_stats[['Matches_Played', 'Goals_Scored', 'Goals_Conceded']].mean()
        avg_gls = avg_stats['Goals_Scored'] / avg_stats['Matches_Played']
        
        t1 = self.df_stats[self.df_stats['Team'] == team1].iloc[0]
        t2 = self.df_stats[self.df_stats['Team'] == team2].iloc[0]

        # Hücum ve Savunma
        t1_att = (t1['Goals_Scored'] / t1['Matches_Played']) / avg_gls
        t1_def = (t1['Goals_Conceded'] / t1['Matches_Played']) / avg_gls
        t2_att = (t2['Goals_Scored'] / t2['Matches_Played']) / avg_gls
        t2_def = (t2['Goals_Conceded'] / t2['Matches_Played']) / avg_gls

        # Beklenen gol sayısı (Lambda)
        lam1 = t1_att * t2_def * avg_gls
        lam2 = t2_att * t1_def * avg_gls

        # Olasılıkları hesapla
        x_range = range(6)
        matrix = np.outer([poisson.pmf(i, lam1) for i in x_range], [poisson.pmf(i, lam2) for i in x_range])

        # En yüksek ihtimalleri bulup sırala
        flat_indices = np.argsort(matrix.ravel())[::-1]
        
        print("\n" + "="*50)
        print(f"📊 MAÇ TAHMİN RAPORU ({team1} vs {team2})")
        print("="*50)
        
        # İlk 3 senaryoyu yazdır
        for i in range(3):
            idx = flat_indices[i]
            score = np.unravel_index(idx, matrix.shape)
            prob = matrix.ravel()[idx]
            
            emoji = ["🥇", "🥈", "🥉"][i]
            print(f"{emoji} Senaryo: {team1} {score[0]} - {score[1]} {team2}  (Olasılık: %{prob*100:.2f})")
            
        print("-" * 50)
        print(f"Beklenen Gol (xG): {team1} {lam1:.2f} - {lam2:.2f} {team2}")
        print("="*50 + "\n")

        # Isı haritası 
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix.T, annot=True, fmt=".1%", cmap='Reds', xticklabels=x_range, yticklabels=x_range)
        
       
        best_score = np.unravel_index(flat_indices[0], matrix.shape)
        plt.title(f'Skor Olasılık Matrisi (Poisson)\nEn Güçlü Tahmin: {team1} {best_score[0]} - {best_score[1]} {team2}')
        
        plt.xlabel(f'{team1} Gol')
        plt.ylabel(f'{team2} Gol')
        plt.show()

    def analyze_intervals(self):
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        
        # Grafik 1: FB Atıyor - GS Yiyor
        sns.barplot(data=self.df_times, x='Time_Interval', y='FB_Goals_Scored', ax=ax[0], color=COLORS['Fenerbahce'], alpha=0.8, label='FB Attığı')
        sns.lineplot(data=self.df_times, x='Time_Interval', y='GS_Goals_Conceded', ax=ax[0], color='red', marker='o', lw=3, label='GS Yediği')
        ax[0].set_title('Analiz 1: Fenerbahçe Hücumu Hangi Dakikalarda Etkili?')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # Grafik 2: GS Atıyor - FB Yiyor
        sns.barplot(data=self.df_times, x='Time_Interval', y='GS_Goals_Scored', ax=ax[1], color=COLORS['Galatasaray'], alpha=0.8, label='GS Attığı')
        sns.lineplot(data=self.df_times, x='Time_Interval', y='FB_Goals_Conceded', ax=ax[1], color='blue', marker='o', lw=3, label='FB Yediği')
        ax[1].set_title('Analiz 2: Galatasaray Hücumu Hangi Dakikalarda Etkili?')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Files names
    STATS_FILE = 'stats.csv'
    TIMES_FILE = 'Times.csv'
    
    try:
        analyst = DerbyAnalytics(STATS_FILE, TIMES_FILE)
        
        analyst.create_radar_chart("Fenerbahce", "Galatasaray")
        analyst.predict_poisson("Fenerbahce", "Galatasaray")
        analyst.analyze_intervals()
        
    except Exception as e:
        print(f"Program durdu: {e}")