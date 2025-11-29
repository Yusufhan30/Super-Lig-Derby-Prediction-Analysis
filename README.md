# ⚽ Süper Lig Derbi Analizi ve Tahmin Modeli (Python)

Bu proje, Fenerbahçe ve Galatasaray arasındaki derbi maçı öncesinde takımların sezon istatistiklerini analiz etmek ve istatistiksel yöntemlerle maç sonucuna dair olasılıkları hesaplamak için geliştirilmiştir.

## 🎯 Projenin Amacı
Bir İstatistik öğrencisi olarak, futbol verilerini veri bilimi araçlarıyla işleyerek;
- Takımların oyun karakterlerini karşılaştırmak,
- Gol beklentilerini (xG) analiz etmek,
- **Poisson Dağılımı** kullanarak maç skoru ihtimallerini bilimsel bir temele oturtmak amaçlanmıştır.

## 📊 Kullanılan Yöntemler ve Analizler

1.  **Kadro ve Performans Analizi (Radar Chart):** Takımların şut, pas isabeti, topa sahip olma gibi metriklerdeki güç dengelerini görselleştirir.
2.  **Skor Tahmin Simülasyonu (Poisson Distribution):** Lig ortalamaları ve takım güçleri baz alınarak maçın en olası skorlarını (% olasılıklarıyla) hesaplar.
3.  **Tehlikeli Dakikalar Analizi:** Takımların hangi dakika aralıklarında gol attığı ve yediği analiz edilerek maçın kırılma anları tespit edilir.

## 🛠️ Kurulum ve Kullanım

Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### 1. Gerekli Kütüphaneler
Projeyi çalıştırmadan önce gerekli Python kütüphanelerini yükleyin:

```bash
pip install pandas numpy matplotlib seaborn scipy
