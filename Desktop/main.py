# Gerekli kütüphanelerin import edilmesi
import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QHBoxLayout, QTextEdit
)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import evaluate  # BERTScore değerlendirmesi için

# Ana uygulama sınıfı
class SentenceSimilarityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🟩 Cümle Anlam Benzerliği Karşılaştırma")
        self.resize(900, 600)

        # SBERT ve SimCSE modellerinin yüklenmesi
        self.sbert_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        self.simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        self.simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        self.bertscore = evaluate.load("bertscore")

        self.df = None              # Veri çerçevesi (CSV'den okunacak)
        self.file_path = None       # Dosya yolu saklama

        # Arayüz yerleşimi
        self.layout = QVBoxLayout()

        # Cümle sütunlarını ve model seçimlerini yapabilmek için etiketler ve açılır kutular
        self.col1_label = QLabel("1. Cümle Sütunu:")
        self.col1_selector = QComboBox()
        self.col2_label = QLabel("2. Cümle Sütunu:")
        self.col2_selector = QComboBox()

        self.model_label = QLabel("Model Seçimi:")
        self.model_selector = QComboBox()
        self.model_selector.addItems(["SBERT", "SimCSE"])

        # Dosya yükleme ve karşılaştırma butonları
        self.upload_button = QPushButton("CSV Dosyası Yükle")
        self.upload_button.clicked.connect(self.load_csv)

        self.compare_button = QPushButton("Karşılaştır")
        self.compare_button.clicked.connect(self.compare_sentences)

        # Sonuçların gösterileceği tablo
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels(["Index", "Cümle 1", "Cümle 2", "Cosine", "Dot", "Euclidean"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.cellDoubleClicked.connect(self.show_detail)  # Satıra tıklanınca detay göster

        # Detaylı benzerlik bilgilerini göstermek için metin kutusu
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)

        # Arayüz bileşenlerini layout'a ekle
        self.layout.addWidget(self.col1_label)
        self.layout.addWidget(self.col1_selector)
        self.layout.addWidget(self.col2_label)
        self.layout.addWidget(self.col2_selector)
        self.layout.addWidget(self.model_label)
        self.layout.addWidget(self.model_selector)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.upload_button)
        btn_layout.addWidget(self.compare_button)
        self.layout.addLayout(btn_layout)

        self.layout.addWidget(QLabel("Sonuçlar (İlk 20 Gözlem):"))
        self.layout.addWidget(self.result_table)
        self.layout.addWidget(QLabel("Detay:"))
        self.layout.addWidget(self.detail_text)

        self.setLayout(self.layout)

    # CSV dosyası yükleme işlemi
    def load_csv(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "CSV Dosyası Seç", "", "CSV Dosyaları (*.csv)")
        if file_path:
            encodings = ['utf-8', 'ISO-8859-9', 'windows-1254']  # Olası Türkçe uyumlu encoding listesi
            for enc in encodings:
                try:
                    self.df = pd.read_csv(file_path, sep=None, engine='python', encoding=enc, on_bad_lines='skip')
                    break
                except Exception as e:
                    print(f"Encoding denemesi ({enc}) başarısız: {e}")
                    continue
            if self.df is not None:
                self.file_path = file_path
                self.populate_column_selectors()
            else:
                QMessageBox.critical(self, "Hata", "CSV dosyası okunamadı!")

    # Yüklenen CSV içindeki sütunları seçicilere ekleme
    def populate_column_selectors(self):
        columns = list(self.df.columns)
        self.col1_selector.clear()
        self.col2_selector.clear()
        self.col1_selector.addItems(columns)
        self.col2_selector.addItems(columns)

    # Seçilen modele göre embedding (vektör) çıkarımı
    def get_embedding(self, text, model_name):
        if model_name == "SBERT":
            return self.sbert_model.encode(text, convert_to_tensor=True)
        else:
            inputs = self.simcse_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.simcse_model(**inputs)
            return outputs.pooler_output[0]

    # Cosine, Dot Product ve Euclidean benzerlik metriklerini hesaplama
    def compute_metrics(self, v1, v2):
        v1_np = v1.cpu().numpy()
        v2_np = v2.cpu().numpy()
        cosine = float(cosine_similarity([v1_np], [v2_np])[0][0])
        dot = float(np.dot(v1_np, v2_np))
        euclid = float(euclidean(v1_np, v2_np))
        return {
            "Cosine": round(cosine, 4),
            "Dot": round(dot, 4),
            "Euclidean": round(euclid, 4)
        }

    # Cümle karşılaştırma işlemi başlatılır
    def compare_sentences(self):
        if self.df is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir CSV dosyası yükleyin.")
            return

        col1 = self.col1_selector.currentText()
        col2 = self.col2_selector.currentText()
        model = self.model_selector.currentText()

        results = []
        for i, row in self.df.iterrows():
            try:
                s1 = str(row[col1]).strip()
                s2 = str(row[col2]).strip()
                if s1 and s2:
                    emb1 = self.get_embedding(s1, model)
                    emb2 = self.get_embedding(s2, model)
                    scores = self.compute_metrics(emb1, emb2)
                    bert = self.bertscore.compute(predictions=[s2], references=[s1], lang="en")
                    scores["BERTScore"] = round(bert["f1"][0], 4)
                    results.append([i, s1, s2, scores["Cosine"], scores["Dot"], scores["Euclidean"], scores["BERTScore"]])
            except Exception as e:
                print(f"Hata satır {i}: {e}")
                continue

        # İlk 20 sonucu tabloya yazdır
        self.result_table.setRowCount(min(len(results), 20))
        for row_idx, result in enumerate(results[:20]):
            for col_idx in range(6):  # ilk 6 sütun gösteriliyor
                self.result_table.setItem(row_idx, col_idx, QTableWidgetItem(str(result[col_idx])))

        self.all_results = results  # Detay gösterimi için sonuçları sakla

    # Tabloya çift tıklanınca detaylı açıklamaları metin kutusuna yazdır
    def show_detail(self, row, _):
        if hasattr(self, "all_results") and row < len(self.all_results):
            r = self.all_results[row]
            detail = (
                f"Index: {r[0]}\n"
                f"1. Cümle: {r[1]}\n"
                f"2. Cümle: {r[2]}\n"
                f"Cosine Similarity: {r[3]}\n"
                f"Dot Product: {r[4]}\n"
                f"Euclidean Distance: {r[5]}\n"
                f"BERTScore (F1): {r[6]}"
            )
            self.detail_text.setText(detail)

# Uygulama çalıştırma bloğu
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SentenceSimilarityApp()
    window.show()
    sys.exit(app.exec_())
