# Gerekli kütüphanelerin içe aktarılması
from flask import Flask, render_template, request, redirect, url_for, send_file
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import numpy as np
import pandas as pd
import torch
import evaluate
import os
import io

# Flask uygulaması oluşturuluyor
app = Flask(__name__)

# Modeller yükleniyor
sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')  # Çok dilli SBERT modeli
simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")    # SimCSE için tokenizer
simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")            # SimCSE modeli
bertscore = evaluate.load("bertscore")  # BERTScore metriği

# Yüklenen dosyaların kaydedileceği klasör
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Sonuçların geçici saklanması için global değişken
downloadable_results = None

# SimCSE için embedding çıkarımı
def get_simcse_embedding(text):
    inputs = simcse_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = simcse_model(**inputs)
    return outputs.pooler_output[0]

# Model seçimine göre embedding döndürülmesi
def get_embedding(model_name, text):
    if model_name == "SBERT":
        return sbert_model.encode(text, convert_to_tensor=True)
    else:
        return get_simcse_embedding(text)

# Benzerlik metriklerinin hesaplanması
def compute_metrics(v1, v2):
    v1_np = v1.cpu().numpy()
    v2_np = v2.cpu().numpy()
    cosine = float(cosine_similarity([v1_np], [v2_np])[0][0])
    dot = float(np.dot(v1_np, v2_np))
    euclid = float(euclidean(v1_np, v2_np))
    return {
        "Cosine Similarity": round(cosine, 4),
        "Dot Product": round(dot, 4),
        "Euclidean Distance": round(euclid, 4),
    }

# Ana sayfa yönlendirmesi
@app.route('/')
def home():
    return redirect(url_for('manual'))

# Manuel giriş sayfası
@app.route('/manual', methods=['GET', 'POST'])
def manual():
    if request.method == 'POST':
        sentence1 = request.form['sentence1']
        sentence2 = request.form['sentence2']
        model_choice = request.form['model']

        emb1 = get_embedding(model_choice, sentence1)
        emb2 = get_embedding(model_choice, sentence2)

        # Sayısal metriklerin hesaplanması
        base_scores = compute_metrics(emb1, emb2)

        # BERTScore hesaplanması
        bert_result = bertscore.compute(predictions=[sentence2], references=[sentence1], lang="en")
        base_scores["BERTScore (F1)"] = round(bert_result["f1"][0], 4)

        # Sayfaya veri gönderimi
        return render_template("manual.html", sentence1=sentence1, sentence2=sentence2,
                               model_choice=model_choice, scores=base_scores)
    return render_template("manual.html")

# Toplu karşılaştırma sayfası
@app.route('/batch', methods=['GET', 'POST'])
def batch():
    global downloadable_results
    if request.method == 'POST':
        if 'csv_file' in request.files:
            file = request.files['csv_file']
            if file.filename.endswith('.csv'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                # Farklı encodinglerle CSV okunmaya çalışılır
                for encoding in ['utf-8', 'ISO-8859-9', 'windows-1254']:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', sep=None)
                        break
                    except Exception as e:
                        print(f"Encoding hatası ({encoding}): {e}")
                        continue

                # Sütun adları temizlenir
                df.columns = [col.strip() for col in df.columns]
                columns = df.columns.tolist()

                # Kullanıcıdan sütun seçim sayfasına geç
                return render_template("batch.html", columns=columns, filename=file.filename)

        elif 'col1' in request.form and 'col2' in request.form:
            filename = request.form.get('csv_filename')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # CSV tekrar okunur
            for encoding in ['utf-8', 'ISO-8859-9', 'windows-1254']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', sep=None)
                    break
                except Exception as e:
                    print(f"Encoding hatası ({encoding}): {e}")
                    continue

            df.columns = [col.strip() for col in df.columns]
            col1 = request.form.get('col1')
            col2 = request.form.get('col2')
            model_choice = request.form.get('model')

            df[col1] = df[col1].astype(str)
            df[col2] = df[col2].astype(str)

            # Sonuçları hesapla
            results = []
            for index, row in df.iterrows():
                try:
                    s1 = row[col1].strip()
                    s2 = row[col2].strip()
                    if s1 and s2:
                        emb1 = get_embedding(model_choice, s1)
                        emb2 = get_embedding(model_choice, s2)
                        metrics = compute_metrics(emb1, emb2)
                        bert = bertscore.compute(predictions=[s2], references=[s1], lang="en")
                        metrics["BERTScore (F1)"] = round(bert["f1"][0], 4)
                        results.append({"Index": index, col1: s1, col2: s2, **metrics})
                except Exception as e:
                    print(f"Hata (satır {index}): {e}")
                    continue

            # Sonuçlar geçici belleğe kaydedilir ve önizleme oluşturulur
            if results:
                full_df = pd.DataFrame(results)
                downloadable_results = full_df.copy()
                preview_df = full_df.head(20)
                html_table = preview_df.to_html(classes='table table-bordered', index=False)
            else:
                html_table = "<p>⚠️ Hiçbir karşılaştırma sonucu üretilemedi.</p>"

            return render_template("batch.html", table=html_table, columns=df.columns.tolist(), filename=filename)

    return render_template("batch.html")

# Sonuçların Excel olarak indirilmesi
@app.route('/download_excel')
def download_excel():
    global downloadable_results
    if downloadable_results is not None:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            downloadable_results.to_excel(writer, index=False, sheet_name='Sonuclar')
        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True, download_name='karsilastirma_sonuclari.xlsx')
    return "⚠️ Henüz karşılaştırma yapılmadı.", 400

# Flask uygulamasını başlat
if __name__ == '__main__':
    app.run(debug=True)
