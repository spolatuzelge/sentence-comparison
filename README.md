
# 🧠 Sentence Similarity App

Bu proje, cümleler arasındaki **anlamsal benzerliği** ölçmek için geliştirilmiş bir uygulamadır. Kullanıcılar hem **manuel olarak iki cümle** girerek, hem de bir **CSV dosyasındaki cümle sütunları** üzerinden toplu olarak benzerlik karşılaştırması yapabilir.  
Karşılaştırmalar **SBERT** veya **SimCSE** modelleriyle gerçekleştirilir ve aşağıdaki metrikler sunulur:

- Cosine Similarity  
- Dot Product  
- Euclidean Distance  
- BERTScore (F1)

---

## 📸 Ekran Görüntüsü

<img src="e9c72331-e426-4674-b844-96803f424efc.png" alt="Uygulama Görseli" width="700"/>

---

## 🚀 Özellikler

- ✅ SBERT ve SimCSE model seçimi  
- ✅ Cümle benzerlik ölçümü (tekli veya çoklu)  
- ✅ Cosine, Dot, Euclidean ve BERTScore (F1) hesaplama  
- ✅ İlk 20 sonucu tablo olarak gösterme  
- ✅ Tüm sonuçları `.xlsx` olarak dışa aktarma  
- ✅ Alternatif olarak PyQt5 masaüstü uygulaması desteği  

---

## 🧰 Kullanılan Teknolojiler

- Flask & Jinja2  
- PyTorch & HuggingFace Transformers  
- Sentence-Transformers  
- BERTScore (`evaluate`)  
- scikit-learn, scipy, numpy  
- PyQt5 (desktop için)

---

## 📁 Proje Yapısı

```
.
├── Desktop/
│   └── main.py              # PyQt5 masaüstü uygulaması
├── web/
│   ├── app.py               # Flask tabanlı web uygulaması
│   ├── templates/
│   │   ├── manual.html      # Manuel giriş arayüzü
│   │   └── batch.html       # Toplu karşılaştırma arayüzü
│   └── static/
│       └── style.css        # Ortak stil dosyası
├── requirements.txt         # Bağımlılık listesi
└── README.md                # Proje açıklaması (bu dosya)
```

---

## ⚙️ Kurulum Adımları

### 1. Sanal Ortam Oluşturun (Opsiyonel ama önerilir)

```bash
python -m venv venv
source venv/bin/activate        # (Linux/Mac)
venv\Scripts\activate           # (Windows)
```

### 2. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 3. Flask Uygulamasını Başlatın

```bash
cd web
python app.py
```

Tarayıcıda açmak için: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧪 Kullanım

### 🔹 Manuel Karşılaştırma

- `/manual` sayfasına gidin  
- İki cümle girin  
- Modeli seçin  
- Benzerlik sonuçlarını inceleyin  

### 🔹 CSV ile Toplu Karşılaştırma

- `/batch` sayfasına gidin  
- CSV dosyanızı yükleyin  
- Cümle sütunlarını ve modeli seçin  
- İlk 20 sonucu tablo olarak görüntüleyin  
- Excel olarak çıktıyı indirin  

---

## 🖥️ Masaüstü Uygulama (GUI)

PyQt5 tabanlı masaüstü uygulamayı başlatmak için:

```bash
cd Desktop
python main.py
```

CSV dosyanızı yükleyerek aynı karşılaştırma işlemlerini masaüstü arayüzle gerçekleştirebilirsiniz.

---

## 📄 Lisans

Bu proje MIT lisansı ile lisanslanmıştır.
