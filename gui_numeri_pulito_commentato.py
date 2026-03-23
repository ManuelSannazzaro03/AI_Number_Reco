import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "modello_numeri.keras"

# Controlla che il modello esista davvero nella cartella corrente
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modello non trovato: {MODEL_PATH}")

# Carica il modello addestrato dal file
model = tf.keras.models.load_model(MODEL_PATH)

# Definisce una funzione che prepara l'immagine esterna come il dataset MNIST
def preprocess_image(image_path):

    img = Image.open(image_path)

    img = img.convert("L")    # Converte l'immagine in scala di grigi

    img = img.resize((28, 28)) # Ridimensiona l'immagine a 28x28 pixel, che è il formato atteso dal modello

    img_array = np.array(img).astype("float32") # Converte l'immagine in array NumPy

    if img_array.mean() > 127:
        img_array = 255.0 - img_array # Se lo sfondo è chiaro e la cifra è scura, inverte i colori

    img_array = img_array / 255.0 # Normalizza i pixel tra 0 e 1

    img_array = np.expand_dims(img_array, axis=0) # Aggiunge la dimensione batch
    
    img_array = np.expand_dims(img_array, axis=-1) # Aggiunge la dimensione del canale


    return img_array  

# Definisce una funzione che esegue la predizione della cifra
def predict_number(image_path):
    
    img_array = preprocess_image(image_path)

    pred = model.predict(img_array, verbose=0)[0]
    
    predicted_number = int(np.argmax(pred))  # Trova la cifra con probabilità più alta
    
    confidence = float(np.max(pred)) * 100  # Converte la probabilità massima in percentuale
    
    return predicted_number, confidence, pred  # Restituisce cifra prevista, confidenza e tutte le probabilità

# Definisce la funzione che apre il file scelto dall'utente
def open_image():
    
    file_path = filedialog.askopenfilename(       # Apre la finestra per scegliere un'immagine
        title="Seleziona immagine",
        filetypes=[("Immagini", "*.png *.jpg *.jpeg *.bmp")],
    )

    # Se l'utente annulla, esce subito dalla funzione
    if not file_path:
        return

    
    try:
        
        img = Image.open(file_path)

        # Riduce l'anteprima mantenendo le proporzioni
        img.thumbnail((220, 220))

        # Converte l'immagine in formato compatibile con tkinter
        img_tk = ImageTk.PhotoImage(img)

        # Aggiorna l'etichetta con l'anteprima
        image_label.config(image=img_tk, text="")

        # Mantiene un riferimento all'immagine per evitare che sparisca
        image_label.image = img_tk

        # Esegue la predizione sul file scelto
        number, confidence, probs = predict_number(file_path)

        # Mostra il risultato principale
        result_label.config(text=f"Numero previsto: {number}\nConfidenza: {confidence:.2f}%")

        # Costruisce il testo con le probabilità di tutte le cifre
        probs_text = "\n".join([f"{i}: {probs[i] * 100:.2f}%" for i in range(10)])

        # Mostra il dettaglio delle probabilità
        details_label.config(text=probs_text)

    # Se succede un errore, mostra un messaggio a schermo
    except Exception as e:
        messagebox.showerror("Errore", str(e))

# Crea la finestra principale
root = tk.Tk()

# Imposta il titolo della finestra
root.title("Riconoscimento Numeri")

# Imposta la dimensione della finestra
root.geometry("500x650")

# Impedisce il ridimensionamento manuale
root.resizable(False, False)

# Crea il titolo in alto
title_label = tk.Label(root, text="Seleziona una foto del numero", font=("Arial", 16, "bold"))

# Inserisce il titolo nella finestra con spazio verticale
title_label.pack(pady=15)

# Crea il pulsante per aprire un'immagine
btn = tk.Button(root, text="Apri immagine", font=("Arial", 12), command=open_image, width=20, height=2)

# Inserisce il pulsante nella finestra
btn.pack(pady=10)

# Crea l'area dove verrà mostrata l'anteprima
image_label = tk.Label(root, text="Nessuna immagine selezionata")

# Inserisce l'area immagine nella finestra
image_label.pack(pady=20)

# Crea l'etichetta del risultato finale
result_label = tk.Label(root, text="Qui comparirà il numero", font=("Arial", 18, "bold"))

# Inserisce l'etichetta del risultato nella finestra
result_label.pack(pady=20)

# Crea l'etichetta con le probabilità dettagliate
details_label = tk.Label(root, text="", font=("Courier", 11), justify="left")

# Inserisce il dettaglio delle probabilità nella finestra
details_label.pack(pady=10)

# Avvia il ciclo principale della finestra
root.mainloop()
