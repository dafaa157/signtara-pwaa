import tensorflow as tf

print("Memuat model .keras...")
# Buka model aslimu
model = tf.keras.models.load_model('model_signtara_bilstm.keras')

print("Menyimpan ulang sebagai .h5...")
# Simpan ulang dengan ekstensi .h5
model.save('model_signtara_bilstm.h5')

print("Selesai! File .h5 siap dikonversi ke TFJS.")