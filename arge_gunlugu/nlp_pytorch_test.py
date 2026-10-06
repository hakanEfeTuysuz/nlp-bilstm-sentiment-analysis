import torch

print("--- PyTorch & Donanım Kontrol Merkezi ---")
print(f"PyTorch Sürümü: {torch.__version__}")

# CUDA, NVIDIA ekran kartlarının yapay zeka ile konuşmasını sağlayan dildir
cuda_aktif_mi = torch.cuda.is_available()
print(f"CUDA (GPU Hızlandırması) Aktif mi?: {cuda_aktif_mi}")

if cuda_aktif_mi:
    ekran_karti_ismi = torch.cuda.get_device_name(0)
    print(f"Tespit Edilen Ekran Kartı: {ekran_karti_ismi}")
    
    # NumPy matrisinin PyTorch'taki karşılığı olan 'Tensor' oluşturuyoruz
    cpu_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print("\nCPU'daki Tensor (Tembel):")
    print(cpu_tensor)
    
    # Tensoru işlemciden alıp ekran kartının belleğine (VRAM) yolluyoruz
    gpu_tensor = cpu_tensor.to('cuda')
    print("\nGPU'ya Taşınan Tensor (Hızlı):")
    print(gpu_tensor)
else:
    print("\nUyarı: PyTorch şu an sadece İşlemciyi (CPU) görüyor.")
    print("Derin öğrenme eğitimleri daha yavaş çalışacaktır.")