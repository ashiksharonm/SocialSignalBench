import torch
import torch.nn as nn
import time
import numpy as np
from src.models.baselines import AudioBaseline, FaceBaseline
from src.models.transfer import AudioResNet, AudioEfficientNet
from src.models.fusion import MultimodalFusion

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_inference(model, audio_input, face_input, device='cpu', n_iters=50):
    model = model.to(device)
    model.eval()
    audio_input = audio_input.to(device)
    face_input = face_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(audio_input, face_input)
            
    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(audio_input, face_input)
    end = time.time()
    
    return (end - start) / n_iters

def main():
    print("Benchmarking Models...")
    
    # Config
    n_classes = 4
    n_mfcc = 40
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Dummy Inputs (Batch Size 1 for Latency)
    bs = 1
    audio_dummy = torch.randn(bs, n_mfcc, 94) # (1, 40, 94)
    face_dummy = torch.randn(bs, 1, 128, 128) # (1, 1, 128, 128)
    
    results = []
    
    # 1. Baseline
    print("Initializing Baseline...")
    a_base = AudioBaseline(n_mfcc=n_mfcc, n_classes=n_classes)
    f_base = FaceBaseline(in_channels=1, n_classes=n_classes)
    m_base = MultimodalFusion(a_base, f_base, n_classes)
    
    results.append({
        "Model": "Baseline (Custom CNN)",
        "Params": count_parameters(m_base),
        "Latency (ms)": benchmark_inference(m_base, audio_dummy, face_dummy, device) * 1000
    })
    
    # 2. ResNet-18 (Current Best)
    print("Initializing ResNet-18...")
    a_res = AudioResNet(n_classes=n_classes, pretrained=False)
    f_res = AudioResNet(n_classes=n_classes, pretrained=False) # We use AudioResNet class for Face too (1->3 chan)
    m_res = MultimodalFusion(a_res, f_res, n_classes, audio_dim=512, face_dim=512)
    
    results.append({
        "Model": "ResNet-18 (Transfer)",
        "Params": count_parameters(m_res),
        "Latency (ms)": benchmark_inference(m_res, audio_dummy, face_dummy, device) * 1000
    })
    
    # 3. EfficientNet-B0
    print("Initializing EfficientNet-B0...")
    try:
        a_eff = AudioEfficientNet(n_classes=n_classes, pretrained=False)
        f_eff = AudioEfficientNet(n_classes=n_classes, pretrained=False)
        m_eff = MultimodalFusion(a_eff, f_eff, n_classes, audio_dim=1280, face_dim=1280)
        
        results.append({
            "Model": "EfficientNet-B0",
            "Params": count_parameters(m_eff),
            "Latency (ms)": benchmark_inference(m_eff, audio_dummy, face_dummy, device) * 1000
        })
    except Exception as e:
        print(f"Skipping EfficientNet: {e}")

    # Print Table
    print("\n" + "="*60)
    print(f"{'Model':<25} | {'Params':<12} | {'Latency (ms)':<15}")
    print("-" * 60)
    for res in results:
        print(f"{res['Model']:<25} | {res['Params']:<12,} | {res['Latency (ms)']:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
