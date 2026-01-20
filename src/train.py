import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Training"):
        if len(batch) == 2:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
        else:
            # Paired: audio, face, label
            audio, face, labels = batch
            audio, face, labels = audio.to(device), face.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(audio, face)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if len(batch) == 2:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
            else:
                # Paired
                audio, face, labels = batch
                audio, face, labels = (
                    audio.to(device),
                    face.to(device),
                    labels.to(device),
                )
                outputs = model(audio, face)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def run_training(config, model, train_dataset, val_dataset):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    model = model.to(device)

    batch_size = config["model"]["batch_size"]
    lr = config["model"]["learning_rate"]
    epochs = config["model"]["epochs"]
    weight_decay = config["model"].get("weight_decay", 0.0)

    # Use num_workers (set to 0 for safety, but can try 2)
    # Actually, keep 0 for now to avoid fork issues on Mac unless imperative
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine Annealing (T_max = epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, verbose=False)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    output_dir = config["paths"]["outputs"]
    run_id = "latest_run" 
    run_dir = os.path.join(output_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Step Scheduler
        scheduler.step() # CosineAnnealing doesn't take metric


        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pt"))
            print("Saved Best Model")

    # Save Metrics
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(history, f)

    return history
