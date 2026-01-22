import torch

def incremental_update(model, new_loader, cfg, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["incremental"]["learning_rate"])
    criterion = torch.nn.MSELoss()

    for epoch in range(cfg["incremental"]["num_epochs"]):
        model.train()
        for xb, yb in new_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model
