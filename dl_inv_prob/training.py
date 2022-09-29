import torch
import torch.optim
import torch.nn.functional as F


def check_loss(model, loss_fn, dataloader, device, mode="denoising"):
    """Compute the loss in evaluation mode."""
    loss = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if mode == "denoising":
                noisy, target = batch
                noisy, target = noisy.to(device), target.to(device)
                output = model(noisy)
                loss += loss_fn(output, target).item()
            elif mode == "denoising_inpainting":
                noisy, target, mask = batch
                noisy, target, mask = (
                    noisy.to(device),
                    target.to(device),
                    mask.to(device),
                )
                output = model(noisy)
                loss += loss_fn(mask * output, target).item()
            elif mode == "denoising_deblurring":
                noisy, target, blurr = batch
                noisy, target, blurr = (
                    noisy.to(device),
                    target.to(device),
                    blurr[0].to(device)[None, :],
                )
                output = model(noisy)
                loss += loss_fn(
                    F.conv2d(output, torch.flip(blurr, dims=[2, 3]), padding="same"),
                    F.conv2d(target, torch.flip(blurr, dims=[2, 3]), padding="same")
                ).item()
            else:
                loss += loss_fn(model, batch.to(device)).item()
    model.train()

    return loss / len(dataloader)


def train(
    model,
    loss_fn,
    train_dataloader,
    val_dataloader,
    mode="denoising",
    optimizer=None,
    scheduler=None,
    learning_rate=0.001,
    weight_decay=0.0,
    device="cpu",
    n_epochs=100,
    show_every=10,
):
    """Train the model."""
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.8,
            patience=50,
            verbose=True,
        )

    losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        for batch in train_dataloader:
            model.train()
            # Forward pass
            if mode == "denoising":
                noisy, target = batch
                noisy, target = noisy.to(device), target.to(device)
                output = model(noisy)
                loss = loss_fn(output, target)
            elif mode == "denoising_inpainting":
                noisy, target, mask = batch
                noisy, target, mask = (
                    noisy.to(device),
                    target.to(device),
                    mask.to(device),
                )
                output = model(noisy)
                loss = loss_fn(mask * output, target)
            elif mode == "denoising_deblurring":
                noisy, target, blurr = batch
                noisy, target, blurr = (
                    noisy.to(device),
                    target.to(device),
                    blurr[0].to(device)[None, :]
                )
                output = model(noisy)
                loss = loss_fn(
                    F.conv2d(output, torch.flip(blurr, dims=[2, 3]), padding="same"),
                    F.conv2d(target, torch.flip(blurr, dims=[2, 3]), padding="same")
                )
            else:
                loss = loss_fn(model, batch.to(device))
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        val_loss = check_loss(
            model, loss_fn, val_dataloader, device, mode=mode
        )
        val_losses.append(val_loss)

        scheduler.step(val_loss)

    return model, losses, val_losses
