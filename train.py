def train(net, num_epochs, optimizer, criterion, trainloader):
    net.train()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} Starting...")  # Debugging Print
        running_loss = 0.0  # Track loss per epoch

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {running_loss:.4f}")  # Print epoch loss

    print("Training Complete.")
