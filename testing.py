# steel ned to work on it!

"""### Testing now:"""

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=testing_batches, num_workers=num_workers)
correct = 0
total = 0
loss = 0
with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss += criterion(outputs, labels)/len(val_loader)


logs_writer.add_scalar('Accuracy', {'Test':correct/total}, num_epoches)
print(f" ---> On Test data: \n    Loss: {loss}   Accuracy: {correct/total}")

logs_writer.close()