import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

def test_pytorch_cli():
    X = pd.DataFrame({
        'feature_1': [1, 4, 7],
        'feature_2': [2, 5, 8],
        'feature_3': [3, 6, 9]
    })
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 2)
        def forward(self, x):
            return self.fc(x)
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(torch.tensor(X.values, dtype=torch.float32))
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    torch.save(model, 'mock_model_pytorch.pt')
    X.to_csv('mock_data_pytorch.csv', index=False)
    # CLI command example (to run manually):
    # python cli_explain.py --model mock_model_pytorch.pt --data mock_data_pytorch.csv --method integrated_gradients
