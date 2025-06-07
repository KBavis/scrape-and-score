import pytest
import torch
from torch import nn
from unittest.mock import patch
from scrape_and_score.nnutils.optimization import (
    optimization_loop,
    train_loop,
    test_loop as run_test_loop
)


@pytest.fixture
def dummy_dataset():
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return torch.tensor([float(idx)]), torch.tensor([float(idx * 2)])

    return DummyDataset()


@pytest.fixture
def dummy_dataloader(dummy_dataset):
    return torch.utils.data.DataLoader(dummy_dataset, batch_size=10)


@pytest.fixture
def dummy_model():
    model = nn.Linear(1, 1)
    return model


def test_train_loop_runs(dummy_dataloader, dummy_model):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)
    device = "cpu"

    train_loop(dummy_dataloader, dummy_model, loss_fn, optimizer, device)

    # after training, parameters should have gradients zeroed
    for param in dummy_model.parameters():
        assert param.grad is None or torch.all(param.grad == 0)


def test_test_loop_correctness(dummy_dataloader, dummy_model):
    loss_fn = nn.MSELoss()
    device = "cpu"

    # act
    test_loss = run_test_loop(dummy_dataloader, dummy_model, loss_fn, device)

    assert isinstance(test_loss, float)
    assert test_loss >= 0


def test_optimization_loop_runs_and_early_stops(dummy_dataloader, dummy_model):
    device = "cpu"
    learning_rate = 0.01

    with patch("scrape_and_score.nnutils.optimization.train_loop") as mock_train_loop, patch("scrape_and_score.nnutils.optimization.test_loop") as mock_test_loop:
        # simulate decreasing losses with a plateau
        losses = [10.0, 9.0, 8.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]  
        mock_test_loop.side_effect = lambda *args, **kwargs: losses.pop(0) if losses else 7.0

        optimization_loop(dummy_dataloader, dummy_dataloader, dummy_model, device, learning_rate)

        assert mock_train_loop.call_count > 0
        assert mock_test_loop.call_count == mock_train_loop.call_count
        assert mock_test_loop.call_count <= 250


@patch("builtins.print")
def test_train_loop_batch_logging_and_loss_value(mock_print):
    X = torch.randn(20, 1)
    y = torch.randn(20)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = "cpu"

    train_loop(dataloader, model, loss_fn, optimizer, device)

    # ensure loss is logged
    assert any("loss:" in str(call.args[0]) for call in mock_print.call_args_list)


@patch("builtins.print")
def test_test_loop_tolerance_effect(mock_print):

    # create model mock that is only 2 away from actual value
    class ModelMock(nn.Module):
        def forward(self, x):
            return x + 2.0

    device = "cpu"
    model = ModelMock()

    X = torch.arange(10).float().unsqueeze(1)
    y = X.squeeze(1) + 2.0
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    loss_fn = nn.MSELoss()

    # tolerance = 0.5 means no correct predictions; tolerance = 3 means all correct
    loss = run_test_loop(dataloader, model, loss_fn, device, tolerance=0.5)
    assert loss >= 0

    loss = run_test_loop(dataloader, model, loss_fn, device, tolerance=3.0)
    assert loss >= 0

    # ensure testing accuracy logged
    assert any("Accuracy:" in str(call.args[0]) for call in mock_print.call_args_list)
