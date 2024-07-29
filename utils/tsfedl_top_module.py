import torch

class TSFEDL_TopModule(torch.nn.Module):
    """
    Top module for TSFEDL

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        out_features: int
            Number of features of the output tensors

        npred: int
            Number of steps to forecast. Will be placed as 1 by default.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """

    def __init__(self, in_features=103, out_features=103, npred=1):
        super(TSFEDL_TopModule, self).__init__()
        self.npred = npred
        self.model = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=in_features, out_features=50),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=50, out_features=npred*out_features)
        )

    def forward(self, x):
        out = self.model(x)
        if len(out.shape)>2:
            out = out[:, -1, :]
        if self.npred > 1:
            # Reshape to (batch_size, npred, out_features)
            out = out.reshape(out.shape[0], self.npred, -1)
        return out