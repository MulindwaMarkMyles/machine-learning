from sklearn.decomposition import PCA
import torch
import torch.nn as nn

class BaseNetwork(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(BaseNetwork, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

class LinearRegressionNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(LinearRegressionNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class LogisticRegressionNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(LogisticRegressionNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class TreeBasedNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(TreeBasedNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class SVMNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(SVMNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class KNNNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(KNNNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class GBMNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(GBMNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class XGBoostNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(XGBoostNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class DeepNeuralNet(BaseNetwork):
    def __init__(self, input_dim, embed_dim):
        super(DeepNeuralNet, self).__init__(input_dim, embed_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim//2),
            nn.LayerNorm(embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)

class PCANet(BaseNetwork):
    def __init__(self, input_dim, embed_dim, n_components=None):
        super(PCANet, self).__init__(input_dim, embed_dim)
        if n_components is None:
            n_components = min(input_dim, embed_dim)
        self.pca = PCA(n_components=n_components)
        self.fitted = False
        
        self.features = nn.Sequential(
            nn.Linear(n_components, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim//2),
            nn.LayerNorm(embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # Convert to numpy for PCA
        x_numpy = x.cpu().detach().numpy()
        
        if not self.fitted:
            desired_n_components = self.pca.n_components
            actual_n_components = min(x_numpy.shape[0], x_numpy.shape[1], desired_n_components)
            if actual_n_components < desired_n_components:
                print(f"Warning: PCA n_components={desired_n_components} is too large, reducing to {actual_n_components}")
                self.pca = PCA(n_components=actual_n_components)
                self.features[0] = nn.Linear(actual_n_components, self.embed_dim)
                self.features.to(x.device)
            self.pca.fit(x_numpy)
            self.fitted = True
        
        # Transform the data using PCA
        x_pca = self.pca.transform(x_numpy)
        
        # Convert back to tensor
        x_pca = torch.FloatTensor(x_pca).to(x.device)
        
        return self.features(x_pca)
