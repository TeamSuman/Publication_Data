import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class VAE(nn.Module):
    def __init__(self, input_dim, num_layers = 3,
                nodes=None, activation = "GELU", code = 2, learning_rate=0.00001, 
                batch_size=64, epochs=50, beta = 1.5, shuffle=True, thresh = 0.0001):

        """
        input_dim : flattened input vector length
        hidden1 : node number of hidden layer 1
        hidden2 : node number of hidden layer 2
        code : dimension of latent space
        learning_rate : name suggests
        thresh : thresh to compare while earlystopping
        train_data : trainning dataset
        val_data : validation dataset
        """
        
        if nodes is None:
            nodes = [128, 64, 16]
        super(VAE, self).__init__()
        # Setting seed
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.thresh = thresh
        self.code = code
        self.learning_rate = learning_rate
        self.nodes = nodes
        self.num_layers = num_layers

        self.activation = activation
        self.beta = beta
        self.learning_rate = learning_rate
        self.encoder = self.MakeEncoder(self.input_dim, self.num_layers, self.nodes, self.code, self.activation)
        self.mu = nn.Linear(self.nodes[-1], self.code)
        self.logvar = nn.Linear(self.nodes[-1], self.code)
        self.decoder = self.MakeDecoder(self.input_dim, self.num_layers, self.nodes, self.code, self.activation)

    ## encoder
    def MakeEncoder(self, input_dim, num_layers, nodes, code, activation = "GELU"):
        """
        Create an encoder neural network with specified architecture.

        Args:
            input_dim: The dimension of the input data.
            num_layers: The number of layers in the encoder.
            nodes: A list containing the number of nodes in each layer.
            code: The code dimension for the encoder.
            activation: The activation function to use (default is "GELU").

        Returns:
            The encoder neural network.
        """

        layers = self._extracted_from_MakeDecoder_2(input_dim, nodes, 0, activation)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(nodes[i], nodes[i+1]))
            if activation == "GELU":
                layers.append(nn.GELU())
            elif activation == "ReLU":
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        return self._extracted_from_MakeDecoder_17(layers) 
    ## decoder 
    def MakeDecoder(self, input_dim, num_layers, nodes, code, activation = "GELU"):
        layers = self._extracted_from_MakeDecoder_2(code, nodes, -1, activation)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(nodes[::-1][i], nodes[::-1][i+1]))
            if activation == "GELU":
                layers.append(nn.GELU())
            elif activation  == "ReLU":
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(nodes[::-1][-1], input_dim))
        return self._extracted_from_MakeDecoder_17(layers)

    # TODO: Rename this here and in `MakeEncoder` and `MakeDecoder`
    def _extracted_from_MakeDecoder_17(self, layers):
        encoder = nn.Sequential()
        for layer in layers:
            encoder.append(layer)
        return encoder

    # TODO: Rename this here and in `MakeEncoder` and `MakeDecoder`
    def _extracted_from_MakeDecoder_2(self, arg0, nodes, arg2, activation):
        result = [nn.Linear(arg0, nodes[arg2])]
        if activation == "GELU":
            result.append(nn.GELU())
        elif activation == "ReLU":
            result.append(nn.ReLU())
        result.append(nn.Dropout(0.2))
        return result
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, encoded
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mu, logvar, _ = self.encode(x)
        decoded = self.decode(z)
        return decoded, mu, logvar
    
    def fit(self, train_data = None, val_data = None):
        # sourcery skip: low-code-quality
        self.train_data = train_data
        self.val_data = val_data
        if torch.cuda.is_available():
            self.to('cuda')
        train_dataset = TensorDataset(torch.Tensor(self.train_data))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        val_dataset = TensorDataset(torch.Tensor(self.val_data))
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        best_val_loss = float('inf')
        prv = float('inf')
        stop_counter = 0
        for epoch in range(self.epochs):
            train_loss = 0
            for x_batch, in train_dataloader:
                x_batch = x_batch.to('cuda') if torch.cuda.is_available() else x_batch
                optimizer.zero_grad()
                decoded, mu, logvar = self(x_batch)
                recon_loss = criterion(decoded, x_batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                #_,code_space, _, _ = self.encode(x_batch)

                # TODO: change this to a more efficient code
                loss = recon_loss + (self.beta * kl_loss) #+ dev
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_loss = 0
            for x_batch, in val_dataloader:
                x_batch = x_batch.to('cuda') if torch.cuda.is_available() else x_batch
                decoded, mu, logvar = self(x_batch)
                recon_loss = criterion(decoded, x_batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + (self.beta * kl_loss)
                val_loss += loss.item()
            val_loss = val_loss / len(val_dataloader)
            if abs(val_loss - prv) < self.thresh:
                break
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                stop_counter = 0
                # You can save the best model using torch.save(self.state_dict(), 'best_model.pt')          
            else:
                stop_counter += 1
                if stop_counter >= 2:
                    print('Early stopping')
                    break
                    
            prv = val_loss

            print(f'Epoch: {epoch + 1}/{self.epochs} | '
                  f'Train Loss: {train_loss / len(train_dataloader):.5f} | '
                  f'Val Loss: {val_loss:.5f} | ')  