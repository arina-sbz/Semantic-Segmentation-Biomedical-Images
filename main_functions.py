import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset


class ConvNet(nn.Module):
    def __init__(self):
        """
        Convolutional Neural Network for image segmentation

        The network consists of an encoder and a decoder
        """
        super(ConvNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvNet_dropout(nn.Module):
    def __init__(self):
        """
        Convolutional Neural Network with Dropout

        The network consists of an encoder and a decoder with dropout layers
        """
        super(ConvNet_dropout, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.1),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvNet_batch(nn.Module):
    def __init__(self):
        """
        Convolutional Neural Network with Batch Normalization

        The network consists of an encoder and a decoder with batch normalization layers
        """
        super(ConvNet_batch, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        """
        U-Net architecture for image segmentation

        The network consists of an encoder and a decoder with skip connections
        """
        super(Unet, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.encoder_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.encoder_conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.decoder_conv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(
            32 + 32, 16, 4, stride=2, padding=1)  # Skip connection
        self.decoder_conv3 = nn.ConvTranspose2d(
            16 + 16, 1, 3, stride=1, padding=1)  # Skip connection

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder path
        encoder_1 = self.relu(self.encoder_conv1(x))
        p1 = self.pool(encoder_1)

        encoder_2 = self.relu(self.encoder_conv2(p1))
        p2 = self.pool(encoder_2)

        encoder_3 = self.relu(self.encoder_conv3(p2))

        # Decoder path with skip connections
        decoder_1 = self.relu(self.decoder_conv1(encoder_3))

        # Apply transpose convolution with skip connection and ReLU
        decoder_2 = self.relu(self.decoder_conv2(
            torch.cat([decoder_1, encoder_2], dim=1)))

        # Apply transpose convolution with skip connection
        output = self.decoder_conv3(torch.cat([decoder_2, encoder_1], dim=1))

        return output


class ResidualBlock(nn.Module):
    """
    Residual Block for ResNet

    The block consists of two convolutional layers with batch normalization and a skip connection
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Initialize the Residual Block

        Parameters:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the convolution
        downsample: Downsampling layer (Default is None)
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x  # Save the input for the skip connection

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # Add the skip connection
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet architecture for image segmentation

    The network consists of an initial convolutional layer, residual blocks, and a decoder.
    """

    def __init__(self):
        super(ResNet, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.res_block1 = ResidualBlock(16, 32, stride=1, downsample=nn.Sequential(
            nn.Conv2d(16, 32, 1, stride=1, bias=False),
            nn.BatchNorm2d(32)
        ))
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.res_block2 = ResidualBlock(32, 64, stride=1, downsample=nn.Sequential(
            nn.Conv2d(32, 64, 1, stride=1, bias=False),
            nn.BatchNorm2d(64)
        ))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.pool1(x)
        x = self.res_block2(x)
        x = self.decoder(x)
        return x


def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device, optimizer_type, use_scheduler, hasValidation=False):
    """
    Train the model

    Parameters:
        model: The model
        train_loader: DataLoader for the training data
        test_loader: DataLoader for the test data
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
        device: Device to train the model on
        optimizer_type: Type of optimizer
        use_scheduler: Whether to use a learning rate scheduler
        hasValidation: Whether to use a validation set

    Returns:
        train_losses: List of training losses for each epoch
        test_losses: List of testing losses for each epoch
        train_dice_scores: List of training dice scores for each epoch
        test_dice_scores: List of testing dice scores for each epoch
    """
    train_losses = []
    test_losses = []
    train_dice_scores = []
    test_dice_scores = []
    loss_function = nn.BCEWithLogitsLoss()
    
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if use_scheduler:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_dice_score = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, masks)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_dice_score += dice_score(masks, outputs)
        
        # Calculate average train loss and dice score
        train_length = len(train_loader)
        epoch_train_loss /= train_length
        epoch_train_dice_score /= train_length
        train_losses.append(epoch_train_loss)
        train_dice_scores.append(epoch_train_dice_score)
        
        # Evaluate on test/ validation set
        model.eval()
        epoch_test_loss = 0.0
        epoch_test_dice_score = 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = loss_function(outputs, masks)
                
                epoch_test_loss += loss.item()
                epoch_test_dice_score += dice_score(masks, outputs)
            
            test_length = len(test_loader)
            epoch_test_loss /= test_length
            epoch_test_dice_score /= test_length
            test_losses.append(epoch_test_loss)
            test_dice_scores.append(epoch_test_dice_score)
        
        if use_scheduler:
            scheduler.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, ' + (f'Validation Loss: {epoch_test_loss:.4f}' if hasValidation else f'Test Loss: {epoch_test_loss:.4f}') + f', Train Dice Score: {epoch_train_dice_score:.4f}')
 
    return train_losses, test_losses,train_dice_scores,test_dice_scores
             
def dice_score(true_labels, predicted_logits):
    """
    Calculate the Dice score

    Parameters:
        true_labels: Ground truth binary masks
        predicted_logits: Predicted logits

    Returns:
        Dice score
    """
    
    # Convert logits to probabilities using sigmoid
    predicted_probabilities = torch.sigmoid(predicted_logits)
    # Apply a threshold to convert probabilities to binary mask
    predicted_labels = (predicted_probabilities > 0.5).float()
    
    # Calculate the Dice score
    intersection = (predicted_labels * true_labels).sum()
    union = predicted_labels.sum() + true_labels.sum()
    
    # Add a small epsilon to avoid division by zero
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.item()

def test_model(model, test_loader, device):
    """
    Evaluate the model on the test set.

    Parameters:
        model: The neural network model
        test_loader: DataLoader for the test data
        device: Device to evaluate the model on

    Returns:
        avg_test_loss: The average loss on test set
        avg_dice_score: The average dice score on test set
        test_storage: Dictionary containing the following:
            'images': List of input images from the test set
            'masks': List of ground truth masks from the test set
            'outputs': List of predicted masks from the model
            'dice_scores': List of Dice scores for each image in the test set
    """
    model.eval()
    test_losses = []
    dice_scores = []
    loss_function = nn.BCEWithLogitsLoss()
    test_storage = {
        'images': [],
        'masks': [],
        'outputs': [],
        'dice_scores': []
    }
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = loss_function(outputs, masks)
            test_losses.append(loss.item())
            
            for i in range(images.size(0)):  # Iterate through each image in the batch
                dice_score_val = dice_score(masks[i].unsqueeze(0), outputs[i].unsqueeze(0))
                dice_scores.append(dice_score_val)
                test_storage['images'].extend(images.cpu().numpy())  # Move tensors to CPU and convert to numpy
                test_storage['masks'].extend(masks.cpu().numpy())
                test_storage['outputs'].extend(outputs.sigmoid().cpu().numpy())  # Apply sigmoid and convert to numpy
                test_storage['dice_scores'].append(dice_score_val)
    
    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_dice_score = sum(dice_scores) / len(dice_scores)
    print(f"Average Test Loss: {avg_test_loss:.4f}, Average Dice Score: {avg_dice_score:.4f}")

    return avg_test_loss, avg_dice_score, test_storage

def split_train_data(data, size = 0.2):
    """
    Split the dataset into training and validation subsets.

    Parameters:
        data: training dataset to be split
        size: Proportion of the dataset to be used as the validation set

    Returns:
        train_subset: Subset of the data to be used for training
        val_subset: Subset of the data to be used for validation
    """
    train_length = len(data)
    indices = list(range(train_length))
    split = int(np.floor(size * train_length))

    # Randomly shuffle indices
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    train_subset = Subset(data, train_idx)
    val_subset = Subset(data, val_idx)

    return train_subset, val_subset
    
def show_output(images, masks, outputs, index, title):
    """
    Display the input image, true mask, and predicted mask for a given index

    Parameters:
        images: Batch of input images
        masks: Batch of ground truth masks
        outputs: Batch of predicted masks
        index: Index of the sample to show
        title: Title of the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    
    axes[0].imshow(np.transpose(images[index], (1, 2, 0)))
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(masks[index].squeeze(), cmap='gray')
    axes[1].set_title('True Mask')
    axes[1].axis('off')
    
    # Apply a threshold to convert probabilities to binary mask
    predicted_mask = outputs[index].squeeze() > 0.5
    
    axes[2].imshow(predicted_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.savefig(f"{title}.png",dpi=300) 
    plt.show()

def training_curve_plot(title, subtitle,train_losses, test_losses, train_dice_scores, test_dice_scores, hasValidation = False):
    lg=13
    md=10
    sm=9
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=lg)
    plt.figtext(0.5, 0.90, subtitle, ha='center', fontsize=md, fontweight='normal') 
    x = range(1, len(train_losses)+1)
    axs[0].plot(x, train_losses, label=f'Final train loss: {train_losses[-1]:.4f}')
    axs[0].plot(x, test_losses, label=f'Final validation loss: {test_losses[-1]:.4f}' if hasValidation else f'Final test loss: {test_losses[-1]:.4f}')
    axs[0].set_title('Losses', fontsize=md)
    axs[0].set_xlabel('Epoch', fontsize=md)
    axs[0].set_ylabel('Loss', fontsize=md)
    axs[0].legend(fontsize=sm)
    axs[0].tick_params(axis='both', labelsize=sm)
    axs[0].grid(True, which="both", linestyle='--', linewidth=0.5)
    axs[1].plot(x, train_dice_scores, label=f'Final train score: {train_dice_scores[-1]:.4f}')
    axs[1].plot(x, test_dice_scores, label=f'Final validation score: {test_dice_scores[-1]:.4f}' if hasValidation else f'Final test score: {test_dice_scores[-1]:.4f}')
    axs[1].set_title('Dice Scores', fontsize=md)
    axs[1].set_xlabel('Epoch', fontsize=md)
    axs[1].set_ylabel('Dice Score', fontsize=sm)
    axs[1].legend(fontsize=sm)
    axs[1].tick_params(axis='both', labelsize=sm)
    axs[1].grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(f"{title}.png",dpi=300) 
    plt.show()
    
def display_img(input_tensor, title=None, ax=None, figsize=(5, 5), normalize=True):
    """
    Display an image from a tensor

    Parameters:
        input: The input image tensor.
        title: title for the image
        ax: Optional Matplotlib axis object to draw the image on
        figsize: Size of the figure
        normalize: Whether to normalize the image using mean and standard deviation
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    input = input_tensor.numpy().transpose((1, 2, 0))
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input = std * input + mean
        input = np.clip(input, 0, 1)
    ax.imshow(input)
    ax.axis('off')
    if title is not None:
        ax.set_title(title)

def show_augmented_images(data_loader, num_images=6):
    """
    Show a batch of augmented images and masks

    Parameters:
        data_loader: DataLoader providing the images and masks
        num_images: Number of images to show
    """
    data_iteretor = iter(data_loader)
    images, masks = next(data_iteretor)

    fig, axs = plt.subplots(2, num_images, figsize=(15, 5))
    for i in range(num_images):
        display_img(images[i], ax=axs[0, i])
        display_img(masks[i], ax=axs[1, i], normalize=False)
