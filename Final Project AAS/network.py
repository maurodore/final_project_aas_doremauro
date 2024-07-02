from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, Rescaling, MaxPooling2D


class ResidualBlock(Layer):
    """
        Custom layer for a Residual Block used in deep convolutional networks.
        Implements the identity shortcut connection: output = input + residual.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(channels, 3, strides=1, padding='same', activation='relu')
        self.conv2 = Conv2D(channels, 3, strides=1, padding='same', activation='relu')

    def call(self, inputs):
        """
            Function to perform the computation associated with this layer.

            Args:
                inputs: Input tensor to the residual block.

            Returns:
                Tensor: Output tensor of the residual block.
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs


class ImpalaModelV2(Model):
    """
        ImpalaV2 Model is a deep convolutional neural network model which
        incorporate residual blocks and Convolutional layers for complex feature extraction.
    """
    def __init__(self, num_actions):
        super(ImpalaModelV2, self).__init__()

        # 16 ch.
        self.conv1 = Conv2D(16, kernel_size=(3, 3), strides=1, activation='relu')
        self.maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')
        self.residual_blocks1 = ResidualBlock(channels=16)
        self.residual_blocks2 = ResidualBlock(channels=16)

        # 32 ch.
        self.conv2 = Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu')
        self.maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')
        self.residual_blocks3 = ResidualBlock(channels=32)
        self.residual_blocks4 = ResidualBlock(channels=32)

        # 32 ch.
        self.conv3 = Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu')
        self.maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')
        self.residual_blocks5 = ResidualBlock(channels=32)
        self.residual_blocks6 = ResidualBlock(channels=32)

        self.flatten = Flatten()

        self.fc = Dense(256, activation='relu', )

        # Output layers
        self.action = Dense(num_actions, activation='softmax', name='action')
        self.value = Dense(1, activation=None, name='value')

    def call(self, inputs):
        """
            Defines the computation from inputs to outputs.

            Args:
                inputs: Input image tensor.

            Returns:
                tuple: Tuple containing action probabilities and state value estimate.
        """
        x = Rescaling(scale=1.0 / 255.0)(inputs)

        x1 = self.conv1(x)
        x1 = self.maxpool1(x1)
        x_residual1 = self.residual_blocks1(x1)
        x_residual2 = self.residual_blocks2(x_residual1)

        x2 = self.conv2(x_residual2)
        x2 = self.maxpool2(x2)
        x_residual3 = self.residual_blocks3(x2)
        x_residual4 = self.residual_blocks4(x_residual3)

        x3 = self.conv3(x_residual4)
        x3 = self.maxpool3(x3)
        x_residual5 = self.residual_blocks5(x3)
        x_residual6 = self.residual_blocks6(x_residual5)

        x = self.flatten(x_residual6)

        x = self.fc(x)

        return self.action(x), self.value(x)