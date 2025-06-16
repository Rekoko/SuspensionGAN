import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import numpy as np
from tqdm import tqdm
from utils import extract_time, batch_generator, random_generator
import time

def stacked_rnn(num_layers, hidden_dim, return_sequences=True):
    return [
        layers.LSTM(hidden_dim, return_sequences=return_sequences) for _ in range(num_layers)
    ]

class StackedRNNModel(tf.keras.Model):
    def __init__(self, num_layers, hidden_dim, output_dim, activation='sigmoid'):
        super().__init__()
        self.rnn_layers = stacked_rnn(num_layers, hidden_dim)
        self.output_layer = tf.keras.layers.Dense(output_dim, activation=activation)

    def call(self, X, T):
        # X shape: [batch_size, max_seq_len, dim]
        mask = tf.sequence_mask(T, maxlen=tf.shape(X)[1])  # Create mask
        for lstm_layer in self.rnn_layers:
            X = lstm_layer(X, mask=mask)
        H = self.output_layer(X)
        return H
    
class Embedder(tf.keras.Model):
    def __init__(self, num_layers, hidden_dim, output_dim):
        super().__init__()
        self.rnn_layers = stacked_rnn(num_layers, hidden_dim)
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')

    def call(self, X, T):
        # X shape: [batch_size, max_seq_len, dim]
        mask = tf.sequence_mask(T, maxlen=tf.shape(X)[1])  # Create mask
        for lstm_layer in self.rnn_layers:
            X = lstm_layer(X, mask=mask)
        H = self.output_layer(X)
        return H

class Recovery(tf.keras.Model):
    def __init__(self, num_layers, hidden_dim, output_dim):
        super().__init__()
        self.rnn_layers = stacked_rnn(num_layers, hidden_dim)
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')

    def call(self, X, T):
        # X shape: [batch_size, max_seq_len, dim]
        mask = tf.sequence_mask(T, maxlen=tf.shape(X)[1])  # Create mask
        for lstm_layer in self.rnn_layers:
            X = lstm_layer(X, mask=mask)
        X_tilde = self.output_layer(X)
        return X_tilde

class Generator(tf.keras.Model):
    def __init__(self, num_layers, hidden_dim, output_dim):
        super().__init__()
        self.rnn_layers = stacked_rnn(num_layers, hidden_dim)
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')

        
class Supervisor(tf.keras.Model):
    def __init__(self, num_layers, hidden_dim, output_dim):
        super().__init__()
        self.rnn_layers = stacked_rnn(num_layers, hidden_dim)
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')

        
class Discriminator(tf.keras.Model):
    def __init__(self, num_layers, hidden_dim, output_dim):
        super().__init__()
        self.rnn_layers = stacked_rnn(num_layers, hidden_dim)
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')

        



def timegan (ori_data, parameters):
    
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
        
    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)   
    
    # Network Parameters
    hidden_dim   = parameters['hidden_dim'] 
    num_layers   = parameters['num_layer']
    iterations   = parameters['iterations']
    batch_size   = parameters['batch_size']
    module_name  = parameters['module'] 
    z_dim        = dim
    gamma        = 1

    X = ori_data
    T = max_seq_len
    Z = tf.random.normal(shape=(batch_size, seq_len, z_dim))

    embedder = StackedRNNModel(num_layers, hidden_dim, hidden_dim, activation='sigmoid')
    recovery = StackedRNNModel(num_layers, hidden_dim, dim, activation='sigmoid')
    generator = StackedRNNModel(num_layers, hidden_dim, hidden_dim, activation='sigmoid')
    supervisor = StackedRNNModel(num_layers - 1, hidden_dim, hidden_dim, activation='sigmoid')
    discriminator = StackedRNNModel(num_layers, hidden_dim, 1, activation=None)
        
    # Discriminator loss
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()

    # Loss for embedder and recovery
    optimizer_embedder_network = tf.keras.optimizers.Adam()

    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)

        # Convert to tensors if necessary
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        T_mb = tf.convert_to_tensor(T_mb, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            latent_space = embedder(X_mb, T_mb)
            X_tilde = recovery(latent_space, T_mb)
            loss_value = tf.reduce_mean(mse(X_mb, X_tilde))
    
        e_vars = embedder.trainable_variables
        r_vars = recovery.trainable_variables
        # Compute gradients
        grads = tape.gradient(loss_value, e_vars + r_vars)
        
        # Apply gradients
        optimizer_embedder_network.apply_gradients(zip(grads, e_vars + r_vars))
        
        # Logging
        if itt % 10 == 0:
            print('step:', itt, '/', iterations, ', e_loss:', np.round(np.sqrt(loss_value.numpy()), 4))

    print('Finish Embedding Network Training')
    
    # 2. Training only with supervised loss
    print('Start Training with Supervised Loss Only')
    optimizer_supervised = tf.keras.optimizers.Adam()

    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)    
        # Random vector generation   
        # Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        # Z = Z_mb

        # Convert to tensors if necessary
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        T_mb = tf.convert_to_tensor(T_mb, dtype=tf.int32)

        X = X_mb
        T = T_mb

        with tf.GradientTape() as tape:
            # Generate synthetic data
            H = embedder(X, T)
            H_hat_supervise = supervisor(H, T)

            G_loss_S = tf.reduce_mean(mse(H[:,1:,:], H_hat_supervise[:,:-1,:]))

        g_vars = generator.trainable_variables
        s_vars = supervisor.trainable_variables
        # Compute gradients
        grads = tape.gradient(G_loss_S, g_vars + s_vars)
        
        # Apply gradients
        optimizer_supervised.apply_gradients(zip(grads, g_vars + s_vars))

        # Train generator       
        # Checkpoint
        if itt % 10 == 0:
            print('step: '+ str(itt)  + '/' + str(iterations) +', s_loss: ' + str(np.round(np.sqrt(G_loss_S),4)) )

    print("Finish Training with Supervised Loss Only")

     # 3. Joint Training
    print('Start Joint Training')
    optimizer_e = tf.keras.optimizers.Adam()
    optimizer_d = tf.keras.optimizers.Adam()
    optimizer_g = tf.keras.optimizers.Adam()
    for itt in range(iterations):
        # Generator training (twice more than discriminator training)
        for kk in range(2):
            # Set mini-batch
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)               
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            X = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            T = tf.convert_to_tensor(T_mb, dtype=tf.int32)
            Z = tf.convert_to_tensor(Z_mb, dtype=tf.float32)


            with tf.GradientTape() as tape:
                H = embedder(X, T)
                
                E_hat = generator(Z, T)
                H_hat = supervisor(E_hat, T)
                Y_fake = discriminator(H_hat, T)

                X_hat = recovery(H_hat, T)

                G_loss_U = tf.reduce_mean(bce(tf.ones_like(Y_fake), Y_fake))
                G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
                G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
                G_loss_S = tf.reduce_mean(mse(H[:,1:,:], H_hat_supervise[:,:-1,:]))

                G_loss_V = G_loss_V1 + G_loss_V2

                G_loss = G_loss_U + G_loss_S + G_loss_V

            g_vars = generator.trainable_variables
            s_vars = supervisor.trainable_variables
            # Train Generator
            grads = tape.gradient(G_loss, g_vars + s_vars) # Removed G_loss_S due to no embedder usage
            optimizer_g.apply_gradients(zip(grads, g_vars + s_vars))

            with tf.GradientTape() as tape:
                H = embedder(X, T)
                X_tilde = recovery(H, T)
                    
                E_loss_T0 = tf.reduce_mean(mse(X, X_tilde))

            # Train Embedder
            grads = tape.gradient(E_loss_T0, e_vars + r_vars)
            optimizer_e.apply_gradients(zip(grads, e_vars + r_vars))

        # Discriminator training        
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)           
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

        X = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        T = tf.convert_to_tensor(T_mb, dtype=tf.int32)
        Z = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

        with tf.GradientTape() as tape:
            E_hat = generator(Z, T)
            H_hat = supervisor(E_hat, T)
            X_hat = recovery(H_hat, T)


            Y_fake = discriminator(H_hat, T)
            Y_real = discriminator(H, T)     
            Y_fake_e = discriminator(E_hat, T)

            # Discriminator loss
            D_loss_real = bce(tf.ones_like(Y_real), Y_real)
            D_loss_fake = bce(tf.zeros_like(Y_fake), Y_fake)
            D_loss_fake_e = bce(tf.zeros_like(Y_fake_e), Y_fake_e)
            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        d_vars = discriminator.trainable_variables
        if D_loss > 0.15:  # Only update if the discriminator is not working well
            grads = tape.gradient(D_loss, d_vars)
            optimizer_d.apply_gradients(zip(grads, d_vars))
            # Print multiple checkpoints
        if itt % 10 == 0:
            print('step: '+ str(itt) + '/' + str(iterations) + 
                    ', G_loss: ' + str(np.round(G_loss,4)) + 
                    ', e_loss_t0: ' + str(np.round(np.sqrt(E_loss_T0),4)) + 
                    ', d_loss: ' + str(np.round(D_loss,4)))
    print('Finish Joint Training')


    # TODO: Generate Sample Data
