import streamlit as st
import torch
import torch.nn as nn
import numpy as np

IMG_SIZE = 28
INPUT_DIM = IMG_SIZE * IMG_SIZE
NUM_CLASSES = 10
LATENT_DIM = 20

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim):
        super(ConditionalVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Tanh(),
        )
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x, c):
        x_flat = x.view(x.size(0), -1)
        c_one_hot = nn.functional.one_hot(c, num_classes=NUM_CLASSES).float()
        combined = torch.cat([x_flat, c_one_hot], dim=1)
        h = self.encoder(combined)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        z_combined = torch.cat([z, c_one_hot], dim=1)
        recon_x = self.decoder(z_combined)
        return recon_x.view_as(x), mu, log_var

@st.cache_resource
def load_model():
    model = ConditionalVAE(INPUT_DIM, NUM_CLASSES, LATENT_DIM)
    # Cargar los pesos entrenados. Asegúrate de que el archivo .pth esté en el repo.
    # 'map_location' asegura que funcione incluso si no tienes GPU.
    model.load_state_dict(torch.load('cvae_mnist_model.pth', map_location=torch.device('cpu')))
    model.eval() # Poner el modelo en modo de evaluación
    return model

model = load_model()

def generate_images(digit, num_images=5):
    with torch.no_grad():
        labels = torch.LongTensor([digit] * num_images)

        z = torch.randn(num_images, LATENT_DIM)

        c_one_hot = nn.functional.one_hot(labels, num_classes=NUM_CLASSES).float()
        z_combined = torch.cat([z, c_one_hot], dim=1)
        generated_images = model.decoder(z_combined).view(num_images, 28, 28)
        
        generated_images = (generated_images + 1) / 2
        return generated_images.numpy()

# --- Interfaz de Usuario de la App ---
st.set_page_config( layout="wide",
    initial_sidebar_state="expanded")
st.title("🖌️ Generador de Dígitos Manuscritos con IA")
st.write(
    "This app use a model trained on the MNIST dataset to generate new handwritten digit images."
    "Select a digit and click 'Generate'."
)

st.sidebar.header("Control Panel")
# User selects the digit
selected_digit = st.sidebar.selectbox("Select a digit (0-9):", list(range(10)))

# Button to start generation
if st.sidebar.button("✨ Generate 5 Images"):
    st.subheader(f"Generated Images for Digit: {selected_digit}")

    images = generate_images(selected_digit, num_images=5)

    cols = st.columns(5)
    for i, image_array in enumerate(images):
        with cols[i]:
            st.image(
                image_array, 
                caption=f"Img {i+1}", 
                width=150, 
                use_column_width='auto'
            )
else:
    st.info("Select a digit and press the 'Generate' button on the left panel.")