import os
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Garante o determinismo matemático em todas as bibliotecas.
    Desativa o benchmark adaptativo do cuDNN para forçar reprodutibilidade na GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configurações do backend cuDNN para garantir multiplicações determinísticas
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Semente {seed} configurada. Ambiente determinístico!")


def init_weights(m: nn.Module, init_type: str = 'kaiming_normal', activation_type: str = 'relu') -> None:
    """
    Aplica inicializações de pesos específicas (Kaiming/He ou Xavier/Glorot)
    para evitar o desvanecimento ou explosão de gradientes.
    """
    if isinstance(m, nn.Linear):
        if init_type == 'kaiming_normal':
            nonlinearity = 'leaky_relu' if activation_type == 'leaky_relu' else 'relu'
            nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        elif init_type == 'kaiming_uniform':
            nonlinearity = 'leaky_relu' if activation_type == 'leaky_relu' else 'relu'
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(m.weight)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight)
        else:
            # Mantém a padrão do PyTorch (Kaiming Uniform por padrão)
            pass

        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class EarlyStopping:
    """
    Interrompe o treinamento se a perda de validação não melhorar após um
    determinado número de épocas (patience). Salva os melhores pesos via state_dict.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, checkpoint_path: str = 'best_model.pt', verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

        # Garante que o diretório pai exista
        parent_dir = os.path.dirname(checkpoint_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] Sem melhora na perda de validação por {self.counter}/{self.patience} épocas.")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model: nn.Module) -> None:
        if self.verbose:
            print(f"[EarlyStopping] Perda de validação melhorou para {self.best_loss:.6f}. Salvando checkpoint em '{self.checkpoint_path}'...")
        # Descompacta o modelo caso esteja envelopado em DataParallel
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), self.checkpoint_path)
