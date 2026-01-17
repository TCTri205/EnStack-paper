"""
Model architectures for EnStack vulnerability detection.

This module provides the EnStackModel wrapper for transformer-based models
used in vulnerability detection tasks.
"""

import logging
from typing import Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    RobertaForSequenceClassification,
)

logger = logging.getLogger("EnStack")


class EnStackModel(nn.Module):
    """
    Wrapper model for transformer-based vulnerability detection.

    This class wraps HuggingFace transformer models (CodeBERT, GraphCodeBERT, UniXcoder)
    for sequence classification and feature extraction tasks.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_torch_compile: bool = False,
        torch_compile_mode: str = "default",
    ) -> None:
        """
        Initialize the EnStackModel.

        Args:
            model_name (str): HuggingFace model identifier or path.
            num_labels (int): Number of output labels for classification.
            pretrained (bool): Whether to load pretrained weights.
            dropout_rate (float): Dropout rate for the classifier head.
            label_smoothing (float): Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing).
                Helps prevent overfitting and improves generalization.
            class_weights (Optional[torch.Tensor]): Weights for each class to handle imbalance.
                Shape: [num_labels]. If None, all classes are weighted equally.
            use_gradient_checkpointing (bool): Enable gradient checkpointing to save VRAM.
                Trades computation for memory (slower but uses less VRAM).
                Recommended for long sequences (>512 tokens) or limited VRAM.
            use_torch_compile (bool): Enable torch.compile() for graph optimization.
                OPTIMIZATION: Provides 10-20% speedup on modern GPUs (PyTorch 2.0+).
                Modes: 'default' (balanced), 'reduce-overhead' (faster), 'max-autotune' (slowest compile, fastest run).
            torch_compile_mode (str): Compilation mode for torch.compile.
        """
        super(EnStackModel, self).__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

        # Load the transformer model for sequence classification
        if pretrained:
            self.model = RobertaForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                hidden_dropout_prob=dropout_rate,
                attention_probs_dropout_prob=dropout_rate,
            )
            logger.info(f"Loaded pretrained model: {model_name}")
        else:
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = num_labels
            config.hidden_dropout_prob = dropout_rate
            config.attention_probs_dropout_prob = dropout_rate
            self.model = RobertaForSequenceClassification(config)
            logger.info(f"Initialized model from config: {model_name}")

        # Store the base model for feature extraction
        self.base_model = self.model.roberta

        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (saves VRAM, slightly slower)")

        # OPTIMIZATION: Enable torch.compile for graph optimization (PyTorch 2.0+)
        if use_torch_compile:
            if hasattr(torch, "compile"):
                logger.info(
                    f"Compiling model with torch.compile (mode={torch_compile_mode})..."
                )
                self.model = torch.compile(self.model, mode=torch_compile_mode)
                logger.info(
                    "✅ torch.compile enabled - expect 10-20% speedup on modern GPUs"
                )
            else:
                logger.warning(
                    "⚠️ torch.compile not available (requires PyTorch 2.0+). "
                    "Upgrade PyTorch for performance boost: pip install --upgrade torch"
                )

        # Log configuration
        if label_smoothing > 0:
            logger.info(f"Label smoothing enabled: {label_smoothing}")
        if class_weights is not None:
            logger.info(f"Class weighting enabled: {class_weights.tolist()}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Token IDs (shape: [batch_size, seq_length]).
            attention_mask (torch.Tensor): Attention mask (shape: [batch_size, seq_length]).
            labels (Optional[torch.Tensor]): Target labels (shape: [batch_size]).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - loss: Cross-entropy loss (if labels provided)
                - logits: Prediction logits (shape: [batch_size, num_labels])
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # We'll compute loss manually to support label smoothing & class weights
            return_dict=True,
        )

        logits = outputs.logits
        result = {"logits": logits}

        if labels is not None:
            # Compute loss with label smoothing and class weights
            if self.label_smoothing > 0 or self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(
                    weight=(
                        self.class_weights.to(logits.device)
                        if self.class_weights is not None
                        else None
                    ),
                    label_smoothing=self.label_smoothing,
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                # Use default model loss
                outputs_with_loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs_with_loss.loss

            result["loss"] = loss

        return result

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str = "cls",
    ) -> torch.Tensor:
        """
        Extracts feature embeddings from the model for stacking.

        Args:
            input_ids (torch.Tensor): Token IDs (shape: [batch_size, seq_length]).
            attention_mask (torch.Tensor): Attention mask (shape: [batch_size, seq_length]).
            pooling (str): Pooling strategy - 'cls' or 'mean'.
                - 'cls': Uses [CLS] token embedding (default)
                - 'mean': Uses mean pooling over all tokens (excluding padding)

        Returns:
            torch.Tensor: Embeddings (shape: [batch_size, hidden_size]).
        """
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        if pooling == "mean":
            # Mean pooling: average all token embeddings, weighted by attention_mask
            token_embeddings = (
                outputs.last_hidden_state
            )  # [batch_size, seq_len, hidden_size]

            # Expand attention mask to match embedding dimensions
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )

            # Sum all token embeddings, then divide by the number of non-padding tokens
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            return cast(torch.Tensor, mean_embeddings)
        else:
            # CLS pooling: extract [CLS] token representation (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            return cast(torch.Tensor, cls_embedding)

    def get_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Gets prediction logits without computing loss.

        Args:
            input_ids (torch.Tensor): Token IDs (shape: [batch_size, seq_length]).
            attention_mask (torch.Tensor): Attention mask (shape: [batch_size, seq_length]).

        Returns:
            torch.Tensor: Prediction logits (shape: [batch_size, num_labels]).
        """
        outputs = self.forward(input_ids, attention_mask, labels=None)
        return outputs["logits"]

    def save_pretrained(self, save_path: str) -> None:
        """
        Saves the model to a directory.

        Args:
            save_path (str): Path to save directory.
        """
        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load_pretrained(cls, model_path: str, num_labels: int = 5) -> "EnStackModel":
        """
        Loads a pretrained model from a directory.

        Args:
            model_path (str): Path to the saved model directory.
            num_labels (int): Number of output labels.

        Returns:
            EnStackModel: Loaded model instance.
        """
        instance = cls(model_path, num_labels=num_labels, pretrained=True)
        logger.info(f"Model loaded from {model_path}")
        return instance

    def export_onnx(self, save_path: str, max_length: int = 512) -> None:
        """
        Exports the model to ONNX format for optimized inference.

        Args:
            save_path (str): Path to save the ONNX model.
            max_length (int): Maximum sequence length for dummy input.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Create dummy input
        dummy_input_ids = torch.zeros((1, max_length), dtype=torch.long).to(device)
        dummy_attention_mask = torch.ones((1, max_length), dtype=torch.long).to(device)

        # Export
        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask),
            save_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
        )
        logger.info(f"Model exported to ONNX: {save_path}")

    def export_torchscript(self, save_path: str, max_length: int = 512) -> None:
        """
        Exports the model to TorchScript format.

        Args:
            save_path (str): Path to save the TorchScript model.
            max_length (int): Maximum sequence length for tracing.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Create dummy input
        dummy_input_ids = torch.zeros((1, max_length), dtype=torch.long).to(device)
        dummy_attention_mask = torch.ones((1, max_length), dtype=torch.long).to(device)

        # Trace the model
        traced_model = torch.jit.trace(
            self.model, (dummy_input_ids, dummy_attention_mask), check_trace=False
        )
        traced_model.save(save_path)
        logger.info(f"Model exported to TorchScript: {save_path}")


def create_model(
    model_name: str, config: Dict, pretrained: bool = True
) -> Tuple[EnStackModel, AutoTokenizer]:
    """
    Factory function to create a model and its corresponding tokenizer.

    Args:
        model_name (str): Model identifier (e.g., 'codebert', 'graphcodebert').
        config (Dict): Configuration dictionary.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        Tuple[EnStackModel, AutoTokenizer]: Model and tokenizer instances.

    Raises:
        ValueError: If model_name is not found in the config model_map.
    """
    # Map short names to HuggingFace model identifiers
    model_map = config["model"].get("model_map", {})

    if model_name not in model_map:
        raise ValueError(
            f"Model '{model_name}' not found in config. "
            f"Available models: {list(model_map.keys())}"
        )

    hf_model_name = model_map[model_name]
    num_labels = config["model"].get("num_labels", 5)

    # Create model
    model = EnStackModel(
        model_name=hf_model_name,
        num_labels=num_labels,
        pretrained=pretrained,
        dropout_rate=config["model"].get("dropout_rate", 0.1),
        label_smoothing=config["model"].get("label_smoothing", 0.0),
        use_gradient_checkpointing=config["model"].get(
            "use_gradient_checkpointing", False
        ),
        use_torch_compile=config["model"].get("use_torch_compile", False),
        torch_compile_mode=config["model"].get("torch_compile_mode", "default"),
    )

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    logger.info(f"Created tokenizer for {hf_model_name}")

    return model, tokenizer
