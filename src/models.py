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
    ) -> None:
        """
        Initialize the EnStackModel.

        Args:
            model_name (str): HuggingFace model identifier or path.
            num_labels (int): Number of output labels for classification.
            pretrained (bool): Whether to load pretrained weights.
            dropout_rate (float): Dropout rate for the classifier head.
        """
        super(EnStackModel, self).__init__()

        self.model_name = model_name
        self.num_labels = num_labels

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
            labels=labels,
            return_dict=True,
        )

        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = outputs.loss

        return result

    def get_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts feature embeddings from the model for stacking.

        This method returns the [CLS] token embedding from the final hidden state.

        Args:
            input_ids (torch.Tensor): Token IDs (shape: [batch_size, seq_length]).
            attention_mask (torch.Tensor): Attention mask (shape: [batch_size, seq_length]).

        Returns:
            torch.Tensor: CLS token embeddings (shape: [batch_size, hidden_size]).
        """
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Extract [CLS] token representation (first token)
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
        model_name=hf_model_name, num_labels=num_labels, pretrained=pretrained
    )

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    logger.info(f"Created tokenizer for {hf_model_name}")

    return model, tokenizer
