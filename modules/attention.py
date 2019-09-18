from functools import lru_cache
from typing import Optional
import torch
from torch import nn

def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result

class BottomUpTopDownAttention(nn.Module):
    r"""
    A PyTorch module to compute bottom-up top-down attention
    (`Anderson et al. 2017 <https://arxiv.org/abs/1707.07998>`_). Used in
    :class:`~updown.modules.updown_cell.UpDownCell`
    Parameters
    ----------
    query_size: int
        Size of the query vector, typically the output of Attention LSTM in
        :class:`~updown.modules.updown_cell.UpDownCell`.
    image_feature_size: int
        Size of the bottom-up image features.
    projection_size: int
        Size of the projected image and textual features before computing bottom-up top-down
        attention weights.
    """

    def __init__(self, query_size: int, image_feature_size: int, projection_size: int):
        super().__init__()

        self._query_vector_projection_layer = nn.Linear(query_size, projection_size, bias=False)
        self._image_features_projection_layer = nn.Linear(
            image_feature_size, projection_size, bias=False
        )
        self._attention_layer = nn.Linear(projection_size, 1, bias=False)

    def forward(
        self,
        query_vector: torch.Tensor,
        image_features: torch.Tensor,
        image_features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Compute attention weights over image features by applying bottom-up top-down attention
        over image features, using the query vector. Query vector is typically the output of
        attention LSTM in :class:`~updown.modules.updown_cell.UpDownCell`. Both image features
        and query vectors are first projected to a common dimension, that is ``projection_size``.
        Parameters
        ----------
        query_vector: torch.Tensor
            A tensor of shape ``(batch_size, query_size)`` used for attending the image features.
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes, image_feature_size)``. ``num_boxes`` for
            each instance in a batch might be different. Instances with lesser boxes are padded
            with zeros up to ``num_boxes``.
        image_features_mask: torch.Tensor
            A mask over image features if ``num_boxes`` are different for each instance. Elements
            where mask is zero are not attended over.
        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, num_boxes)`` containing attention weights for each
            image features of each instance in the batch. If ``image_features_mask`` is provided
            (for adaptive features), then weights where the mask is zero, would be zero.
        """

        # project the query_vector(output of attention LSTM) to has shape: (batch_size, projection_size)
        projected_query_vector = self._query_vector_projection_layer(query_vector)

        # Image features are projected by a method call, which is decorated using LRU cache, to
        # save some computation. Refer method docstring.
        # shape: (batch_size, num_boxes, projection_size)
        projected_image_features = self._project_image_features(image_features)

        # Broadcast query_vector as image_features for addition.
        # shape: (batch_size, num_boxes, projection_size)
        projected_query_vector = projected_query_vector.unsqueeze(1).repeat(
            1, projected_image_features.size(1), 1
        )

        # use a Linear layer to produce logits which used to calculate attention
        # shape: (batch_size, num_boxes, 1)
        attention_logits = self._attention_layer(
            torch.tanh(projected_query_vector + projected_image_features)
        )

        # shape: (batch_size, num_boxes)
        attention_logits = attention_logits.squeeze(-1)

        # `\alpha`s as importance weights for boxes (rows) in the `image_features`.
        # shape: (batch_size, num_boxes)
        if image_features_mask is not None:
            # we do not want count padding boxes into attention calculation
            attention_weights = masked_softmax(attention_logits, image_features_mask, dim=-1)
        else:
            attention_weights = torch.softmax(attention_logits, dim=-1)

        return attention_weights

    @lru_cache(maxsize=10)
    def _project_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        r"""
        Project image features to a common dimension for applying attention.
        Extended Summary
        ----------------
        For a single training/evaluation instance, the image features remain the same from first
        time-step to maximum decoding steps. To keep a clean API, we use LRU cache -- which would
        maintain a cache of last 10 return values because on call signature, and not actually
        execute itself if it is called with the same image features seen at least once in last
        10 calls. This saves some computation.
        Parameters
        ----------
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes, image_feature_size)``. ``num_boxes`` for
            each instance in a batch might be different. Instances with lesser boxes are padded
            with zeros up to ``num_boxes``.
        Returns
        -------
        torch.Tensor
            Projected image features of shape ``(batch_size, num_boxes, image_feature_size)``.
        """

        return self._image_features_projection_layer(image_features)