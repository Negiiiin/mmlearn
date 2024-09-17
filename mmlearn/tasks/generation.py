"""Generation task."""

import inspect
import itertools
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import lightning as L  # noqa: N812
import torch
import torch.distributed
import torch.distributed.nn
from hydra_zen import store
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import nn

from mmlearn.datasets.core import Modalities, find_matching_indices
from mmlearn.datasets.core.modalities import Modality
from mmlearn.modules.losses import CLIPLoss
from mmlearn.tasks.hooks import EvaluationHooks

@dataclass
class EvaluationSpec:
    """Specification for an evaluation task."""

    task: Any  # `EvaluationHooks` expected
    run_on_validation: bool = True
    run_on_test: bool = True
    

@dataclass
class ModuleKeySpec:
    """Module key specification for mapping modules to modalities."""

    encoder_key: Optional[str] = None
    head_key: Optional[str] = None
    postprocessor_key: Optional[str] = None


_unsupported_modality_error = (
    "Found unsupported modality `{}` in the input. Supported modalities are "
    f"{Modalities.list_modalities()}."
    "HINT: New modalities can be added with `Modalities.register_modality` method."
)  


@store(group="task", provider="mmlearn")
class Generation(L.LightningModule):
    def __init__(  # noqa: PLR0912, PLR0915
        self,
        encoders: Dict[str, nn.Module],
        optimizer: Optional[partial[torch.optim.Optimizer]] = None,
        lr_scheduler: Optional[
            Union[
                Dict[str, Union[partial[torch.optim.lr_scheduler.LRScheduler], Any]],
                partial[torch.optim.lr_scheduler.LRScheduler],
            ]
        ] = None,
        loss: Optional[CLIPLoss] = None,
        compute_validation_loss: bool = True,
        compute_test_loss: bool = True,
        evaluation_tasks: Optional[Dict[str, EvaluationSpec]] = None,
    )->None:
        """Initialize the module."""
        super().__init__()
        
        modality_module_mapping = {}
        
        for key in encoders:
            modality_module_mapping[key] = ModuleKeySpec(
                encoder_key=key,
                head_key=key,
                postprocessor_key=key,
            )
                
                
        modality_encoder_mapping: Dict[str, Optional[str]] = {}
        modality_head_mapping: Dict[str, Optional[str]] = {}
        modality_postprocessor_mapping: Dict[str, Optional[str]] = {}
        for modality_key, module_mapping in modality_module_mapping.items():
            if not Modalities.has_modality(modality_key):
                raise ValueError(_unsupported_modality_error.format(modality_key))
            modality_encoder_mapping[modality_key] = module_mapping.encoder_key
            modality_head_mapping[modality_key] = module_mapping.head_key
            modality_postprocessor_mapping[modality_key] = (
                module_mapping.postprocessor_key
            )
        
        self.encoders = nn.ModuleDict(
            {
                Modalities.get_modality(modality_key): encoders[encoder_key]
                for modality_key, encoder_key in modality_encoder_mapping.items()
                if encoder_key is not None
            }
        )
        
        self.loss_fn = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.compute_validation_loss = compute_validation_loss
        self.compute_test_loss = compute_test_loss
        
    def encode(
        self, inputs: Dict[Union[str, Modality], Any], modality: Modality
    ) -> torch.Tensor:
        """Encode the input values for the given modality.

        Parameters
        ----------
        inputs : Dict[Union[str, Modality], Any]
            Input values.
        modality : Modality
            The modality to encode.

        Returns
        -------
        torch.Tensor
            The encoded values for the specified modality.
        """
        output = self.encoders[modality](inputs)[0]

        if self.heads and modality in self.heads:
            output = self.heads[modality](output)

        if self.postprocessors and modality in self.postprocessors:
            output = self.postprocessors[modality](output)

        return output
    
    
    def forward(
        self, inputs: Dict[Union[str, Modality], Any]
    ) -> Dict[str, torch.Tensor]:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[Union[str, Modality], Any]
            The input tensors to encode.

        Returns
        -------
        Dict[str, torch.Tensor]
            The encodings for each modality.
        """
        outputs = {
            modality.embedding: self.encode(inputs, modality)
            for modality in self._available_modalities
        }

        if not all(
            output.size(-1) == list(outputs.values())[0].size(-1)
            for output in outputs.values()
        ):
            raise ValueError("Expected all model outputs to have the same dimension.")

        return outputs
    
    
    def _compute_loss(
        self,
        batch: Dict[Union[str, Modality], Any],
        batch_idx: int,
        outputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.loss_fn is None:
            return None

        contrastive_losses: list[torch.Tensor] = []
        for loss_pair in self.modality_loss_pairs:
            modality_a = Modalities.get_modality(loss_pair.modalities[0])
            modality_b = Modalities.get_modality(loss_pair.modalities[1])

            indices_a, indices_b = find_matching_indices(
                batch["example_ids"][modality_a],
                batch["example_ids"][modality_b],
            )
            if indices_a.numel() == 0 or indices_b.numel() == 0:
                continue

            contrastive_losses.append(
                self.loss_fn(
                    outputs[modality_a.embedding][indices_a],
                    outputs[modality_b.embedding][indices_b],
                )
                * loss_pair.weight
            )

        auxiliary_losses: list[torch.Tensor] = []
        if self.auxiliary_tasks:
            for task_name, task_spec in self.aux_task_specs.items():
                auxiliary_task_output = self.auxiliary_tasks[task_name].training_step(
                    batch, batch_idx
                )
                if isinstance(auxiliary_task_output, torch.Tensor):
                    auxiliary_task_loss = auxiliary_task_output
                elif isinstance(auxiliary_task_output, Mapping):
                    auxiliary_task_loss = auxiliary_task_output["loss"]
                else:
                    raise ValueError(
                        "Expected auxiliary task output to be a tensor or a mapping "
                        f"containing a 'loss' key, but got {type(auxiliary_task_output)}."
                    )

                auxiliary_losses.append(task_spec.loss_weight * auxiliary_task_loss)
                if self.log_auxiliary_tasks_loss:
                    self.log(
                        f"train/{task_name}_loss",
                        auxiliary_task_loss
                        if not self.fabric
                        else self.all_gather(
                            auxiliary_task_loss.clone().detach()
                        ).mean(),
                        sync_dist=True,
                    )

        return torch.stack(contrastive_losses + auxiliary_losses).sum()
    
    
    